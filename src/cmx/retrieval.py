"""Hybrid retrieval core — the floor of the whole design.

Fuses FTS5 (word/BM25) ⊕ trigram (substring/identifier) [⊕ embeddings, optional]
via Reciprocal-Rank Fusion, then applies bounded recency + pin priors. Returns
verbatim slices carrying their message id so injected evidence is citable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .config import CmxConfig
from .store import VerbatimStore


@dataclass
class Slice:
    id: int
    content: str
    score: float
    role: str
    turn_index: int
    sources: tuple  # which retrievers hit it: ("fts","trgm","emb")

    def as_evidence(self) -> dict:
        return {"id": self.id, "role": self.role, "turn_index": self.turn_index,
                "content": self.content}


def _rrf(rank_lists: list[list[int]], rrf_k: int) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ids in rank_lists:
        for rank, _id in enumerate(ids, start=1):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (rrf_k + rank)
    return scores


class HybridRetriever:
    def __init__(self, store: VerbatimStore, config: CmxConfig,
                 embedder: Optional[Callable[[str], list[float]]] = None,
                 expander: Optional[Callable[[str], list]] = None,
                 reranker: Optional[Callable[[str, list], list]] = None):
        self.store = store
        self.cfg = config
        self.embedder = embedder  # injection point; None => embeddings disabled
        self.expander = expander  # query expansion (lever 3); None => off
        self.reranker = reranker  # iter3 L4: LLM relevance reranker; None => off

    def retrieve(self, query: str, *, k: Optional[int] = None,
                 session_id: Optional[str] = None) -> list[Slice]:
        k = k or self.cfg.k_default
        pool = max(k * 4, 20)

        queries = [query]
        if self.expander is not None and query.strip():
            try:
                for q in self.expander(query):
                    if q and q not in queries:
                        queries.append(q)
            except Exception:
                pass

        rank_lists: list[list[int]] = []
        which: dict[int, set] = {}
        for q in queries:
            parts = [("fts", [h["id"] for h in self.store.search_fts(q, pool, session_id)])]
            if self.cfg.use_trigram:
                parts.append(("trgm", [h["id"] for h in self.store.search_trgm(q, pool, session_id)]))
            if self._emb_enabled():
                parts.append(("emb", self._embed_search(q, pool, session_id)))
                if self.cfg.use_chunks:
                    parts.append(("chunk", self._chunk_embed_search(q, pool, session_id)))
            if self._graph_enabled():
                parts.append(("graph", self._graph_search(q, pool, session_id)))
            for name, lst in parts:
                if lst:
                    rank_lists.append(lst)
                    for _id in lst:
                        which.setdefault(_id, set()).add(name)

        if not rank_lists:
            return []
        fused = _rrf(rank_lists, self.cfg.rrf_k)

        max_turn = self.store.max_turn(session_id) if session_id else None
        tdir = (self.cfg.temporal_rerank and bool(query))  # compute temporal signal?
        slices: list[Slice] = []
        for _id, base in fused.items():
            row = self.store.get_message(_id)
            if not row:
                continue
            score = base
            if max_turn and max_turn > 0:
                score += self.cfg.recency_weight * (row["turn_index"] / max_turn)
            if row.get("pinned"):
                score += self.cfg.pin_boost
            # iter4 lever: query-aware temporal re-ranking.
            if tdir and max_turn:
                from .rerank_signals import temporal_signal
                score += self.cfg.temporal_weight * temporal_signal(
                    query, row["turn_index"], max_turn, row["content"])
            # iter4 lever: importance weighting.
            if self.cfg.importance_rerank:
                from .rerank_signals import importance_signal
                score += self.cfg.importance_weight * importance_signal(row["content"])
            slices.append(Slice(id=_id, content=row["content"], score=score,
                                role=row["role"], turn_index=row["turn_index"],
                                sources=tuple(which.get(_id, ()))))
        if self.cfg.rerank and self._emb_enabled() and slices and session_id is not None:
            self._rerank_by_cosine(query, slices, session_id)
        slices.sort(key=lambda s: s.score, reverse=True)
        # iter3 L4: LLM relevance reranker over the fused pool — promotes model-judged
        # relevant slices into the top-k (a prefix of explicit indices). Degrades to the
        # score order on any failure.
        if self.reranker is not None and self.cfg.llm_rerank and len(slices) > 1:
            try:
                pool = slices[: max(k * 3, 20)]
                order = self.reranker(query, [s.content for s in pool])
                if order:
                    picked = [pool[i] for i in order if 0 <= i < len(pool)]
                    rest = [s for s in slices if s not in picked]
                    slices = picked + rest
            except Exception:
                pass
        return slices[:k]

    def _rerank_by_cosine(self, query: str, slices: list, session_id) -> None:
        """Lever 4: re-score the fused pool by cosine(query_embedding, candidate) so
        semantically-best evidence is promoted into the top-k (reuses stored vectors)."""
        import numpy as np
        ids, mat = self.store.get_session_vectors(session_id)
        if not ids:
            return
        idx = {i: j for j, i in enumerate(ids)}
        try:
            qv = np.asarray(self.embedder.embed([query])[0], dtype="float32")
        except Exception:
            return
        qn = qv / (np.linalg.norm(qv) + 1e-8)
        Mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
        for s in slices:
            j = idx.get(s.id)
            if j is not None:
                s.score += self.cfg.rerank_weight * float(Mn[j] @ qn)

    # -- embeddings (optional, degradable) --------------------------------
    def _emb_enabled(self) -> bool:
        return bool(self.cfg.use_embeddings and self.embedder is not None)

    def _embed_search(self, query: str, k: int, session_id) -> list[int]:
        emb = self.embedder
        if emb is None or session_id is None:
            return []
        try:
            # backfill embeddings for any new messages (batched), then query kNN
            missing = self.store.messages_missing_embeddings(session_id)
            if missing:
                vecs = emb.embed([c for _, c in missing])
                self.store.store_embeddings(
                    zip([i for i, _ in missing], vecs),
                    model=getattr(emb, "model", ""), dim=getattr(emb, "dim", 0))
            ids, matrix = self.store.get_session_vectors(session_id)
            if not ids:
                return []
            qv = emb.embed([query])[0]
            from .embeddings import cosine_topk
            return cosine_topk(qv, ids, matrix, k)
        except Exception:
            return []  # embeddings are an accelerator — never let them break retrieval

    def _chunk_embed_search(self, query: str, k: int, session_id) -> list[int]:
        """Lever 3 (iter3): kNN over CHUNK embeddings — a fact buried in a long turn embeds
        cleanly as its own chunk (the whole-turn average would dilute it). Returns parent
        message ids (deduped) so a chunk hit promotes its turn into the fused pool."""
        emb = self.embedder
        if emb is None or session_id is None:
            return []
        try:
            from .embeddings import chunk_text, cosine_topk_dedup
            missing = self.store.messages_missing_chunk_embeddings(session_id, self.cfg.chunk_min_chars)
            for pid, content in missing:
                chunks = chunk_text(content, self.cfg.chunk_size_chars)
                if not chunks:
                    continue
                vecs = emb.embed(chunks)
                self.store.store_chunk_embeddings(
                    session_id, [(pid, v) for v in vecs],
                    model=getattr(emb, "model", ""), dim=getattr(emb, "dim", 0))
            pids, matrix = self.store.get_session_chunk_vectors(session_id)
            if not pids:
                return []
            qv = emb.embed([query])[0]
            return cosine_topk_dedup(qv, pids, matrix, k)
        except Exception:
            return []

    # -- graph multi-hop (P5) --------------------------------------------
    def _graph_enabled(self) -> bool:
        return bool(self.cfg.use_graph and getattr(self.store, "graph", None) is not None)

    def _graph_search(self, query: str, k: int, session_id) -> list[int]:
        g = self.store.graph
        seeds = g.entities_in(query)
        if not seeds:
            return []
        ids = g.multihop_messages(seeds, self.cfg.graph_hops)
        if session_id is not None:
            ids = [m for m in ids if self.store.in_session(m, session_id)]
        return ids[:k]
