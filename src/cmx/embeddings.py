"""Embeddings backend for semantic retrieval (iteration-2, lever 1).

Injectable like the model client so it is import-safe and unit-testable:
  • HermesEmbedder — real vectors via the Hermes aux client's embeddings endpoint
    (copilot ``text-embedding-3-small``, 1536-dim, batched).
  • MockEmbedder   — deterministic bag-of-words vectors for tests (no network).
Vectors are stored in the cmx store and searched with numpy brute-force cosine kNN
(sqlite-vec isn't available on this host; brute force is fine at session scale).
"""
from __future__ import annotations

import hashlib
from typing import List, Protocol

import numpy as np


class Embedder(Protocol):
    dim: int
    def embed(self, texts: List[str]) -> List[List[float]]: ...


class HermesEmbedder:
    dim = 1536

    def __init__(self, model: str = "text-embedding-3-small", task: str = "compression",
                 batch: int = 64):
        self.model, self.task, self.batch = model, task, batch

    def embed(self, texts: List[str]) -> List[List[float]]:
        from agent.auxiliary_client import get_text_auxiliary_client
        client, _ = get_text_auxiliary_client(self.task, main_runtime=None)
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch):
            chunk = [t if t else " " for t in texts[i:i + self.batch]]
            r = client.embeddings.create(model=self.model, input=chunk)
            out.extend(list(d.embedding) for d in r.data)
        return out


class MockEmbedder:
    """Deterministic, network-free embedder for tests: hashed bag-of-words."""
    def __init__(self, dim: int = 128):
        self.dim = dim

    def _h(self, w: str) -> int:
        return int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype="float32")
            for w in (t or "").lower().split():
                v[self._h(w)] += 1.0
            out.append(v.tolist())
        return out


def pack(vec: List[float]) -> bytes:
    return np.asarray(vec, dtype="float32").tobytes()


def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="float32")


def cosine_topk(query_vec: List[float], ids: List[int], matrix: np.ndarray, k: int):
    if matrix.size == 0:
        return []
    q = np.asarray(query_vec, dtype="float32")
    q = q / (np.linalg.norm(q) + 1e-8)
    M = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    sims = M @ q
    order = np.argsort(-sims)[:k]
    return [ids[i] for i in order]


def cosine_topk_dedup(query_vec: List[float], ids: List[int], matrix: np.ndarray, k: int):
    """Like cosine_topk but ids may repeat (e.g. several chunks per parent message);
    returns the top-k DISTINCT ids by their best-scoring row, preserving rank order."""
    if matrix.size == 0:
        return []
    q = np.asarray(query_vec, dtype="float32")
    q = q / (np.linalg.norm(q) + 1e-8)
    M = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    sims = M @ q
    out, seen = [], set()
    for i in np.argsort(-sims):
        _id = ids[i]
        if _id in seen:
            continue
        seen.add(_id)
        out.append(_id)
        if len(out) >= k:
            break
    return out


def chunk_text(text: str, size: int = 400, overlap: int = 80) -> List[str]:
    """Split long text into overlapping passages at sentence-ish boundaries, so each chunk
    embeds cleanly (a fact buried in a long turn is not diluted by the whole-turn average)."""
    text = (text or "").strip()
    if len(text) <= size:
        return [text] if text else []
    import re
    sents = re.split(r"(?<=[.!?])\s+|\n+", text)
    chunks, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > size:
            chunks.append(cur.strip())
            cur = (cur[-overlap:] + " " + s) if overlap else s
        else:
            cur = (cur + " " + s) if cur else s
    if cur.strip():
        chunks.append(cur.strip())
    # hard-split any oversized chunk (no sentence breaks)
    out = []
    for c in chunks:
        if len(c) <= size * 2:
            out.append(c)
        else:
            for i in range(0, len(c), size):
                out.append(c[i:i + size])
    return [c for c in out if c.strip()]
