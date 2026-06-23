"""Iteration-3 L4: LLM-relevance reranker.

The iter-2 reranker scores candidates by bi-encoder cosine (query vs stored vector). A
cross-encoder / LLM relevance judge reads the (query, candidate) PAIR and can capture
relevance a bi-encoder misses. Batched into ONE call: the model returns the indices of the
most-relevant candidates, best-first; we promote those into the top-k.
"""
from __future__ import annotations

import re
from typing import List

_NUM = re.compile(r"\d+")


class LLMReranker:
    def __init__(self, client, model: str, pool: int = 20, max_tokens: int = 60):
        self.client, self.model, self.pool, self.max_tokens = client, model, pool, max_tokens

    def __call__(self, query: str, candidates: List[str]) -> List[int]:
        """Return candidate indices ranked most→least relevant (a prefix is enough)."""
        cand = candidates[: self.pool]
        if not cand:
            return []
        listing = "\n".join(f"[{i}] {c[:240]}" for i, c in enumerate(cand))
        sys = ("You rank candidate excerpts by how well each ANSWERS the QUESTION. "
               "Return ONLY a comma-separated list of candidate indices, MOST relevant "
               "first, including only those that could contain the answer. No prose.")
        msgs = [{"role": "system", "content": sys},
                {"role": "user", "content": f"QUESTION: {query}\n\nCANDIDATES:\n{listing}"}]
        try:
            out = self.client.complete(msgs, model=self.model).content or ""
        except Exception:
            return []
        seen, order = set(), []
        for tok in _NUM.findall(out):
            i = int(tok)
            if 0 <= i < len(cand) and i not in seen:
                seen.add(i)
                order.append(i)
        return order
