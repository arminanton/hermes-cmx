"""Query expansion (iteration-2, lever 3).

Two expanders, both returning extra query strings the retriever fuses in:
  • keyword_expander — deterministic: strip question-words/stopwords to a de-noised
    keyword query (helps lexical retrievers focus on content terms).
  • HydeExpander     — HyDE: ask a cheap model for a one-sentence *hypothetical answer*
    and search/embed THAT (a hypothetical answer is semantically closer to the verbatim
    evidence sentence than the question is).
"""
from __future__ import annotations

import re
from typing import List

_QWORDS = {
    "what", "when", "where", "who", "whom", "why", "how", "which", "did", "do", "does",
    "is", "are", "was", "were", "the", "a", "an", "of", "to", "in", "on", "for", "with",
    "and", "or", "that", "this", "we", "you", "i", "they", "he", "she", "it", "about",
    "had", "has", "have", "been", "would", "could", "should", "her", "his", "their", "them",
}
_TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_./:-]*")


def keyword_expander(q: str) -> List[str]:
    words = [w for w in _TOK.findall(q) if w.lower() not in _QWORDS and len(w) >= 3]
    kw = " ".join(words)
    return [kw] if kw and kw.lower() != q.lower() else []


class HydeExpander:
    def __init__(self, client, model: str, max_tokens: int = 50):
        self.client, self.model, self.max_tokens = client, model, max_tokens

    def __call__(self, q: str) -> List[str]:
        sys = {"role": "system", "content":
               "Write ONE short plausible ANSWER sentence to the question, as if recalling it "
               "from a personal conversation. Invent concrete specifics (names, dates, places). "
               "Output only the sentence."}
        try:
            hypo = self.client.complete([sys, {"role": "user", "content": q}],
                                        model=self.model).content
        except Exception:
            return []
        hypo = (hypo or "").strip()
        return [hypo] if hypo and not hypo.startswith("(model error") else []


class ChainExpander:
    def __init__(self, *expanders):
        self.expanders = expanders

    def __call__(self, q: str) -> List[str]:
        out: List[str] = []
        for e in self.expanders:
            try:
                out.extend(e(q))
            except Exception:
                pass
        return out
