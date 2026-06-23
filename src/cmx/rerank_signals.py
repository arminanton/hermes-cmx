"""Temporal + importance re-ranking signals (Dakera-style levers, ported deterministically).

cmx's hybrid retrieval fuses lexical+semantic by RELEVANCE but is temporally and
importance-blind. Dakera's two winning levers:
  * temporal re-ranking  — re-weight candidates by how their age matches the QUERY's
    temporal intent ("recent X" → prefer late turns; "first/originally X" → prefer early;
    an explicit year → prefer memories near it). Their hardest category (temporal) gain.
  * importance weighting  — a critical memory (a decision/commitment/fact) outranks an
    incidental mention even at equal relevance.

Both are cheap, deterministic, no model calls — consistent with cmx's "engine, not model"
philosophy. Pure functions; signals in [-1,1] / [0,1] the retriever scales by a config weight.
"""
from __future__ import annotations

import re

# Query temporal-intent cues.
_RECENT = re.compile(r"\b(recent|recently|lately|latest|last|just|now|currently|"
                     r"these days|nowadays|most recent|newest|this (week|month|year)|today)\b",
                     re.IGNORECASE)
_EARLY = re.compile(r"\b(first|initially|originally|at first|early on|back then|used to|"
                    r"the beginning|started|originally|earliest|once|long ago)\b",
                    re.IGNORECASE)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

# Importance cues: decisions / commitments / state changes (the high-value memories).
_DECISION = re.compile(
    r"\b(decid|chose|choose|agreed|agree|commit|promis|plan|will|going to|booked|bought|"
    r"buy|moved|move|started|start|quit|signed|accepted|reject|approv|cancel|switched|"
    r"changed|set up|scheduled|deadline|due|launch|ship|deploy|hired|fired|married|"
    r"divorced|adopt|prefer|favorite|favourite|allerg|diagnos)\w*", re.IGNORECASE)
# Salient fact surface (numbers, ids, dates) — the high-recall-value tokens.
_FACT = re.compile(r'"[^"]+"|\b[\w/:.-]*\d[\w/:.-]*\b|\b[A-Z][a-zA-Z]+\b')
# First-person factual self-statements ("I am / my / I have a ...").
_SELF = re.compile(r"\b(I am|I'm|my|I have|I've|I had|I was|mine)\b", re.IGNORECASE)


def temporal_direction(query: str) -> int:
    """Return the query's temporal intent: +1 prefer-recent, -1 prefer-early, 0 none.
    Recent wins ties (questions skew toward the latest state)."""
    if not query:
        return 0
    recent = bool(_RECENT.search(query))
    early = bool(_EARLY.search(query))
    if recent and not early:
        return 1
    if early and not recent:
        return -1
    return 0


def query_years(query: str) -> set[str]:
    """Explicit years referenced in the query (for date-anchored re-ranking)."""
    return set(_YEAR.findall(query)) if query else set()


def temporal_signal(query: str, turn_index: int, max_turn: int, content: str = "") -> float:
    """Signal in [-1,1]: how well a candidate's age matches the query's temporal intent.

    * directional intent → late turns get +pos for 'recent', early turns +pos for 'first'.
    * explicit year in query AND in the candidate content → strong + boost (date anchor).
    """
    if max_turn <= 0:
        return 0.0
    pos = (turn_index / max_turn)  # 0=oldest .. 1=newest
    sig = 0.0
    direction = temporal_direction(query)
    if direction == 1:
        sig += (pos - 0.5) * 2.0          # recent intent → reward late
    elif direction == -1:
        sig += (0.5 - pos) * 2.0          # early intent → reward early
    qy = query_years(query)
    if qy and content:
        # full year match in the candidate text (its [date] prefix or body)
        cyears = set(re.findall(r"\b(?:19|20)\d{2}\b", content))
        if qy & cyears:
            sig += 1.0
    return max(-1.0, min(1.0, sig))


def importance_signal(content: str) -> float:
    """Signal in [0,1]: how 'load-bearing' a memory is (decision/commitment/fact density).

    Cheap deterministic proxy for Dakera's stored importance score. Combines:
      decision/commitment markers, salient fact tokens, first-person self-statements.
    """
    if not content:
        return 0.0
    text = content
    score = 0.0
    if _DECISION.search(text):
        score += 0.5
    if _SELF.search(text):
        score += 0.2
    # fact density (capped) — more concrete facts ⇒ more important to retrieve precisely
    n_fact = len(_FACT.findall(text))
    score += min(0.3, 0.05 * n_fact)
    return max(0.0, min(1.0, score))
