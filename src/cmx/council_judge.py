"""Council-backed judge — the iter-4 multi-judge (single-model, many-souls).

`CouncilVerifier` is a drop-in replacement for `enforcement.Verifier` whose
LLM-judgment methods consult the **Hermes Council** (a persona-first adversarial
panel with an engine-enforced accuracy ceiling) instead of a single model call.

Why: iter-3 hardening proved a *single* sufficiency judge has a ceiling on
on-topic-but-unanswerable questions (a stronger single judge does not help). The
fix the data points to is an ensemble unanswerability vote. The Council is exactly
that ensemble — and its **accuracy ceiling** caps confidence precisely when critics
disagree / evidence is thin, which is the abstention the residual needs.

Design notes:
  * The deterministic shortcuts in the base class (fact tokens present verbatim) are
    left untouched — the Council is consulted ONLY where the single judge would be,
    so an A/B isolates the *judge mechanism*.
  * **Single-model, many-souls:** independence comes from persona lenses, not models.
    One capable model wears every seat — set `COUNCIL_PROVIDER=hermes` and
    `COUNCIL_MODEL=<judge_model>` (e.g. `claude-opus-4.7`). Per-persona model lanes
    are an optional escalation, not a requirement.
  * Safe degradation: if the Council import/call fails, fall back to the injected
    single-judge `Verifier` (never silently pass).
"""
from __future__ import annotations

import os
import sys
from typing import Optional

from .enforcement import Verifier

# Council verdict vocabulary is exactly {allow, deny, conditional} (arbiter schema).
# Conservative mapping: only an explicit "allow" is treated as sufficient/supported —
# matching cmx's "any doubt -> refuse" discipline.
_ALLOW = "allow"


def _resolve_call_tool():
    """Return the Council ``call_tool(name, args)`` callable.

    Import-lazy so cmx imports without the Council installed. Adds
    ``$COUNCIL_SRC/libs`` to ``sys.path`` if the package is not already importable.
    Patchable in tests (monkeypatch ``cmx.council_judge._resolve_call_tool``).
    """
    try:
        from hermes_council.server import call_tool  # type: ignore
        return call_tool
    except Exception:
        src = os.environ.get("COUNCIL_SRC", "")
        if src:
            libs = os.path.join(src, "libs")
            if libs not in sys.path:
                sys.path.insert(0, libs)
        from hermes_council.server import call_tool  # type: ignore
        return call_tool


class _Sentinel:
    """Truthy stand-in for the base class' ``client`` None-guards.

    The base ``Verifier`` skips its judge whenever ``client is None`` (e.g. in
    ``verify()``). The Council judge overrides every judging method, so we only need
    ``client`` to be *truthy* for the base control flow to reach those overrides.
    """


class CouncilVerifier(Verifier):
    def __init__(self, model: str = "", *, mode: str = "fast", verdict_only: bool = True,
                 panel: Optional[str] = None, preset: Optional[str] = None,
                 peer_review: bool = False, personas: Optional[list] = None, embedder=None,
                 fallback: Optional[Verifier] = None):
        super().__init__(client=_Sentinel(), model=model, embedder=embedder)
        self.mode = mode or "fast"
        self.verdict_only = verdict_only
        self.panel = panel or None
        self.preset = preset or None
        self.peer_review = bool(peer_review)
        self.personas = list(personas) if personas else None  # explicit seats (member-count study)
        self.fallback = fallback  # single-judge Verifier used if the Council is unavailable

    # -- council plumbing -------------------------------------------------
    def _verdict(self, question: str) -> Optional[str]:
        """Run one Council ``query`` and return the lowercased arbiter verdict, or
        ``None`` if the Council is unreachable (→ caller falls back)."""
        try:
            call_tool = _resolve_call_tool()
        except Exception:
            return None
        args = {
            "question": question,
            "mode": self.mode,
            "evidence_search": False,   # grounding evidence is in-prompt, not on the web
            "verdict_only": self.verdict_only,
            "peer_review": self.peer_review,
        }
        if self.panel:
            args["panel"] = self.panel
        if self.preset:
            args["preset"] = self.preset
        if self.personas:
            args["personas"] = self.personas
        try:
            res = call_tool("query", args)
        except Exception:
            return None
        v = res.get("verdict") if isinstance(res, dict) else None
        return str(v).strip().lower() if v else None

    # -- overrides (the judge swaps; base deterministic checks unchanged) --
    def answerable(self, question: str, evidence: str, votes: int = 1,
                   reasoning: bool = False, consensus: str = "unanimous",
                   multihop: bool = False) -> bool:
        """L2 sufficiency gate via the Council. allow => answerable; deny/conditional
        => refuse. (votes/reasoning/consensus/multihop accepted for interface parity; the
        Council gets its scrutiny from its panel + accuracy ceiling. multihop folds a
        combine-across-slices clause into the gate question.)"""
        if not question:
            return True
        combine = ("The answer may require COMBINING facts across several evidence slices; "
                   "allow if every needed fact is present somewhere in the evidence. "
                   if multihop else "")
        q = ("SUFFICIENCY GATE. Decide whether the QUESTION can be answered SOLELY from "
             "the EVIDENCE below. " + combine +
             "Evidence that is merely on-topic (same people, dates, or "
             "subject) but does NOT state a needed fact is "
             "INSUFFICIENT. Do not use outside knowledge and do not guess. Treat this as a "
             "safety gate: return verdict 'allow' ONLY if the evidence explicitly contains "
             "what is needed; otherwise 'deny'.\n\n"
             f"QUESTION: {question}\n\nEVIDENCE:\n{evidence}")
        v = self._verdict(q)
        if v is None:
            if self.fallback is not None:
                return self.fallback.answerable(question, evidence, votes=votes,
                                                reasoning=reasoning, consensus=consensus,
                                                multihop=multihop)
            return True  # no judge at all → don't block (matches base safe default)
        return v == _ALLOW

    def answerable_score(self, question: str, evidence: str) -> int:
        """L6 graded path (rejected as a default). Map the Council's calibrated
        confidence to 0-100; 100 on outage/no-judge (don't block)."""
        if not question:
            return 100
        try:
            call_tool = _resolve_call_tool()
            res = call_tool("query", {
                "question": ("Rate whether the EVIDENCE explicitly contains an answer to the "
                             "QUESTION (allow=yes, deny=no, not merely on-topic).\n\n"
                             f"QUESTION: {question}\n\nEVIDENCE:\n{evidence}"),
                "mode": self.mode, "evidence_search": False, "verdict_only": self.verdict_only,
            })
        except Exception:
            if self.fallback is not None:
                return self.fallback.answerable_score(question, evidence)
            return 100
        verdict = str(res.get("verdict", "")).strip().lower()
        conf = float(res.get("confidence", 0.5) or 0.5)
        if verdict == _ALLOW:
            return int(round(conf * 100))
        return int(round((1.0 - conf) * 100))  # deny/conditional → low sufficiency

    def _judge(self, claim: str, evidence: str) -> str:
        """Layer-4 grounding audit via the Council. allow => SUPPORTED."""
        q = ("GROUNDING AUDIT. Does the EVIDENCE explicitly support the CLAIM? A different "
             "value, name, or entity than the evidence states (a substitution or "
             "contradiction) is NOT supported. Paraphrase with the SAME meaning IS supported. "
             "If the evidence does not state it, it is NOT supported. Return verdict 'allow' "
             "if fully supported, otherwise 'deny'.\n\n"
             f"CLAIM: {claim}\n\nEVIDENCE:\n{evidence}")
        v = self._verdict(q)
        if v is None:
            if self.fallback is not None:
                return self.fallback._judge(claim, evidence)
            return "UNSUPPORTED"  # outage + no fallback → safe default (refuse)
        return "SUPPORTED" if v == _ALLOW else "UNSUPPORTED"


def make_verifier(cfg, client, model: str, embedder=None) -> Verifier:
    """Factory: a Council-backed verifier when ``cfg.judge_backend == 'council'`` and a
    judge is in play (``client is not None``); otherwise the built-in single-model
    ``Verifier``. Keeps ``engine.respond`` agnostic to the judge backend.

    When the Council is selected, the injected single-judge client becomes the
    *fallback* used only if the Council is unreachable.
    """
    backend = str(getattr(cfg, "judge_backend", "single") or "single").lower()
    if backend == "council" and client is not None:
        fallback = Verifier(client, model, embedder=embedder)
        return CouncilVerifier(
            model,
            mode=getattr(cfg, "council_mode", "fast"),
            verdict_only=getattr(cfg, "council_verdict_only", True),
            panel=getattr(cfg, "council_panel", "") or None,
            preset=getattr(cfg, "council_preset", "") or None,
            peer_review=getattr(cfg, "council_peer_review", False),
            personas=getattr(cfg, "council_personas", None) or None,
            embedder=embedder,
            fallback=fallback,
        )
    return Verifier(client, model, embedder=embedder)
