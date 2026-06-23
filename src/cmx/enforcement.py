"""Grounding enforcement primitives (Layers 2–5).

Deterministic where possible: citation checking and "salient fact token" support
are pure SQLite/string operations — no LLM judgment, so a weak foreground model
cannot cite a phantom or attribute a fact to a source that does not contain it.
The verifier (Layer 4) only handles claims that escape the deterministic checks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .llm import ModelClient

_CITE = re.compile(r"\[id=(\d+)\]")
_SENT = re.compile(r"[^.!?\n]+[.!?\n]?")
# "fact" tokens: quoted spans, tokens containing a digit, or compound identifiers
# (internal _ . / : -). These are exactly what models hallucinate; plain English
# words are deliberately ignored (semantic support is Layer 4's job).
_FACT = re.compile(r'"[^"]+"|\b[\w/:.-]*\d[\w/:.-]*\b|\b[A-Za-z][\w]*[_./:-][\w./:-]+\b')

# A genuine recall question is SHORT. Instruction/persona/pasted-context blocks (hundreds of
# lines) are never "what did we say earlier?" questions. cmx is now the GLOBAL context engine
# with live answer-replacement enforcement (H2), so a false positive here REPLACES a legitimate
# answer with a refusal — the cost of over-matching flipped from "overhead" to "broken answer".
_MAX_RECALL_QUESTION_CHARS = 600

# Recall questions reference the PAST. Bare "we"/"you" are NOT signals (they appear in almost
# every prompt — "You are…", "Can you help…", "Should we…") and were the root cause of global
# over-refusal. Require an explicit past-reference: a temporal word, a memory verb, or a
# past-decision phrase ("did we/you/i", "we decided", "you said", "what did", "which … did").
_HISTORY_HINTS = re.compile(
    r"\b("
    r"earlier|previously|before|last time|moments? ago|\bago\b|"
    r"already (?:said|told|mentioned|asked|decided|chose|gave|set|picked)|"
    r"recall|remember|forget|forgot|"
    r"did (?:we|you|i)\b|"
    r"(?:we|you|i) (?:said|told|mentioned|asked|decided|chose|picked|selected|set|named|"
    r"called|agreed|discussed|established|settled|used|gave|wanted|configured|specified)|"
    r"what (?:did|was|were)\b|which .*\b(?:did|was|were)\b|who .*\b(?:did|was)\b|"
    r"the one (?:we|you|i)|that (?:we|you|i)|our (?:previous|earlier|last|prior)"
    r")\b", re.IGNORECASE)

# Past commitments / decisions = historical assertions to audit even without a
# fact token (e.g. "we promised a refund policy").
_ASSERT = re.compile(
    r"\b(promised|decided|agreed|chose|committed|guaranteed|confirmed|"
    r"established|concluded|approved|rejected)\b", re.IGNORECASE)

# Forward-looking PROPOSAL / planning / instruction cues. A turn that proposes or
# confirms a FUTURE action ("so let's shoot for <=2000 LOC per PR … right?") is NOT a
# recall question, even when it cites the past as setup ("before it was 100 LOC"). Without
# this guard a single backward word like "before" flips a planning turn into "factual
# recall" and the grounding gate DELETES the model's real answer (the opus-4.8 PR-planning
# regression, 2026-06-15). Recall asks "what WAS X?"; planning says "let's make X this".
_FORWARD_PROPOSAL = re.compile(
    r"(?:\blet'?s\b|\blet us\b|\bshoot for\b|\baim for\b|\bgoing to\b|\bgonna\b|"
    r"\bwe'?ll\b|\bi'?ll\b|\bplan(?:ning)? to\b|\bgoing forward\b|\bfrom now on\b|"
    r"\bnext time\b|\bproceed with\b|\bgo with\b|\bwe should\b|\bshould we\b)",
    re.IGNORECASE)


def classify_factual_history(user_text: str) -> bool:
    """True only if the turn is a CONCISE question that genuinely asks to recall something from
    prior conversation/memory. Conservative by design: cmx is the global engine and a false
    positive replaces a real answer with a refusal, so instruction/persona/task/paste prompts
    (and bare "you"/"we" phrasing) must NOT match. (The benchmark path uses assume_factual_history
    to bypass this — so tightening it here does not change measured numbers.)"""
    t = (user_text or "").strip()
    if not t or len(t) > _MAX_RECALL_QUESTION_CHARS:
        return False
    hist = bool(_HISTORY_HINTS.search(t))
    # bare question with a concrete fact token (asking to recall a specific value) → factual
    factq = ("?" in t) and bool(_FACT.search(t))
    if not (hist or factq):
        return False
    # A forward-looking proposal/planning/confirmation turn is NOT a recall question, even when
    # it references the past as setup — gating it would DELETE the model's real answer.
    if _FORWARD_PROPOSAL.search(t):
        return False
    return True


def strip_citations(text: str) -> str:
    return _CITE.sub("", text)


def salient_fact_tokens(text: str) -> list[str]:
    out, seen = [], set()
    for m in _FACT.findall(strip_citations(text)):
        tok = m.strip('"').strip()
        low = tok.lower()
        if len(low) >= 2 and low not in seen:
            seen.add(low)
            out.append(tok)
    return out


def span_supported(claim: str, content: str) -> bool:
    """Every salient fact token in the claim must appear in the content. If the
    claim has no fact tokens, it is not deterministically checkable here (→ L4)."""
    toks = salient_fact_tokens(claim)
    if not toks:
        return True
    c = content.lower()
    return all(t.lower() in c for t in toks)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.findall(text or "") if s.strip()]


def parse_citations(answer: str) -> list[tuple[str, list[int]]]:
    out = []
    for sent in _sentences(answer):
        ids = [int(x) for x in _CITE.findall(sent)]
        if ids:
            out.append((sent, ids))
    return out


@dataclass
class CitationFailure:
    sentence: str
    message_id: int
    reason: str  # phantom_citation | citation_mismatch


@dataclass
class CitationResult:
    ok: bool
    failures: list[CitationFailure] = field(default_factory=list)

    def message(self) -> str:
        bits = []
        for f in self.failures:
            if f.reason == "phantom_citation":
                bits.append(f"[id={f.message_id}] does not exist")
            else:
                bits.append(f"[id={f.message_id}] does not contain the fact in: \"{f.sentence}\"")
        return "; ".join(bits)


def check_citations(answer: str, store) -> CitationResult:
    """Layer 3 — deterministic. Each cited id must exist AND contain the cited
    sentence's fact tokens."""
    failures: list[CitationFailure] = []
    for sentence, ids in parse_citations(answer):
        for _id in ids:
            row = store.get_message(_id)
            if row is None:
                failures.append(CitationFailure(sentence, _id, "phantom_citation"))
            elif not span_supported(sentence, row["content"]):
                failures.append(CitationFailure(sentence, _id, "citation_mismatch"))
    return CitationResult(ok=not failures, failures=failures)


def uncited_factual_claims(answer: str) -> list[str]:
    """Sentences asserting a fact token OR a historical commitment, carrying NO
    citation (Layer 4 targets)."""
    out = []
    for sent in _sentences(answer):
        if _CITE.search(sent):
            continue
        if salient_fact_tokens(sent) or _ASSERT.search(sent):
            out.append(sent)
    return out


def unsupported_fact_tokens(answer: str, evidence_text: str) -> list[str]:
    """Deterministic strict-value backstop (iter3 experiment 4).

    Returns salient fact tokens present in the answer but absent from the assembled
    evidence corpus. These are high-risk confabulation surfaces (dates, ids, names,
    numeric values). Empty list means every salient token is grounded verbatim.
    """
    ev = (evidence_text or "").lower()
    missing = []
    for tok in salient_fact_tokens(answer):
        if tok.lower() not in ev:
            missing.append(tok)
    return missing


@dataclass
class VerifyResult:
    ok: bool
    unsupported: list[str] = field(default_factory=list)

    def message(self) -> str:
        return "Unsupported by your history: " + " | ".join(self.unsupported)


class Verifier:
    """Layer 4 — independent audit of uncited factual claims.

    Deterministic shortcut first (fact tokens present verbatim in the evidence);
    only otherwise consult the (separate, cheap) verifier model.
    """

    def __init__(self, client: Optional[ModelClient], model: str, embedder=None):
        self.client = client
        self.model = model
        self.embedder = embedder  # optional: cheap cosine pre-filter for semantic_supported

    def verify(self, answer: str, evidence_text: str) -> VerifyResult:
        claims = uncited_factual_claims(answer)
        if not claims:
            return VerifyResult(ok=True)
        ev = (evidence_text or "").lower()
        unsupported = []
        for c in claims:
            toks = salient_fact_tokens(c)
            if toks and all(t.lower() in ev for t in toks):
                continue  # deterministically supported by injected evidence
            if self.client is None:
                unsupported.append(c)  # no judge available → treat as unsupported (safe)
                continue
            verdict = self._judge(c, evidence_text)
            if verdict != "SUPPORTED":
                unsupported.append(c)
        return VerifyResult(ok=not unsupported, unsupported=unsupported)

    def semantic_supported(self, claim: str, evidence: str, prefilter_sim: float = 0.0) -> bool:
        """Lever 6 — strict semantic rescue for a CITED sentence whose fact tokens are not
        verbatim in its cited evidence (paraphrase/synonym). Returns True ONLY if an
        independent strict NLI judge confirms the evidence supports the claim. Safe by
        construction: no judge/outage → False (stay failed); never call this for phantom
        citations. An optional embedding cosine pre-filter (prefilter_sim>0) cheaply rejects
        obviously unrelated pairs without spending a judge call — it can only KEEP a failure,
        never rescue one."""
        if self.client is None or not (claim and evidence):
            return False
        if prefilter_sim > 0.0 and self.embedder is not None:
            try:
                import numpy as np
                a, b = self.embedder.embed([claim, evidence])
                a = np.asarray(a, "float32"); b = np.asarray(b, "float32")
                cos = float(a @ b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8)
                if cos < prefilter_sim:
                    return False  # too dissimilar to be a paraphrase → don't even ask the judge
            except Exception:
                pass
        return self._judge(claim, evidence) == "SUPPORTED"

    def answerable(self, question: str, evidence: str, votes: int = 1,
                   reasoning: bool = False, consensus: str = "unanimous",
                   multihop: bool = False) -> bool:
        """L2 / iter3-hardened — evidence-sufficiency check: can the QUESTION be answered
        SOLELY from the EVIDENCE? Strict — NO if it would require any fact not explicitly
        present (the on-topic-but-unanswerable case). Hardening:
          • reasoning=True  → the judge first names the SPECIFIC fact needed, then checks that
            exact fact is present (decomposition + contrastive priming), reducing the
            on-topic false-positive that is the residual's root cause;
          • votes>1         → run K times at temperature>0 and require UNANIMOUS 'answerable'
            to proceed (an unstable judgment → refuse). L5's consensus idea applied to the
            cheaper decision point.
          • multihop=True   → allow the answer to be DERIVED by COMBINING multiple evidence
            slices (the multi-hop case the strict gate over-refuses), while still requiring
            that EVERY needed sub-fact be explicitly present across the slices (an adversarial
            question whose sub-fact is in NO slice is still NO). Downstream citation-check +
            verifier keep this safe.
        Safe-ish default True (don't block) when no judge or on outage."""
        if self.client is None or not question:
            return True
        if multihop:
            sys = ("Decide if the QUESTION can be answered from the EVIDENCE, possibly by "
                   "COMBINING facts stated across SEVERAL evidence slices (multi-hop). "
                   "First, in ONE short line, name the specific fact(s) the question needs. "
                   "Then check whether EACH needed fact is explicitly stated SOMEWHERE in the "
                   "evidence (it is fine if they are in different slices and must be combined). "
                   "Answer YES if every needed fact is present across the slices so the answer "
                   "can be derived; answer NO if ANY needed fact is missing from ALL slices "
                   "(evidence merely on-topic but not stating a needed fact is NOT enough). "
                   "Do not use outside knowledge or guess. "
                   "End with a final line exactly: ANSWER: YES  or  ANSWER: NO.")

            def _yes(o: str) -> bool:
                u = (o or "").upper()
                i = u.rfind("ANSWER:")
                return (u[i:].lstrip("ANSWER:").strip().startswith("YES") if i >= 0
                        else u.strip().startswith("YES"))
        elif reasoning:
            sys = ("Decide if the QUESTION can be answered SOLELY from the EVIDENCE. "
                   "First, in ONE short line, name the SPECIFIC fact the question asks for. "
                   "Then check whether that EXACT fact is explicitly stated in the evidence. "
                   "Evidence that is merely on-topic (same people, dates, or subject) but does "
                   "NOT state the specific fact is INSUFFICIENT. Do not use outside knowledge "
                   "or guess. "
                   # C3 before-FAIL symmetric guard: do not over-refuse a fact that IS present
                   # in another form. This is the cure for the sufficiency gate's over-refusal.
                   "BEFORE you answer NO, double-check you are not missing the fact because it "
                   "is PARAPHRASED, split across two evidence lines, or stated as a synonym / "
                   "different surface form — if the evidence DOES contain the answer in any "
                   "form, answer YES. Only answer NO when the specific fact is genuinely absent. "
                   "End with a final line exactly: ANSWER: YES  or  ANSWER: NO.")

            def _yes(o: str) -> bool:
                u = (o or "").upper()
                i = u.rfind("ANSWER:")
                return (u[i:].lstrip("ANSWER:").strip().startswith("YES") if i >= 0
                        else u.strip().startswith("YES"))
        else:
            sys = ("Decide if the QUESTION can be answered SOLELY from the EVIDENCE. Reply with "
                   "exactly YES or NO. Answer NO if answering would require ANY fact not "
                   "explicitly stated in the evidence — do not use outside knowledge and do not "
                   "guess. On-topic evidence that does not actually state the answer is NO.")

            def _yes(o: str) -> bool:
                return (o or "").strip().upper().startswith("YES")

        msgs = [{"role": "system", "content": sys},
                {"role": "user", "content": f"QUESTION: {question}\n\nEVIDENCE:\n{evidence}"}]
        k = max(1, votes)
        yes = 0
        for i in range(k):
            try:
                temp = None if k == 1 else 0.6
                out = self.client.complete(msgs, model=self.model, temperature=temp).content or ""
            except Exception:
                return True
            if _yes(out):
                yes += 1
        mode = (consensus or "unanimous").strip().lower()
        if mode == "majority":
            return yes >= (k // 2 + 1)
        return yes == k  # unanimous 'answerable' to proceed; any doubt → refuse

    def answerable_score(self, question: str, evidence: str) -> int:
        """L6 (iter3) — graded sufficiency: 0-100 confidence that the EVIDENCE actually
        contains an answer to the QUESTION. 100 on outage/no-judge (don't block)."""
        if self.client is None or not question:
            return 100
        sys = ("Rate 0-100 how confident you are that the EVIDENCE explicitly contains an "
               "answer to the QUESTION (100 = the answer is plainly stated; 0 = it is not in "
               "the evidence at all, even if the topic is related). Reply with ONLY the number.")
        msgs = [{"role": "system", "content": sys},
                {"role": "user", "content": f"QUESTION: {question}\n\nEVIDENCE:\n{evidence}"}]
        try:
            out = (self.client.complete(msgs, model=self.model).content or "")
        except Exception:
            return 100
        import re
        m = re.search(r"\d{1,3}", out)
        return max(0, min(100, int(m.group()))) if m else 100

    def _judge(self, claim: str, evidence: str) -> str:
        sys = ("You are a strict grounding auditor. Reply with exactly SUPPORTED or "
               "UNSUPPORTED: does the EVIDENCE explicitly support the CLAIM? Treat a "
               "different value, name, or entity than the evidence states (a substitution "
               "or contradiction) as UNSUPPORTED. Paraphrase with the SAME meaning is "
               "SUPPORTED. If the evidence does not state it, answer UNSUPPORTED.")
        msgs = [{"role": "system", "content": sys},
                {"role": "user", "content": f"CLAIM: {claim}\n\nEVIDENCE:\n{evidence}"}]
        try:
            out = self.client.complete(msgs, model=self.model).content.upper()
        except Exception:
            return "UNSUPPORTED"  # outage → safe default (refuse, never confabulate)
        return "SUPPORTED" if "SUPPORTED" in out and "UNSUPPORTED" not in out else "UNSUPPORTED"
