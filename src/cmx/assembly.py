"""Budget-aware context assembly with Layer-1 proactive injection.

Priority order (trim low → high to fit the model's REAL window):
  pinned sinks  >  retrieved evidence  >  recent verbatim window  >  headnotes
The retrieved evidence is rendered as a single, clearly-labelled, **non-assistant**
block carrying [id=N] tags so the model can cite and Layer 3 can verify.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import CmxConfig
from .forcing import (memory_directives_block, multihop_block, teaching_directive_block,
                      tool_protocol_block)
from .profiles import EnforcementProfile
from .retrieval import HybridRetriever, Slice
from .store import VerbatimStore
from .tokenizer import Tokenizer, window_for

EVIDENCE_HEADER = (
    "[CMX EVIDENCE — verbatim excerpts from your own conversation history. "
    "These are facts, not your prior words. Cite [id=N] for any historical claim "
    "you make, and do NOT assert history that is not supported here.]"
)

DURABLE_HEADER = (
    "[CMX DURABLE MEMORY — verbatim user/project memory retrieved for this turn. "
    "Treat as ground truth and cite [id=N]. This is only the slice relevant now; the FULL "
    "memory is always searchable on demand with cmx_grep — nothing here is summarized or "
    "complete, so use cmx_grep if you need more.]"
)


@dataclass
class AssembledContext:
    messages: list[dict]
    evidence_ids: list[int]
    tokens: int
    budget: int
    profile: EnforcementProfile
    trimmed: dict = field(default_factory=dict)
    overflow: bool = False


def _render_evidence(slices: list[Slice]) -> str:
    lines = [EVIDENCE_HEADER]
    for s in slices:
        lines.append(f"[id={s.id}] (turn {s.turn_index}, {s.role}): {s.content}")
    return "\n".join(lines)


def _render_durable(slices: list[Slice]) -> str:
    lines = [DURABLE_HEADER]
    for s in slices:
        lines.append(f"[id={s.id}] ({s.role}): {s.content}")
    return "\n".join(lines)


def assemble(*, store: VerbatimStore, retriever: HybridRetriever, tokenizer: Tokenizer,
             cfg: CmxConfig, profile: EnforcementProfile, model: str, session_id: str,
             user_text: str, system_prompt: str = "", tool_schemas: Optional[list] = None,
             durable_session: Optional[str] = None) -> AssembledContext:
    budget = max(1, window_for(model, cfg.models) - cfg.reserve_output_tokens
                 - cfg.safety_margin_tokens - getattr(cfg, "reserve_overhead_tokens", 0))

    # recent verbatim window (real roles); these ids are already on screen → dedup evidence
    recent_rows = store.recent(session_id, cfg.recent_window_turns)
    recent_ids = {r["id"] for r in recent_rows}
    recent_msgs = [{"role": r["role"], "content": r["content"]} for r in recent_rows]

    # Layer 1: proactive retrieval, deduped against the recent window
    evidence: list[Slice] = []
    if profile.inject and user_text.strip():
        evidence = [s for s in retriever.retrieve(user_text, k=profile.inject_k, session_id=session_id)
                    if s.id not in recent_ids]

    # Stage 1: durable cross-session memory — retrieved from the fixed durable namespace, not
    # dumped. Deduped against the conversation evidence ids. Always reachable on demand via
    # cmx_grep, so trimming a proactive durable slice loses nothing.
    durable: list[Slice] = []
    if durable_session and user_text.strip():
        ev_ids = {s.id for s in evidence} | recent_ids
        durable = [s for s in retriever.retrieve(
                       user_text, k=int(getattr(cfg, "durable_inject_k", 6) or 6),
                       session_id=durable_session)
                   if s.id not in ev_ids]

    pinned = store.pinned(session_id)
    pinned_txt = "\n".join(p["content"] for p in pinned)

    # F1/F2 forcing blocks. Lead the system message so they
    # override any "no memory / tool unavailable" framing the base model carries.
    forcing_mem = memory_directives_block(getattr(cfg, "force_memory_directives", False))
    forcing_tools = tool_protocol_block(tool_schemas, getattr(cfg, "prompted_tool_protocol", False))
    forcing_multihop = multihop_block(getattr(cfg, "multihop", False))
    forcing_teach = teaching_directive_block(getattr(cfg, "teach_retrieval", False))

    def build(ev: list[Slice], dur: list[Slice], rec: list[dict]):
        parts = [p for p in (forcing_mem, forcing_multihop, system_prompt, forcing_tools, pinned_txt) if p]
        if dur:
            parts.append(_render_durable(dur))
        if ev:
            parts.append(_render_evidence(ev))
        # the teaching directive goes LAST so "the material above" refers to all injected context
        if forcing_teach:
            parts.append(forcing_teach)
        system_msg = {"role": "system", "content": "\n\n".join(parts)} if parts else None
        msgs = ([system_msg] if system_msg else []) + rec
        return msgs, tokenizer.count_messages(msgs)

    msgs, tokens = build(evidence, durable, recent_msgs)
    trimmed = {"evidence": 0, "recent": 0, "durable": 0}

    # trim order (low → high priority): durable (always on-demand reachable) → conversation
    # evidence → recent window. pinned + system are never trimmed.
    while tokens > budget and durable:
        durable = durable[:-1]
        trimmed["durable"] += 1
        msgs, tokens = build(evidence, durable, recent_msgs)
    while tokens > budget and evidence:
        evidence = evidence[:-1]
        trimmed["evidence"] += 1
        msgs, tokens = build(evidence, durable, recent_msgs)
    while tokens > budget and len(recent_msgs) > 1:
        recent_msgs = recent_msgs[1:]
        trimmed["recent"] += 1
        msgs, tokens = build(evidence, durable, recent_msgs)

    overflow = tokens > budget  # pinned + system alone exceed budget (cannot trim further)
    return AssembledContext(messages=msgs, evidence_ids=[s.id for s in evidence] + [s.id for s in durable],
                            tokens=tokens, budget=budget, profile=profile,
                            trimmed=trimmed, overflow=overflow)
