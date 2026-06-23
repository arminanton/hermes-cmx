"""Model capability registry + enforcement profiles.

The grounding *contract* never changes across models; only the *strictness* of
enforcement does. A weak/small/unknown model gets the strictest profile so it
receives the same grounding guarantee as a frontier model.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import CmxConfig
from .tokenizer import provider_of, window_for


@dataclass(frozen=True)
class Capability:
    window: int
    follows_instructions: str   # high|low
    forced_tools: bool
    cost: str                   # high|med|low


@dataclass(frozen=True)
class EnforcementProfile:
    name: str                   # strict|balanced|light
    inject: bool                # Layer 1
    inject_k: int
    forced_gate: bool           # Layer 2 on factual-history turns
    require_citations: bool     # Layer 3
    verifier: str               # mandatory|sampled|off  (Layer 4)
    refuse_to_guess: bool       # Layer 5


# Built-in capability hints; CmxConfig.models overrides per model.
_BUILTIN = {
    "high_big": Capability(0, "high", True, "high"),
    "low_small": Capability(0, "low", True, "low"),
}


def capability_for(model: str, cfg: CmxConfig) -> Capability:
    win = window_for(model, cfg.models)
    m = (cfg.models or {}).get(model, {})
    follows = m.get("follows_instructions")
    forced = m.get("forced_tools")
    cost = m.get("cost")
    if follows is None:
        # heuristic: 'mini'/unknown => low; known frontier families => high
        low = ("mini" in (model or "").lower()) or (model not in cfg.models and win <= 64_000)
        follows = "low" if low else "high"
    if forced is None:
        forced = True  # most modern providers support forced tool use
    if cost is None:
        cost = "low" if "mini" in (model or "").lower() else "high"
    return Capability(window=win, follows_instructions=follows,
                      forced_tools=bool(forced), cost=cost)


def profile_for(model: str, cfg: CmxConfig) -> EnforcementProfile:
    cap = capability_for(model, cfg)
    strict = (cap.follows_instructions == "low") or (cap.window <= 64_000) or (model not in cfg.models and model not in _known_models())
    light = (cap.follows_instructions == "high") and (cap.window >= 200_000)

    if strict and not (light and not (cap.follows_instructions == "low" or cap.window <= 64_000)):
        return EnforcementProfile("strict", inject=True, inject_k=_strict_inject_k(cap, cfg),
                                  forced_gate=cfg.forced_gate != "off", require_citations=True,
                                  verifier="mandatory", refuse_to_guess=cfg.refuse_to_guess)
    if light:
        return EnforcementProfile("light", inject=True, inject_k=cfg.k_default,
                                  forced_gate=(cfg.forced_gate == "always"), require_citations=True,
                                  verifier="sampled", refuse_to_guess=cfg.refuse_to_guess)
    return EnforcementProfile("balanced", inject=True, inject_k=cfg.k_default,
                              forced_gate=cfg.forced_gate != "off", require_citations=True,
                              verifier="sampled", refuse_to_guess=cfg.refuse_to_guess)


def _strict_inject_k(cap: Capability, cfg: CmxConfig) -> int:
    """Strict-profile evidence count (iter2 lever 5 — empirically tuned).

    Enforcement strictness here is already maximal (forced gate + mandatory verifier +
    refuse-to-guess). For evidence *coverage* we tested raising inject_k (4 -> k_default)
    on a weak-but-large-window model (gpt-5-mini @128k) and it BACKFIRED: no accuracy gain
    and reproducibly more confabulation (hallucination ~4.5%->~17%, adversarial-refusal
    ~92%->~75%). Weak models turn extra slices into false syntheses. So strict models keep
    a conservative evidence cap regardless of window — precision (levers 1/3/4), not recall,
    is what lifts their accuracy. cfg.strict_inject_k>0 is an explicit opt-in override.

    EXCEPTION (iter4 lever 2): when multihop synthesis is on, the linked slices a multi-hop
    answer must combine often exceed the cap-4 — so raise to multihop_inject_k. Safe because
    the citation-check + verifier still ground every cited sub-fact."""
    if getattr(cfg, "multihop", False):
        return max(cfg.strict_inject_k_small, getattr(cfg, "multihop_inject_k", 8))
    if cfg.strict_inject_k > 0:
        return max(1, cfg.strict_inject_k)
    return max(2, min(cfg.k_default, cfg.strict_inject_k_small))


def _known_models() -> set:
    from .tokenizer import DEFAULT_WINDOWS
    return {k for k in DEFAULT_WINDOWS if k != "_default"}
