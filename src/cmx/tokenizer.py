"""Provider-aware tokenizer registry.

Fixes LCM's single ``cl100k`` estimate (wrong by 10-30% on Claude/Gemini ⇒ wrong
budgets ⇒ "wasted implementation"). Uses a real tokenizer library when importable,
otherwise a provider-calibrated character heuristic. Always returns *something*
sane so budgeting never crashes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Default context windows (tokens). Overridable via CmxConfig.models.
DEFAULT_WINDOWS = {
    "gpt-5.5": 400_000,
    "gpt-5.4": 256_000,
    "gpt-5-mini": 128_000,
    "gpt-5.4-mini": 128_000,
    # GitHub Copilot serves Claude with the prompt cap at the FULL context
    # window: opus/sonnet 4.6+ are 1M (verified live via the Copilot catalog's
    # max_context_window_tokens), while the 4.5 generation and Haiku are 200k.
    # These are the authoritative *fallback* values only — HermesCmxEngine.
    # update_model() still overrides with Hermes' live-resolved window per model
    # (and on_session_start now prefers that resolved value over this table).
    "claude-opus-4.8": 1_000_000,
    "claude-opus-4.7": 1_000_000,
    "claude-opus-4.6": 1_000_000,
    "claude-opus-4.5": 200_000,
    "claude-sonnet-4.8": 1_000_000,
    "claude-sonnet-4.7": 1_000_000,
    "claude-sonnet-4.6": 1_000_000,
    "claude-sonnet-4.5": 200_000,
    "claude-haiku-4.5": 200_000,
    "gemini-3.1-pro-preview": 1_000_000,
    "gemini-3.5-flash": 1_000_000,
    "gemini-2.5-pro": 1_000_000,
    "_default": 32_000,
}

# chars-per-token calibration by provider family (English/code mix).
_CPT = {"openai": 3.8, "anthropic": 3.6, "google": 4.0, "_default": 4.0}

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def provider_of(model: str) -> str:
    m = (model or "").lower()
    if "claude" in m or "anthropic" in m:
        return "anthropic"
    if "gemini" in m or "google" in m:
        return "google"
    if "gpt" in m or "o1" in m or "o200k" in m or "openai" in m:
        return "openai"
    return "_default"


def _try_real_encoder(model: str):
    """Return a callable(text)->int using a real tokenizer if available, else None."""
    prov = provider_of(model)
    if prov == "openai":
        try:
            import tiktoken  # type: ignore
            try:
                enc = tiktoken.encoding_for_model(model)
            except Exception:
                enc = tiktoken.get_encoding("o200k_base")
            return lambda t: len(enc.encode(t))
        except Exception:
            return None
    # anthropic/google: no portable offline tokenizer guaranteed; use heuristic.
    return None


@dataclass
class Tokenizer:
    model: str

    def __post_init__(self):
        self._real = _try_real_encoder(self.model)
        self._cpt = _CPT.get(provider_of(self.model), _CPT["_default"])

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._real is not None:
            try:
                return self._real(text)
            except Exception:
                pass
        # Heuristic: blend char-ratio and word-count for stability on code/identifiers.
        chars = len(text)
        words = len(_WORD.findall(text))
        by_chars = chars / self._cpt
        by_words = words * 1.3
        return max(1, int(round((by_chars + by_words) / 2)))

    def count_messages(self, messages) -> int:
        # +4 tokens/message overhead approximation (role/format framing).
        total = 0
        for m in messages:
            total += self.count(str(m.get("content") or "")) + 4
        return total


_REGISTRY: dict[str, Tokenizer] = {}


def get_tokenizer(model: str) -> Tokenizer:
    if model not in _REGISTRY:
        _REGISTRY[model] = Tokenizer(model)
    return _REGISTRY[model]


def window_for(model: str, models_cfg: dict | None = None) -> int:
    if models_cfg and model in models_cfg and "window" in models_cfg[model]:
        return int(models_cfg[model]["window"])
    if model in DEFAULT_WINDOWS:
        return DEFAULT_WINDOWS[model]
    # family default
    for key, win in DEFAULT_WINDOWS.items():
        if key != "_default" and key.split("-")[0] in (model or "").lower():
            return win
    return DEFAULT_WINDOWS["_default"]
