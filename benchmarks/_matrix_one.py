"""Iteration-3 final matrix runner — one model per process (launch 6 in parallel).

Runs the candidate iter-3 ship config on the LOCOMO gate for ONE model of the matrix and
writes a result file. A shell launcher spawns one of these per model so the whole matrix
runs in parallel (rate-limit cooldowns tolerated).

Usage: python _matrix_one.py <model_key>
Model keys: opus47med opus46fast mini-low mini-med sonnet46 gemini31
Env flags (default to the iter-3 KEEP set): CMX_MX_* override per-lever.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_locomo_eval as L  # noqa: E402

MATRIX = {
    "opus47med":  ("claude-opus-4.7", "copilot", {"reasoning": {"effort": "medium"}}),
    "opus46fast": ("claude-opus-4.6", "copilot", {"reasoning": {"enabled": False}}),
    "opus46med":  ("claude-opus-4.6", "copilot", {"reasoning": {"effort": "medium"}}),
    "mini-low":   ("gpt-5-mini", "copilot", {"reasoning": {"effort": "low"}}),
    "mini-med":   ("gpt-5-mini", "copilot", {"reasoning": {"effort": "medium"}}),
    "sonnet46":   ("claude-sonnet-4.6", "copilot", {"reasoning": {"enabled": False}}),
    "sonnet46med": ("claude-sonnet-4.6", "copilot", {"reasoning": {"effort": "medium"}}),
    "gemini31":   ("gemini-3.1-pro-preview", "", None),
}


def _flag(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def main():
    key = sys.argv[1]
    model, provider, xb = MATRIX[key]
    n = int(os.environ.get("CMX_MX_N", "18"))
    adv = int(os.environ.get("CMX_MX_ADV", "12"))
    # iter-3 ship candidate: full retrieval stack + L2 sufficiency gate (capable judge).
    # L4 (LLM rerank), L5 (self-consistency), L6 (graded) toggled via env for ablations.
    lbl, r = L.run_locomo(
        f"{key} [{model}]", model, provider,
        n_answerable=n, n_adversarial=adv,
        use_embeddings=True, hyde=True, rerank=True, semantic_verifier=True, tools=False,
        sufficiency_gate=_flag("CMX_MX_SUFF", True),
        sufficiency_threshold=int(os.environ.get("CMX_MX_SUFF_THRESH", "0")),
        self_consistency=int(os.environ.get("CMX_MX_SC", "0")),
        llm_rerank=_flag("CMX_MX_LLMRR", False),
        sufficiency_votes=int(os.environ.get("CMX_MX_VOTES","1")),
        sufficiency_reasoning=_flag("CMX_MX_SUFFREASON", False),
        strict_value_grounding=_flag("CMX_MX_STRICTVAL", False),
        sufficiency_consensus=os.environ.get("CMX_MX_CONS","unanimous"),
        verifier_model=os.environ.get("CMX_MX_JUDGE", "gpt-5.4"),
        verifier_provider="copilot",
        extra_body=xb)
    report = L.report(lbl, r)
    print(report, flush=True)
    out = Path(__file__).resolve().parent / "results" / f"matrix-{key}.md"
    out.write_text(f"# iter3 matrix — {key} ({model})\n\n{report}\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
