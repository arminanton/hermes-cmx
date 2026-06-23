"""Judge ablation runner (iteration-2 follow-up).

Foreground model fixed (the hardest case — a weak model); vary the semantic-verifier
judge to learn whether judge strength matters for the hallucination guardrail / accuracy.
Usage: python _judge_ablation.py <judge_model> <judge_provider> <out_tag>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_locomo_eval as L  # noqa: E402

judge_model = sys.argv[1] if len(sys.argv) > 1 else "gpt-5-mini"
judge_provider = sys.argv[2] if len(sys.argv) > 2 else "copilot"
tag = sys.argv[3] if len(sys.argv) > 3 else judge_model

# foreground = gpt-5-mini (copilot, low reasoning): the weakest/hardest grounding case
lbl, r = L.run_locomo(
    f"mini-fg(low) / judge={tag}", "gpt-5-mini", "copilot",
    n_answerable=18, n_adversarial=12,
    use_embeddings=True, hyde=True, rerank=True, semantic_verifier=True, tools=False,
    extra_body={"reasoning": {"effort": "low"}},
    verifier_model=judge_model, verifier_provider=judge_provider)
report = L.report(lbl, r)
print(report)
out = Path(__file__).resolve().parent / "results" / f"judge-ablation-{tag}.md"
out.write_text(f"# Judge ablation — foreground gpt-5-mini(low), judge={tag}\n\n{report}\n")
print(f"wrote {out}")
