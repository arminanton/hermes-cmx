"""FORCING-LAYER benchmark — does the prompt-level forcing lift accuracy?

Compares, on the SAME maxed-retrieval + sufficiency-gate config, two arms:
  * baseline : locked config, NO forcing (the 58.3%/8.9% locked result's config)
  * forcing  : + F1 memory directives + F2 prompted protocol + F3 dialect parser
               + F4 reasoning strip + F5 forced agentic search + the C3 before-FAIL judge guard
Across both fast/cheap foreground models (gpt-5-mini, gemini-2.5-pro). Hypothesis: forcing lifts
answerable accuracy (model stops refusing / searches harder) WITHOUT raising hallucination (the
citation-check + verify + refuse guardrail still holds). Resumable.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_council_matrix as M  # noqa: E402  (reuse _metrics)
import run_locomo_eval as R     # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "matrix" / "iter4-forcing.jsonl"
MD = Path(__file__).resolve().parent / "results" / "matrix" / "iter4-FORCING.md"

FGS = [m.strip() for m in os.environ.get("FORCE_MODELS", "gpt-5-mini,gemini-2.5-pro").split(",")]
N_ANS = int(os.environ.get("FORCE_N", "10"))
N_ADV = int(os.environ.get("FORCE_ADV", "12"))
SEEDS = [int(x) for x in os.environ.get("FORCE_SEEDS", "0").split(",")]

# locked retrieval + gate, held fixed across arms
BASE = dict(use_embeddings=True, rerank=True, semantic_verifier=True, sufficiency_gate=True,
            sufficiency_reasoning=True, assume_factual_history=True, judge_backend="single",
            verifier_model="claude-opus-4.7", n_answerable=N_ANS, n_adversarial=N_ADV)

ARMS = [
    ("baseline", {}),
    ("forcing", dict(force_memory_directives=True, prompted_tool_protocol=True,
                     parse_tool_dialects=True, strip_reasoning=True, agentic_search=True)),
]


def done_ids():
    s = set()
    if OUT.exists():
        for ln in OUT.read_text().splitlines():
            try:
                s.add(json.loads(ln)["id"])
            except Exception:
                pass
    return s


def render():
    rows = []
    if OUT.exists():
        for ln in OUT.read_text().splitlines():
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    rows.sort(key=lambda r: (r["foreground"], r["seed"], r["arm"]))
    out = [f"# iter-4 FORCING-LAYER benchmark — n={N_ANS} ans + {N_ADV} adv, seeds {SEEDS}",
           "_baseline (locked, no forcing) vs forcing (F1–F5 + C3 judge guard). "
           "Goal: ↑accuracy, hallucination not worse._\n",
           "| foreground | seed | arm | answer_acc | adv_refusal | halluc | latency |",
           "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        if r.get("error"):
            out.append(f"| {r['foreground']} | {r['seed']} | {r['arm']} | ERR | "
                       f"{str(r['error'])[:34]} | | {r.get('elapsed',0):.0f}s |")
            continue
        m = r["metrics"]
        out.append(f"| {r['foreground']} | {r['seed']} | {r['arm']} | {m['answer_acc']}% | "
                   f"{m['adv_refusal']}% | {m['halluc']}% | {r.get('elapsed',0):.0f}s |")
    # delta summary per model (pooled arms, seed 0 or pooled)
    out.append("\n## Δ (forcing − baseline), per foreground (pooled seeds)\n")
    out.append("| foreground | Δ answer_acc | Δ adv_refusal | Δ halluc |")
    out.append("|---|---:|---:|---:|")
    bym = {}
    for r in rows:
        if r.get("error"):
            continue
        bym.setdefault(r["foreground"], {}).setdefault(r["arm"], []).append(r["metrics"])
    for fg, arms in sorted(bym.items()):
        if "baseline" in arms and "forcing" in arms:
            avg = lambda L, k: sum(x[k] for x in L) / len(L)
            da = avg(arms["forcing"], "answer_acc") - avg(arms["baseline"], "answer_acc")
            dr = avg(arms["forcing"], "adv_refusal") - avg(arms["baseline"], "adv_refusal")
            dh = avg(arms["forcing"], "halluc") - avg(arms["baseline"], "halluc")
            out.append(f"| {fg} | {da:+.1f}pt | {dr:+.1f}pt | {dh:+.1f}pt |")
    MD.write_text("\n".join(out) + "\n")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plan = [(fg, seed, arm, kw) for fg in FGS for seed in SEEDS for (arm, kw) in ARMS]
    done = done_ids()
    for i, (fg, seed, arm, kw) in enumerate(plan, 1):
        cid = f"{fg}|seed{seed}|{arm}"
        if cid in done:
            print(f"[{i}/{len(plan)}] SKIP {cid}", flush=True)
            continue
        print(f"[{i}/{len(plan)}] RUN  {cid}", flush=True)
        t0 = time.time()
        rec = {"id": cid, "foreground": fg, "seed": seed, "arm": arm, "ts": time.strftime("%H:%M:%S")}
        try:
            _, res = R.run_locomo(cid, fg, "copilot", sample_idx=seed, **BASE, **kw)
            rec["metrics"] = M._metrics(res)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["elapsed"] = round(time.time() - t0, 1)
        with OUT.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        render()
        mm = rec.get("metrics") or {}
        print(f"  -> acc={mm.get('answer_acc')}% advref={mm.get('adv_refusal')}% "
              f"halluc={mm.get('halluc')}% {rec.get('error','')} ({rec['elapsed']:.0f}s)", flush=True)
    print(f"DONE -> {MD}", flush=True)


if __name__ == "__main__":
    main()
