"""iter-4 retrieval-lever benchmark — does temporal/importance re-ranking lift accuracy?

Compares, on the locked config + LLM-judge scoring, across multiple LOCOMO conversations:
  baseline      : locked config (no new levers)
  temporal      : + query-aware temporal re-ranking
  importance    : + importance weighting
  both          : + temporal + importance
gpt-5-mini foreground. Pools raw counts across seeds (low noise). Resumable.

Goal: raise answerable accuracy WITHOUT raising hallucination — the honest accuracy route the
external-benchmark investigation pointed to (Dakera's two winning levers).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_locomo_eval as R  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "matrix" / "iter4-levers.jsonl"
MD = Path(__file__).resolve().parent / "results" / "matrix" / "iter4-LEVERS.md"

FG = os.environ.get("LV_MODEL", "gpt-5-mini")
SEEDS = [int(x) for x in os.environ.get("LV_SEEDS", "0,1,2,3,4").split(",")]
N_ANS = int(os.environ.get("LV_N", "15"))
N_ADV = int(os.environ.get("LV_ADV", "10"))
JUDGE = os.environ.get("LV_JUDGE", "claude-opus-4.7")

BASE = dict(use_embeddings=True, rerank=True, semantic_verifier=True, sufficiency_gate=True,
            sufficiency_reasoning=True, assume_factual_history=True, judge_backend="single",
            verifier_model="claude-opus-4.7", n_answerable=N_ANS, n_adversarial=N_ADV,
            judge_accuracy=True, accuracy_judge_model=JUDGE)

ARMS = [
    ("baseline", {}),
    ("temporal", dict(temporal_rerank=True)),
    ("importance", dict(importance_rerank=True)),
    ("both", dict(temporal_rerank=True, importance_rerank=True)),
]


def done():
    s = set()
    if OUT.exists():
        for ln in OUT.read_text().splitlines():
            try:
                s.add(json.loads(ln)["id"])
            except Exception:
                pass
    return s


def _pool(rows, arm):
    keys = ["answerable", "correct", "correct_judge", "adversarial", "refused_adv",
            "hallucinations", "shipped"]
    agg = {k: 0 for k in keys}
    for r in rows:
        if r["arm"] == arm and not r.get("error"):
            for k in keys:
                agg[k] += r["raw"].get(k, 0)
    return agg


def render():
    rows = []
    if OUT.exists():
        for ln in OUT.read_text().splitlines():
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    p = lambda a, b: 100.0 * a / b if b else 0.0
    out = [f"# iter-4 retrieval-lever benchmark — {FG}, locked config + LLM judge ({JUDGE})",
           f"_pooled over seeds {SEEDS}, n={N_ANS} ans + {N_ADV} adv each. Goal: ↑acc, halluc not worse._\n",
           "| arm | ans | acc (judge) | acc (token) | adv_refusal | halluc | Δacc vs base |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    base = _pool(rows, "baseline")
    base_acc = p(base["correct_judge"], base["answerable"]) if base["answerable"] else None
    for arm, _ in ARMS:
        agg = _pool(rows, arm)
        if not agg["answerable"]:
            out.append(f"| {arm} | — | — | — | — | — | — |")
            continue
        ja = p(agg["correct_judge"], agg["answerable"])
        ta = p(agg["correct"], agg["answerable"])
        da = f"{ja-base_acc:+.1f}pt" if base_acc is not None else "—"
        out.append(f"| {arm} | {agg['answerable']} | {ja:.1f}% | {ta:.1f}% | "
                   f"{p(agg['refused_adv'],agg['adversarial']):.1f}% | "
                   f"{p(agg['hallucinations'],max(1,agg['shipped'])):.1f}% | {da} |")
    MD.write_text("\n".join(out) + "\n")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plan = [(arm, kw, s) for (arm, kw) in ARMS for s in SEEDS]
    d = done()
    for i, (arm, kw, seed) in enumerate(plan, 1):
        cid = f"{arm}|seed{seed}"
        if cid in d:
            print(f"[{i}/{len(plan)}] SKIP {cid}", flush=True)
            continue
        print(f"[{i}/{len(plan)}] RUN  {cid}", flush=True)
        t0 = time.time()
        rec = {"id": cid, "arm": arm, "seed": seed, "ts": time.strftime("%H:%M:%S")}
        try:
            _, res = R.run_locomo(cid, FG, "copilot", sample_idx=seed, **BASE, **kw)
            rec["raw"] = res
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["elapsed"] = round(time.time() - t0, 1)
        with OUT.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        render()
        m = rec.get("raw") or {}
        if m:
            p = lambda a, b: 100.0 * a / b if b else 0.0
            print(f"  -> judge {p(m['correct_judge'],m['answerable']):.0f}% halluc "
                  f"{p(m['hallucinations'],max(1,m['shipped'])):.0f}% ({rec['elapsed']:.0f}s)", flush=True)
    print(f"DONE -> {MD}", flush=True)


if __name__ == "__main__":
    main()
