"""iter-4 LOCK CONFIRMATION — the chosen config at high n across multiple seeds.

The finalists/gate-sweep were single-run at modest n (noisy per cell). This confirms the FINAL
JUDGE DECISION (single capable gate) before merge, at high n across 3 independent LOCOMO
conversations (samples 0/1/2 = 3 seeds), aggregating raw counts so the percentages are robust.

Two variants of the chosen config (single capable gate, reasoning ON, maxed retrieval, gpt-5-mini
foreground = the efficiency case):
  A) gate = opus-4.7        (the SAFE recommended judge)
  B) gate = gpt-5-mini      (the CHEAP judge — confirm it really holds)

Per seed: n=20 answerable + up to 30 adversarial (capped by what each sample has).
Reports per-seed rows + the aggregate (pooled counts) + spread across seeds.
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

OUT = Path(__file__).resolve().parent / "results" / "matrix" / "iter4-lock.jsonl"
MD = Path(__file__).resolve().parent / "results" / "matrix" / "iter4-LOCK-CONFIRMATION.md"

FG = os.environ.get("LOCK_MODEL", "gpt-5-mini")
SEEDS = [int(x) for x in os.environ.get("LOCK_SEEDS", "0,1,2").split(",")]
N_ANS = int(os.environ.get("LOCK_N", "20"))
N_ADV = int(os.environ.get("LOCK_ADV", "30"))

VARIANTS = [
    ("gate=opus-4.7", "claude-opus-4.7"),
    ("gate=gpt-5-mini", "gpt-5-mini"),
]
COMMON = dict(use_embeddings=True, rerank=True, semantic_verifier=True, sufficiency_gate=True,
              sufficiency_reasoning=True, assume_factual_history=True, judge_backend="single",
              n_answerable=N_ANS, n_adversarial=N_ADV)


def done_ids():
    s = set()
    if OUT.exists():
        for ln in OUT.read_text().splitlines():
            try:
                s.add(json.loads(ln)["id"])
            except Exception:
                pass
    return s


def _pool(records):
    """Aggregate raw counts across seeds → pooled percentages."""
    k = ["answerable", "correct", "adversarial", "refused_adv", "hallucinations", "shipped"]
    agg = {x: 0 for x in k}
    for r in records:
        for x in k:
            agg[x] += r["raw"].get(x, 0)
    p = lambda a, b: round(100.0 * a / b, 1) if b else 0.0
    return {
        "answer_acc": p(agg["correct"], agg["answerable"]),
        "adv_refusal": p(agg["refused_adv"], agg["adversarial"]),
        "halluc": p(agg["hallucinations"], max(1, agg["shipped"])),
        "n_ans": agg["answerable"], "n_adv": agg["adversarial"], "shipped": agg["shipped"],
    }


def render():
    rows = []
    if OUT.exists():
        for ln in OUT.read_text().splitlines():
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    out = [f"# iter-4 LOCK CONFIRMATION — {FG} foreground, n={N_ANS} ans + {N_ADV} adv × seeds {SEEDS}",
           "_Pooled across seeds (raw-count aggregation). 0% halluc guardrail → max adv-refusal → "
           "max acc → min latency._\n"]
    for vlabel, _ in VARIANTS:
        recs = [r for r in rows if r["variant"] == vlabel and not r.get("error")]
        out.append(f"\n## {vlabel}\n")
        out.append("| seed (sample) | answer_acc | adv_refusal | halluc | latency |")
        out.append("|---|---:|---:|---:|---:|")
        for r in sorted(recs, key=lambda r: r["seed"]):
            m = r["metrics"]
            out.append(f"| {r['seed']} | {m['answer_acc']}% | {m['adv_refusal']}% | {m['halluc']}% "
                       f"| {r.get('elapsed',0):.0f}s |")
        errs = [r for r in rows if r["variant"] == vlabel and r.get("error")]
        for r in errs:
            out.append(f"| {r['seed']} | ERROR: {str(r['error'])[:50]} | | | {r.get('elapsed',0):.0f}s |")
        if recs:
            pooled = _pool(recs)
            out.append(f"\n**POOLED ({pooled['n_ans']} ans + {pooled['n_adv']} adv, "
                       f"{pooled['shipped']} shipped): acc {pooled['answer_acc']}% · "
                       f"adv-refusal {pooled['adv_refusal']}% · halluc {pooled['halluc']}%**")
    MD.write_text("\n".join(out) + "\n")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("COUNCIL_PROVIDER", "hermes")
    done = done_ids()
    plan = [(vl, gm, s) for (vl, gm) in VARIANTS for s in SEEDS]
    for i, (vlabel, gate, seed) in enumerate(plan, 1):
        cid = f"{vlabel}|seed{seed}"
        if cid in done:
            print(f"[{i}/{len(plan)}] SKIP {cid}", flush=True)
            continue
        print(f"[{i}/{len(plan)}] RUN  {cid}", flush=True)
        t0 = time.time()
        rec = {"id": cid, "variant": vlabel, "seed": seed, "gate": gate,
               "ts": time.strftime("%H:%M:%S")}
        try:
            _, res = R.run_locomo(cid, FG, "copilot", verifier_model=gate, sample_idx=seed, **COMMON)
            p = lambda a, b: round(100.0 * a / b, 1) if b else 0.0
            rec["raw"] = res
            rec["metrics"] = {"answer_acc": p(res["correct"], res["answerable"]),
                              "adv_refusal": p(res["refused_adv"], res["adversarial"]),
                              "halluc": p(res["hallucinations"], max(1, res["shipped"]))}
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
