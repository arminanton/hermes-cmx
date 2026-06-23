"""iter-4 DEFINITIVE re-score — locked config, multiple LOCOMO conversations, BOTH scorers.

Investigation finding: our token-match scorer is honest (ties the LLM judge on a slice) but
single-conversation variance is huge (seed1 83% vs pooled 58%). This pools several conversations
at higher n with BOTH token-match and the field-standard LLM judge, to produce the real,
low-noise cmx number comparable to Dakera/Mem0 — plus the token-vs-judge delta.

Locked config: maxed retrieval + single capable (opus) sufficiency gate. gpt-5-mini foreground
(the efficiency case). Resumable.
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

OUT = Path(__file__).resolve().parent / "results" / "matrix" / "iter4-rescore.jsonl"
MD = Path(__file__).resolve().parent / "results" / "matrix" / "iter4-RESCORE.md"

FG = os.environ.get("RS_MODEL", "gpt-5-mini")
SEEDS = [int(x) for x in os.environ.get("RS_SEEDS", "0,1,2,3,4").split(",")]
N_ANS = int(os.environ.get("RS_N", "15"))
N_ADV = int(os.environ.get("RS_ADV", "15"))
JUDGE = os.environ.get("RS_JUDGE", "claude-opus-4.7")

CFG = dict(use_embeddings=True, rerank=True, semantic_verifier=True, sufficiency_gate=True,
           sufficiency_reasoning=True, assume_factual_history=True, judge_backend="single",
           verifier_model="claude-opus-4.7", n_answerable=N_ANS, n_adversarial=N_ADV,
           judge_accuracy=True, accuracy_judge_model=JUDGE)


def done():
    s = set()
    if OUT.exists():
        for ln in OUT.read_text().splitlines():
            try:
                s.add(json.loads(ln)["seed"])
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
    rows.sort(key=lambda r: r["seed"])
    out = [f"# iter-4 DEFINITIVE re-score — {FG}, locked config, judge={JUDGE}",
           f"_n={N_ANS} ans + {N_ADV} adv per conversation. token-match vs LLM-judge accuracy._\n",
           "| seed | answerable | acc (token) | acc (LLM-judge) | adv_refusal | halluc | latency |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    agg = {"answerable": 0, "correct": 0, "correct_judge": 0, "adversarial": 0,
           "refused_adv": 0, "hallucinations": 0, "shipped": 0}
    for r in rows:
        if r.get("error"):
            out.append(f"| {r['seed']} | ERR | {str(r['error'])[:40]} | | | | {r.get('elapsed',0):.0f}s |")
            continue
        m = r["raw"]
        p = lambda a, b: 100.0 * a / b if b else 0.0
        out.append(f"| {r['seed']} | {m['answerable']} | {p(m['correct'],m['answerable']):.1f}% | "
                   f"{p(m['correct_judge'],m['answerable']):.1f}% | "
                   f"{p(m['refused_adv'],m['adversarial']):.1f}% | "
                   f"{p(m['hallucinations'],max(1,m['shipped'])):.1f}% | {r.get('elapsed',0):.0f}s |")
        for k in agg:
            agg[k] += m.get(k, 0)
    if agg["answerable"]:
        p = lambda a, b: 100.0 * a / b if b else 0.0
        out.append(f"\n**POOLED ({agg['answerable']} ans + {agg['adversarial']} adv): "
                   f"token-acc {p(agg['correct'],agg['answerable']):.1f}% · "
                   f"LLM-judge-acc {p(agg['correct_judge'],agg['answerable']):.1f}% · "
                   f"adv-refusal {p(agg['refused_adv'],agg['adversarial']):.1f}% · "
                   f"hallucination {p(agg['hallucinations'],max(1,agg['shipped'])):.1f}%**")
        out.append(f"\n_token-vs-judge delta: {p(agg['correct_judge'],agg['answerable'])-p(agg['correct'],agg['answerable']):+.1f}pt "
                   "(if ~0, our scorer is honest; if large, token-match was undercounting)._")
    MD.write_text("\n".join(out) + "\n")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = done()
    for i, seed in enumerate(SEEDS, 1):
        if seed in d:
            print(f"[{i}/{len(SEEDS)}] SKIP seed{seed}", flush=True)
            continue
        print(f"[{i}/{len(SEEDS)}] RUN  seed{seed}", flush=True)
        t0 = time.time()
        rec = {"seed": seed, "foreground": FG, "ts": time.strftime("%H:%M:%S")}
        try:
            _, res = R.run_locomo(f"rescore-s{seed}", FG, "copilot", sample_idx=seed, **CFG)
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
            print(f"  -> token {p(m['correct'],m['answerable']):.0f}% | judge "
                  f"{p(m['correct_judge'],m['answerable']):.0f}% | halluc "
                  f"{p(m['hallucinations'],max(1,m['shipped'])):.0f}% ({rec['elapsed']:.0f}s)", flush=True)
    print(f"DONE -> {MD}", flush=True)


if __name__ == "__main__":
    main()
