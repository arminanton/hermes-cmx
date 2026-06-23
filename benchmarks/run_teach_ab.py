"""Teaching-directive A/B: does the light teach_retrieval directive regress a COOPERATIVE
model (copilot gpt-5-mini)? Locked config, teach off vs on, pooled across samples.

The -40pt forcing regression was the HEAVY bundle on this same cooperative model; this isolates
whether the LIGHT directive alone moves accuracy. Pool >=3 conversations (per-conv variance 20-80%).
    PYTHONPATH=src python3 \
      benchmarks/run_teach_ab.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_council_matrix as M  # noqa: E402
import run_locomo_eval as R     # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "teach-ab.jsonl"
MD = Path(__file__).resolve().parent / "results" / "TEACH-AB.md"
FG = os.environ.get("AB_MODEL", "gpt-5-mini")
N_ANS = int(os.environ.get("AB_N", "10"))
N_ADV = int(os.environ.get("AB_ADV", "10"))
SAMPLES = [int(x) for x in os.environ.get("AB_SAMPLES", "0,1,2").split(",")]

BASE = dict(use_embeddings=True, rerank=True, semantic_verifier=True, sufficiency_gate=True,
            sufficiency_reasoning=True, assume_factual_history=True, judge_backend="single",
            verifier_model="claude-opus-4.7", n_answerable=N_ANS, n_adversarial=N_ADV)
ARMS = [("baseline", dict(teach_retrieval=False)), ("teach", dict(teach_retrieval=True))]


def _rows():
    rows = []
    if OUT.exists():
        for ln in OUT.read_text().splitlines():
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return rows


def render():
    rows = _rows()
    agg: dict = {}
    for r in rows:
        if r.get("error"):
            continue
        agg.setdefault(r["arm"], []).append(r["metrics"])
    out = [f"# Teaching-directive A/B — {FG} (copilot), pooled samples {SAMPLES}",
           "_baseline (teach_retrieval off) vs teach (on). A light directive should NOT regress "
           "answer_acc; the heavy forcing bundle did (-40pt) on this same model._\n",
           "| arm | n | answer_acc | adv_refusal | halluc |", "|---|---:|---:|---:|---:|"]
    means = {}
    for a in ("baseline", "teach"):
        ms = agg.get(a, [])
        if not ms:
            continue
        n = len(ms)
        means[a] = tuple(round(sum(x[k] for x in ms) / n, 1)
                         for k in ("answer_acc", "adv_refusal", "halluc"))
        out.append(f"| {a} | {n} | {means[a][0]}% | {means[a][1]}% | {means[a][2]}% |")
    if "baseline" in means and "teach" in means:
        b, t = means["baseline"], means["teach"]
        out += ["", f"**Δ (teach − baseline): acc {t[0]-b[0]:+.1f}pt · adv_refusal "
                    f"{t[1]-b[1]:+.1f}pt · halluc {t[2]-b[2]:+.1f}pt**"]
    out += ["", "## per-sample", "| arm | sample | acc | adv | halluc | s |", "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        if r.get("error"):
            out.append(f"| {r['arm']} | {r.get('sample')} | ERR {str(r['error'])[:24]} | | | |")
            continue
        m = r["metrics"]
        out.append(f"| {r['arm']} | {r.get('sample')} | {m['answer_acc']}% | {m['adv_refusal']}% "
                   f"| {m['halluc']}% | {r.get('elapsed',0):.0f}s |")
    MD.write_text("\n".join(out) + "\n")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in _rows()}
    jobs = [(a, kw, s) for (a, kw) in ARMS for s in SAMPLES]
    for i, (arm, kw, s) in enumerate(jobs, 1):
        cid = f"{FG}|s{s}|{arm}|n{N_ANS}-{N_ADV}"
        if cid in done:
            print(f"[{i}/{len(jobs)}] SKIP {cid}", flush=True)
            continue
        print(f"[{i}/{len(jobs)}] RUN  {cid}", flush=True)
        t0 = time.time()
        rec = {"id": cid, "arm": arm, "sample": s}
        try:
            _, res = R.run_locomo(cid, FG, "copilot", sample_idx=s, **BASE, **kw)
            rec["metrics"] = M._metrics(res)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["elapsed"] = round(time.time() - t0, 1)
        with OUT.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        render()
        mm = rec.get("metrics") or {}
        print(f"  -> acc={mm.get('answer_acc')}% adv={mm.get('adv_refusal')}% "
              f"halluc={mm.get('halluc')}% {rec.get('error','')} ({rec['elapsed']:.0f}s)", flush=True)
    print(f"DONE -> {MD}", flush=True)


if __name__ == "__main__":
    main()
