"""iter-4 GATE SWEEP — single-judge efficiency / small-window / cross-model (fast, no council).

The finalists settled the judge ARCHITECTURE (single capable gate > council). This rounds out the
remaining 'safe decisions across any model' questions cheaply (all single-judge):
  * gate model:  opus-4.7 (capable) vs gpt-5-mini (cheap)  — can the gate be cheap?
  * foreground:  gpt-5-mini vs gemini-2.5-pro              — do fast/small models behave the same?
  * window:      native vs pinned 16k                      — the 'DB holds memory, small-context
                 foreground still runs a long conversation' thesis, made explicit.
Maxed retrieval + reasoning sufficiency gate held fixed. n=8 ans + 16 adv. Resumable.
"""
from __future__ import annotations

import json
import os
import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_council_matrix as M  # noqa: E402
import run_locomo_eval as R     # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "matrix" / "iter4-gate-sweep.jsonl"
MD = Path(__file__).resolve().parent / "results" / "matrix" / "iter4-GATE-SWEEP.md"

GATES = ["claude-opus-4.7", "gpt-5-mini"]
FGS = ["gpt-5-mini", "gemini-2.5-pro"]
WINDOWS = [0, 16000]
N_ANS, N_ADV = 8, 16
COMMON = dict(use_embeddings=True, rerank=True, semantic_verifier=True, sufficiency_gate=True,
              sufficiency_reasoning=True, assume_factual_history=True, judge_backend="single",
              n_answerable=N_ANS, n_adversarial=N_ADV)


def done_ids():
    s = set()
    if OUT.exists():
        for ln in OUT.read_text().splitlines():
            try:
                s.add(json.loads(ln)["cell"])
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
    rows.sort(key=lambda r: (r["foreground"], r["window"], r["gate"]))
    out = [f"# iter-4 GATE SWEEP — single-judge (n={N_ANS} ans + {N_ADV} adv, maxed retrieval)",
           "_Efficiency + small-window + cross-model. 0% halluc guardrail → max adv-refusal → "
           "max acc → min latency._\n",
           "| foreground | window | gate model | answer_acc | adv_refusal | halluc | latency |",
           "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        win = "native" if not r["window"] else f"{r['window']}"
        if r.get("error"):
            out.append(f"| {r['foreground']} | {win} | {r['gate']} | ERR | {str(r['error'])[:30]} | | {r.get('elapsed',0):.0f}s |")
            continue
        m = r["metrics"]
        out.append(f"| {r['foreground']} | {win} | {r['gate']} | {m['answer_acc']}% | "
                   f"{m['adv_refusal']}% | {m['halluc']}% | {r.get('elapsed',0):.0f}s |")
    MD.write_text("\n".join(out) + "\n")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = done_ids()
    cells = list(product(FGS, WINDOWS, GATES))
    for i, (fg, win, gate) in enumerate(cells, 1):
        cell = f"{fg}|w{win}|gate={gate}"
        if cell in done:
            print(f"[{i}/{len(cells)}] SKIP {cell}", flush=True)
            continue
        print(f"[{i}/{len(cells)}] RUN  {cell}", flush=True)
        t0 = time.time()
        rec = {"cell": cell, "foreground": fg, "window": win, "gate": gate,
               "ts": time.strftime("%H:%M:%S")}
        try:
            _, res = R.run_locomo(cell, fg, "copilot", verifier_model=gate, window=win, **COMMON)
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
