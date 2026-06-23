"""iter-4 FINALISTS — the decisive judge experiment (high adversarial-n).

The broad n=6 grid showed single:opus already saturates safety (100%/0%), so it cannot expose
the residual the Council targets, and the Council-as-pre-gate over-refuses answerable questions.
This isolates the real decision with a clean 2x2 at HIGH adversarial-n (24, where iter-3 saw the
single judge fail 5-31%):

    judge backend  x  role
    {single:opus, council:m5cc}  x  {pre-gate, post-verify-only(gate off)}

- pre-gate         : refuse before answering if the judge says evidence is insufficient.
- post-verify-only : no pre-gate (answer freely), the judge audits the shipped answer (Layer 4);
                     tests whether the Council catches confabulations WITHOUT the over-refusal.

Foreground = gpt-5-mini (the weak/fast case + efficiency thesis). Council souls = COUNCIL_MODEL
(opus-4.7). Maxed retrieval held fixed. Resumable.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_council_matrix as M  # noqa: E402  (reuse M5CC + _metrics)
import run_locomo_eval as R     # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "matrix" / "iter4-finalists.jsonl"
MD = Path(__file__).resolve().parent / "results" / "matrix" / "iter4-FINALISTS.md"

N_ANS = int(os.environ.get("FIN_N", "10"))
N_ADV = int(os.environ.get("FIN_ADV", "24"))
FG = os.environ.get("FIN_MODEL", "gpt-5-mini")

CONFIGS = [
    ("single:opus|gate", dict(judge_backend="single", verifier_model="claude-opus-4.7",
                              sufficiency_gate=True)),
    ("single:opus|verify-only", dict(judge_backend="single", verifier_model="claude-opus-4.7",
                                     sufficiency_gate=False)),
    ("council:m5cc|gate", dict(judge_backend="council", council_personas=M.M5CC,
                               council_verdict_only=False, sufficiency_gate=True)),
    ("council:m5cc|verify-only", dict(judge_backend="council", council_personas=M.M5CC,
                                      council_verdict_only=False, sufficiency_gate=False)),
]
COMMON = dict(use_embeddings=True, rerank=True, semantic_verifier=True,
              sufficiency_reasoning=True, assume_factual_history=True,
              n_answerable=N_ANS, n_adversarial=N_ADV)


def done_ids():
    s = set()
    if OUT.exists():
        for ln in OUT.read_text().splitlines():
            try:
                s.add(json.loads(ln)["label"])
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
    order = {c[0]: i for i, c in enumerate(CONFIGS)}
    rows.sort(key=lambda r: order.get(r["label"], 99))
    out = [f"# iter-4 FINALISTS — {FG} foreground, n={N_ANS} ans + {N_ADV} adv, "
           f"council={os.environ.get('COUNCIL_MODEL','?')}",
           "_Decision: 0% halluc guardrail → max adv-refusal → max accuracy → min latency._\n",
           "| judge config | answer_acc | adv_refusal | halluc | latency |",
           "|---|---:|---:|---:|---:|"]
    for r in rows:
        if r.get("error"):
            out.append(f"| {r['label']} | ERROR | {str(r['error'])[:40]} | | {r.get('elapsed',0):.0f}s |")
            continue
        m = r["metrics"]
        out.append(f"| {r['label']} | {m['answer_acc']}% | {m['adv_refusal']}% | {m['halluc']}% | "
                   f"{r.get('elapsed',0):.0f}s |")
    MD.write_text("\n".join(out) + "\n")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("COUNCIL_PROVIDER", "hermes")
    os.environ.setdefault("COUNCIL_MODEL", "claude-opus-4.7")
    os.environ.setdefault("COUNCIL_DAEMON", "0")
    done = done_ids()
    for label, kw in CONFIGS:
        if label in done:
            print(f"SKIP {label}", flush=True)
            continue
        print(f"RUN  {label}", flush=True)
        t0 = time.time()
        rec = {"label": label, "foreground": FG, "ts": time.strftime("%H:%M:%S")}
        try:
            _, res = R.run_locomo(label, FG, os.environ.get("FIN_PROVIDER", "copilot"),
                                  **COMMON, **kw)
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
