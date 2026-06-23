"""iter-4 forcing ABLATION + wording tuning — isolate which knob helps/hurts.

The bundled forcing arm HURT accuracy (gpt-5-mini -40pt). Forcing is a wording-tuning
problem, not a binary knob. This isolates each lever (and wording variant) against the
SAME locked baseline so we can see exactly what each does, then keep only what helps.

Arms (each = locked retrieval+gate, varying ONE thing):
  baseline        : no forcing
  mem-full        : F1 memory directives, full wording (the heavy block)
  mem-lite        : F1 memory directives, one-paragraph wording
  mem-recall      : F1 memory directives, minimal anti-"can't recall" nudge
  strip           : F4 reasoning-strip only (should be neutral/safe)
  protocol        : F2+F3 prompted tool protocol + dialect parse (no forced loop)
  agentic         : F5 forced agentic search loop (suspected culprit)

Default foreground gpt-5-mini (most affected), 1 seed, small n for fast iteration. Resumable.
Override: ABL_MODEL, ABL_N, ABL_ADV, ABL_SEED, ABL_ONLY (substring filter).
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

OUT = Path(__file__).resolve().parent / "results" / "matrix" / "iter4-ablation.jsonl"
MD = Path(__file__).resolve().parent / "results" / "matrix" / "iter4-ABLATION.md"

FG = os.environ.get("ABL_MODEL", "gpt-5-mini")
N_ANS = int(os.environ.get("ABL_N", "10"))
N_ADV = int(os.environ.get("ABL_ADV", "10"))
SEED = int(os.environ.get("ABL_SEED", "1"))   # seed 1 had the cleanest baseline (80%/0%)
ONLY = os.environ.get("ABL_ONLY", "")

BASE = dict(use_embeddings=True, rerank=True, semantic_verifier=True, sufficiency_gate=True,
            sufficiency_reasoning=True, assume_factual_history=True, judge_backend="single",
            verifier_model="claude-opus-4.7", n_answerable=N_ANS, n_adversarial=N_ADV)

ARMS = [
    ("baseline", {}),
    ("mem-full", dict(force_memory_directives="full")),
    ("mem-lite", dict(force_memory_directives="lite")),
    ("mem-recall", dict(force_memory_directives="recall")),
    ("strip", dict(strip_reasoning=True)),
    ("protocol", dict(prompted_tool_protocol=True, parse_tool_dialects=True)),
    ("agentic", dict(agentic_search=True, prompted_tool_protocol=True, parse_tool_dialects=True)),
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
    order = {a[0]: i for i, a in enumerate(ARMS)}
    rows.sort(key=lambda r: order.get(r["arm"], 99))
    base = next((r["metrics"] for r in rows if r["arm"] == "baseline" and not r.get("error")), None)
    out = [f"# iter-4 forcing ABLATION — {FG}, seed {SEED}, n={N_ANS} ans + {N_ADV} adv",
           "_Each arm varies ONE lever vs the locked baseline. Δ vs baseline in the last cols._\n",
           "| arm | answer_acc | adv_refusal | halluc | Δacc | Δhalluc | latency |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r.get("error"):
            out.append(f"| {r['arm']} | ERR | {str(r['error'])[:30]} | | | | {r.get('elapsed',0):.0f}s |")
            continue
        m = r["metrics"]
        da = f"{m['answer_acc']-base['answer_acc']:+.1f}" if base else "—"
        dh = f"{m['halluc']-base['halluc']:+.1f}" if base else "—"
        out.append(f"| {r['arm']} | {m['answer_acc']}% | {m['adv_refusal']}% | {m['halluc']}% | "
                   f"{da} | {dh} | {r.get('elapsed',0):.0f}s |")
    MD.write_text("\n".join(out) + "\n")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    arms = [(a, kw) for (a, kw) in ARMS if not ONLY or ONLY in a]
    done = done_ids()
    for i, (arm, kw) in enumerate(arms, 1):
        cid = f"{FG}|seed{SEED}|{arm}|n{N_ANS}-{N_ADV}"
        if cid in done:
            print(f"[{i}/{len(arms)}] SKIP {cid}", flush=True)
            continue
        print(f"[{i}/{len(arms)}] RUN  {cid}", flush=True)
        t0 = time.time()
        rec = {"id": cid, "foreground": FG, "seed": SEED, "arm": arm, "ts": time.strftime("%H:%M:%S")}
        try:
            _, res = R.run_locomo(cid, FG, "copilot", sample_idx=SEED, **BASE, **kw)
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
