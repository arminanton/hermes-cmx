"""iter-4 diagnostic — WHERE do cmx's answerable misses come from, and is context decoupled?

Answers two questions in one run:
  (1) ARCHITECTURE: run a STRONG foreground (opus-4.8) with a deliberately SMALL window so the
      conversation cannot fit — if accuracy holds, the SQLite store (not the model's window) is
      holding memory → the "context size must match" problem is solved.
  (2) HEADROOM: log every answerable question's (category, evidence-count, outcome) and bucket
      single-hop (1 evidence) vs multi-hop (2+). If cmx is fine on single-hop but fails multi-hop,
      lever (2) (multi-hop synthesis) is the right next build.

Outcome per answerable question: correct | refused | wrong.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cmx.config import CmxConfig            # noqa: E402
from cmx.engine import CmxEngine            # noqa: E402
from cmx.retrieval import HybridRetriever   # noqa: E402
from cmx.store import VerbatimStore         # noqa: E402
import run_eval                              # noqa: E402
import run_real_eval                         # noqa: E402
import run_locomo_eval as R                  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "matrix" / "iter4-DIAGNOSTIC.md"

FG = os.environ.get("DG_MODEL", "claude-opus-4.8")
WINDOW = int(os.environ.get("DG_WINDOW", "8000"))   # force a small window (conversation won't fit)
SEEDS = [int(x) for x in os.environ.get("DG_SEEDS", "0,1,2").split(",")]
N_ANS = int(os.environ.get("DG_N", "20"))
JUDGE = os.environ.get("DG_JUDGE", "claude-opus-4.7")


def run_seed(seed):
    data = json.load(open(R.DATASET))
    sample = data[seed]
    cfg = CmxConfig()
    cfg.forced_gate = "off"; cfg.max_regenerations = 1; cfg.recent_window_turns = 4
    cfg.use_embeddings = True; cfg.rerank = True; cfg.semantic_verifier = True
    cfg.sufficiency_gate = True; cfg.sufficiency_reasoning = True; cfg.assume_factual_history = True
    cfg.verifier_model = "claude-opus-4.7"
    cfg.models = {FG: {"window": WINDOW, "follows_instructions": "high"}}  # PINNED SMALL WINDOW
    store = VerbatimStore(run_eval._tmpdb(), use_trigram=True, use_graph=True)
    from cmx.embeddings import HermesEmbedder
    eng = CmxEngine(cfg, store, HybridRetriever(store, cfg, embedder=HermesEmbedder()),
                    run_real_eval.RealModelClient(FG, provider="copilot"),
                    verifier_client=run_real_eval.RealModelClient(cfg.verifier_model, provider="copilot"))
    nturns, spk_a, spk_b = R.ingest_conversation(eng, sample)
    judge = run_real_eval.RealModelClient(JUDGE, provider="copilot")
    contract = (f"You answer questions about a conversation between {spk_a} and {spk_b}, using ONLY "
                "the [CMX EVIDENCE] verbatim excerpts shown. Cite [id=N] for any historical fact. "
                "If the answer is not in the evidence, reply EXACTLY: I don't have that in my record. "
                "One short answer.")
    ans = [q for q in sample["qa"] if q.get("category") != 5 and q.get("answer") is not None][:N_ANS]
    buckets = defaultdict(lambda: {"n": 0, "correct": 0, "refused": 0, "wrong": 0})
    for q in ans:
        nev = len(q.get("evidence") or [])
        hop = "single" if nev <= 1 else "multi"
        resp = eng.respond("locomo", q["question"], model=FG, system_prompt=contract, persist=False)
        refused = resp.refused or run_eval._is_refusal(resp.text)
        gold = R._gold_str(q["answer"])
        b = buckets[hop]; b["n"] += 1
        if refused:
            b["refused"] += 1
        elif R._llm_judge_correct(q["question"], gold, resp.text, judge, JUDGE):
            b["correct"] += 1
        else:
            b["wrong"] += 1
    return nturns, buckets


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    agg = {"single": defaultdict(int), "multi": defaultdict(int)}
    nturns_seen = []
    lines = [f"# iter-4 DIAGNOSTIC — foreground={FG}, PINNED window={WINDOW} tokens, judge={JUDGE}",
             "_If accuracy holds with a window too small for the conversation, the DB (not the "
             "model window) holds memory → context-size-match problem solved._\n"]
    for seed in SEEDS:
        t0 = time.time()
        try:
            nturns, buckets = run_seed(seed)
        except Exception as e:  # noqa: BLE001
            lines.append(f"- seed {seed}: ERROR {type(e).__name__}: {str(e)[:120]}")
            Path(OUT).write_text("\n".join(lines) + "\n"); print(lines[-1], flush=True); continue
        nturns_seen.append(nturns)
        for hop in ("single", "multi"):
            for k, v in buckets[hop].items():
                agg[hop][k] += v
        p = lambda a, b: 100.0 * a / b if b else 0.0
        s, m = buckets["single"], buckets["multi"]
        lines.append(f"- seed {seed} ({nturns} turns ingested): "
                     f"single-hop {s['correct']}/{s['n']} ({p(s['correct'],s['n']):.0f}%), "
                     f"multi-hop {m['correct']}/{m['n']} ({p(m['correct'],m['n']):.0f}%)")
        Path(OUT).write_text("\n".join(lines) + "\n")
        print(lines[-1], flush=True)
    # pooled
    p = lambda a, b: 100.0 * a / b if b else 0.0
    lines.append("\n## Pooled\n")
    lines.append("| hop type | n | correct | refused | wrong | accuracy |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for hop in ("single", "multi"):
        a = agg[hop]
        lines.append(f"| {hop}-hop | {a['n']} | {a['correct']} | {a['refused']} | {a['wrong']} | "
                     f"{p(a['correct'],a['n']):.1f}% |")
    if nturns_seen:
        lines.append(f"\n_Conversation turns ingested (vs {WINDOW}-token window): "
                     f"{min(nturns_seen)}–{max(nturns_seen)}. The window holds only a few turns; "
                     "the rest lives in SQLite and is retrieved on demand._")
    Path(OUT).write_text("\n".join(lines) + "\n")
    print("DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
