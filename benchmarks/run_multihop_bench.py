"""iter-4 lever 2 benchmark — multi-hop synthesis: does it lift multi-hop accuracy SAFELY?

Buckets answerable questions single-hop vs multi-hop (by gold evidence count) AND tracks the
adversarial set, for baseline (multihop OFF) vs multihop ON. The win condition:
  multi-hop accuracy UP, while adversarial-refusal NOT down and hallucination NOT up
  (the safety guardrail — multi-hop must not become a confabulation license).

Foreground default opus-4.8 (the strong model the user asked about), small pinned window so the
DB-as-memory path is exercised. LLM-judge accuracy. Resumable.
"""
from __future__ import annotations

import json
import os
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

OUT = Path(__file__).resolve().parent / "results" / "matrix" / "iter4-multihop.jsonl"
MD = Path(__file__).resolve().parent / "results" / "matrix" / "iter4-MULTIHOP.md"

FG = os.environ.get("MH_MODEL", "claude-opus-4.8")
WINDOW = int(os.environ.get("MH_WINDOW", "8000"))
SEEDS = [int(x) for x in os.environ.get("MH_SEEDS", "0,1,2").split(",")]
N_ANS = int(os.environ.get("MH_N", "20"))
N_ADV = int(os.environ.get("MH_ADV", "12"))
JUDGE = os.environ.get("MH_JUDGE", "claude-opus-4.7")
ARMS = [("baseline", False), ("multihop", True)]


def run_cell(seed, multihop):
    data = json.load(open(R.DATASET))
    sample = data[seed]
    cfg = CmxConfig()
    cfg.forced_gate = "off"; cfg.max_regenerations = 1; cfg.recent_window_turns = 4
    cfg.use_embeddings = True; cfg.rerank = True; cfg.semantic_verifier = True
    cfg.sufficiency_gate = True; cfg.sufficiency_reasoning = True; cfg.assume_factual_history = True
    cfg.verifier_model = "claude-opus-4.7"
    cfg.multihop = multihop
    cfg.models = {FG: {"window": WINDOW, "follows_instructions": "high"}}
    store = VerbatimStore(run_eval._tmpdb(), use_trigram=True, use_graph=True)
    from cmx.embeddings import HermesEmbedder
    eng = CmxEngine(cfg, store, HybridRetriever(store, cfg, embedder=HermesEmbedder()),
                    run_real_eval.RealModelClient(FG, provider="copilot"),
                    verifier_client=run_real_eval.RealModelClient(cfg.verifier_model, provider="copilot"))
    _, spk_a, spk_b = R.ingest_conversation(eng, sample)
    judge = run_real_eval.RealModelClient(JUDGE, provider="copilot")
    contract = (f"You answer questions about a conversation between {spk_a} and {spk_b}, using ONLY "
                "the [CMX EVIDENCE] verbatim excerpts shown. Cite [id=N] for any historical fact. "
                "If the answer is not in the evidence, reply EXACTLY: I don't have that in my record. "
                "One short answer.")
    out = {"single_n": 0, "single_ok": 0, "multi_n": 0, "multi_ok": 0,
           "adv_n": 0, "adv_refused": 0, "halluc": 0, "halluc_judge": 0, "shipped": 0}
    ans = [q for q in sample["qa"] if q.get("category") != 5 and q.get("answer") is not None][:N_ANS]
    adv = [q for q in sample["qa"] if q.get("category") == 5][:N_ADV]
    for q in ans:
        hop = "single" if len(q.get("evidence") or []) <= 1 else "multi"
        resp = eng.respond("locomo", q["question"], model=FG, system_prompt=contract, persist=False)
        refused = resp.refused or run_eval._is_refusal(resp.text)
        out[f"{hop}_n"] += 1
        if not refused:
            out["shipped"] += 1
            ok = R._llm_judge_correct(q["question"], R._gold_str(q["answer"]), resp.text, judge, JUDGE)
            if ok:
                out[f"{hop}_ok"] += 1
            else:
                out["halluc_judge"] += 1   # FAIR halluc: shipped an answerable the judge ruled WRONG
            if run_eval._unsupported_by_store(resp.text, store, "locomo"):
                out["halluc"] += 1         # STRICT halluc: a value not verbatim (flags correct derivations too)
    for q in adv:
        resp = eng.respond("locomo", q["question"], model=FG, system_prompt=contract, persist=False)
        refused = resp.refused or run_eval._is_refusal(resp.text)
        out["adv_n"] += 1
        if refused:
            out["adv_refused"] += 1
        else:
            out["shipped"] += 1; out["halluc"] += 1; out["halluc_judge"] += 1
    return out


def render():
    rows = []
    if OUT.exists():
        for ln in OUT.read_text().splitlines():
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    p = lambda a, b: 100.0 * a / b if b else 0.0
    agg = {a: defaultdict(int) for a, _ in ARMS}
    for r in rows:
        if r.get("error"):
            continue
        for k, v in r["raw"].items():
            agg[r["arm"]][k] += v
    out = [f"# iter-4 lever 2 — MULTI-HOP synthesis ({FG}, window {WINDOW}, judge {JUDGE})",
           f"_pooled seeds {SEEDS}. halluc(strict)=non-verbatim value (flags correct derivations); "
           "halluc(judge)=shipped answer the judge ruled WRONG (fair to multi-hop derivations)._\n",
           "| arm | single-hop acc | multi-hop acc | adv refusal | halluc(strict) | halluc(judge) |",
           "|---|---:|---:|---:|---:|---:|"]
    for arm, _ in ARMS:
        a = agg[arm]
        if not (a["single_n"] + a["multi_n"]):
            out.append(f"| {arm} | — | — | — | — | — |"); continue
        out.append(f"| {arm} | {p(a['single_ok'],a['single_n']):.1f}% ({a['single_ok']}/{a['single_n']}) "
                   f"| {p(a['multi_ok'],a['multi_n']):.1f}% ({a['multi_ok']}/{a['multi_n']}) "
                   f"| {p(a['adv_refused'],a['adv_n']):.1f}% | {p(a['halluc'],max(1,a['shipped'])):.1f}% "
                   f"| {p(a['halluc_judge'],max(1,a['shipped'])):.1f}% |")
    b, m = agg["baseline"], agg["multihop"]
    if b["multi_n"] and m["multi_n"]:
        dmh = p(m["multi_ok"], m["multi_n"]) - p(b["multi_ok"], b["multi_n"])
        dref = p(m["adv_refused"], m["adv_n"]) - p(b["adv_refused"], b["adv_n"])
        dhal = p(m["halluc_judge"], max(1, m["shipped"])) - p(b["halluc_judge"], max(1, b["shipped"]))
        out.append(f"\n**Δ multihop vs baseline: multi-hop acc {dmh:+.1f}pt · adv-refusal {dref:+.1f}pt "
                   f"· halluc(judge) {dhal:+.1f}pt**")
        verdict = ("SHIP" if dmh > 3 and dref >= -5 and dhal <= 3 else
                   "REJECT" if dmh <= 0 or dref < -10 or dhal > 5 else "TUNABLE")
        out.append(f"\n_verdict (by FAIR judge-halluc): **{verdict}**._")
    MD.write_text("\n".join(out) + "\n")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    have = set()
    if OUT.exists():
        for ln in OUT.read_text().splitlines():
            try:
                j = json.loads(ln); have.add((j["arm"], j["seed"]))
            except Exception:
                pass
    plan = [(arm, mh, s) for (arm, mh) in ARMS for s in SEEDS]
    for i, (arm, mh, seed) in enumerate(plan, 1):
        if (arm, seed) in have:
            print(f"[{i}/{len(plan)}] SKIP {arm} seed{seed}", flush=True); continue
        print(f"[{i}/{len(plan)}] RUN  {arm} seed{seed}", flush=True)
        t0 = time.time()
        rec = {"arm": arm, "seed": seed, "ts": time.strftime("%H:%M:%S")}
        try:
            rec["raw"] = run_cell(seed, mh)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["elapsed"] = round(time.time() - t0, 1)
        with OUT.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        render()
        r = rec.get("raw") or {}
        if r:
            p = lambda a, b: 100.0 * a / b if b else 0.0
            print(f"  -> multi {p(r['multi_ok'],r['multi_n']):.0f}% single {p(r['single_ok'],r['single_n']):.0f}% "
                  f"adv-ref {p(r['adv_refused'],r['adv_n']):.0f}% ({rec['elapsed']:.0f}s)", flush=True)
    print("DONE ->", MD, flush=True)


if __name__ == "__main__":
    main()
