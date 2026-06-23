"""v2 head-to-head: cmx (retrieval + enforcement) vs LCM (lossy compression) on LOCOMO.

Same foreground model, same questions, same scoring oracle. Isolates the ENGINE:
  • LCM arm  — compress the long history once (its real summarization), then single-shot
    answer each question from the managed context (no tool loop — the common case, and
    the user's complaint is that models don't reliably call retrieval tools).
  • cmx arm  — the iteration-2 ship config (proactive verbatim injection + deterministic
    citation check + refuse-to-guess), reused from run_locomo_eval.

Metrics (both arms): answerable accuracy (loose match), refusal-on-adversarial (LOCOMO
category 5 = unanswerable → must refuse), hallucination (shipped answer asserting a fact
absent from the verbatim conversation). The verbatim "truth store" is the same for both
arms, so the hallucination oracle is engine-independent.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cmx.store import VerbatimStore          # noqa: E402
import run_eval                               # noqa: E402
import run_real_eval                          # noqa: E402
import run_locomo_eval as L                   # noqa: E402

DATASET = str(Path(__file__).resolve().parent / "locomo10.json")


def _register_lcm():
    """Make the hermes-lcm plugin importable as the package `hermes_lcm` (its dir name
    has a hyphen, so it is normally aliased by the host loader). Mirrors the plugin's
    tests/conftest.py — register the package WITHOUT exec'ing __init__ (which would try
    to register tools with a host ctx)."""
    import importlib.util
    import os
    if "hermes_lcm" in sys.modules:
        return
    lcm_dir = Path(os.environ.get("LCM_PLUGIN_DIR",
                   "$HERMES_HOME/plugins/hermes-lcm"))
    spec = importlib.util.spec_from_file_location(
        "hermes_lcm", str(lcm_dir / "__init__.py"),
        submodule_search_locations=[str(lcm_dir)])
    mod = importlib.util.module_from_spec(spec)
    mod.__path__ = [str(lcm_dir)]
    mod.__package__ = "hermes_lcm"
    sys.modules["hermes_lcm"] = mod

CONTRACT = (
    "You answer questions about a prior conversation, using ONLY the context shown. "
    "If the answer is not in the context, reply EXACTLY: I don't have that in my record. "
    "One short answer.")


def _truth_store(sample, session="truth"):
    """Verbatim store of every turn — the engine-independent hallucination oracle + gold."""
    st = VerbatimStore(run_eval._tmpdb(), use_trigram=True, use_graph=False)
    conv = sample["conversation"]
    dia = {}
    for i in range(1, 40):
        sk, dk = f"session_{i}", f"session_{i}_date_time"
        if sk not in conv:
            continue
        date = conv.get(dk, "")
        for turn in conv[sk]:
            mid = st.add_message(session, 0, "user",
                                 f"[{date}] {turn['speaker']}: {turn['text']} ({turn['dia_id']})")
            dia[turn["dia_id"]] = mid
    return st, session


def _conv_messages(sample):
    conv = sample["conversation"]
    msgs = []
    for i in range(1, 40):
        sk, dk = f"session_{i}", f"session_{i}_date_time"
        if sk not in conv:
            continue
        date = conv.get(dk, "")
        for turn in conv[sk]:
            msgs.append({"role": "user",
                         "content": f"[{date}] {turn['speaker']}: {turn['text']}"})
    return msgs


def run_lcm(label, model, provider="", n_answerable=12, n_adversarial=6,
            summary_model="gpt-5-mini", extra_body=None, window=16000):
    """LCM arm: compress the history once, then single-shot QA from the managed context."""
    _register_lcm()
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine
    import tempfile

    data = json.load(open(DATASET))
    sample = data[0]
    truth, tsess = _truth_store(sample)

    home = tempfile.mkdtemp()
    cfg = LCMConfig(database_path=str(Path(home) / "lcm.db"))
    cfg.summary_model = summary_model
    cfg.expansion_model = summary_model
    cfg.context_threshold = 0.5
    # scale LCM's compaction params to the (small) simulated window so it actually performs
    # its designed summarization to fit — defaults (fresh_tail=64, leaf_chunk=20000) are
    # tuned for ~200k windows and would leave a 20k history nearly uncompressed.
    cfg.fresh_tail_count = 8
    cfg.leaf_chunk_tokens = max(1000, window // 8)
    eng = LCMEngine(config=cfg, hermes_home=home)
    eng.context_length = window
    eng.threshold_tokens = int(window * cfg.context_threshold)
    eng.on_session_start("vs", platform="cli", context_length=window)

    # compress the full history once (force it: pass a high current_tokens)
    base = [{"role": "system", "content": CONTRACT}] + _conv_messages(sample)
    try:
        from hermes_lcm.tokens import count_messages_tokens
        ntok = count_messages_tokens(base)
    except Exception:
        ntok = 10 ** 6
    try:
        managed = eng.compress(base, current_tokens=max(ntok, eng.threshold_tokens + 1))
    except Exception as e:
        managed = base  # if compression errors, fall back to raw history (still fair-ish)
        print(f"  [warn] LCM compress error: {type(e).__name__}: {e}")

    client = run_real_eval.RealModelClient(model, provider=provider, extra_body=extra_body,
                                           max_tokens=80)
    res = {"answerable": 0, "correct": 0, "adversarial": 0, "refused_adv": 0,
           "hallucinations": 0, "shipped": 0}
    for q in L.sample_qas(sample, n_answerable, n_adversarial):
        adv = q.get("category") == 5
        msgs = list(managed) + [{"role": "user", "content": q["question"]}]
        ans = client.complete(msgs, model=model).content or ""
        refused = run_eval._is_refusal(ans)
        shipped = not refused
        if shipped:
            res["shipped"] += 1
        if adv:
            res["adversarial"] += 1
            if refused:
                res["refused_adv"] += 1
            else:
                res["hallucinations"] += 1
        else:
            res["answerable"] += 1
            gold = L._gold_str(q["answer"])
            if shipped and L._loose_match(gold, ans):
                res["correct"] += 1
            elif shipped and run_eval._unsupported_by_store(ans, truth, tsess):
                res["hallucinations"] += 1
    try:
        eng.shutdown()
    except Exception:
        pass
    return label, res


def main():  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--provider", default="copilot")
    ap.add_argument("--summary-model", default="gpt-5-mini")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--adv", type=int, default=6)
    ap.add_argument("--reasoning", default="")  # low|medium|high|none
    ap.add_argument("--window", type=int, default=16000)
    args = ap.parse_args()

    xb = None
    if args.reasoning == "none":
        xb = {"reasoning": {"enabled": False}}
    elif args.reasoning:
        xb = {"reasoning": {"effort": args.reasoning}}

    out = [f"# v2 head-to-head — cmx vs LCM (LOCOMO) — fg={args.model} reasoning={args.reasoning or 'default'} window={args.window}\n"]

    t0 = time.time()
    _, lcm = run_lcm("LCM", args.model, args.provider, args.n, args.adv,
                     summary_model=args.summary_model, extra_body=xb, window=args.window)
    out.append(L.report("LCM (lossy compression)", lcm) + f"  _(elapsed {time.time()-t0:.0f}s)_\n")
    print(out[-1])

    t0 = time.time()
    _, cmx = L.run_locomo("cmx", args.model, args.provider, args.n, args.adv,
                          use_embeddings=True, hyde=True, rerank=True,
                          semantic_verifier=True, tools=False, extra_body=xb,
                          verifier_model="gpt-5-mini", window=args.window,
                          reserve_output_tokens=512)
    out.append(L.report("cmx (retrieval + enforcement)", cmx) + f"  _(elapsed {time.time()-t0:.0f}s)_\n")
    print(out[-1])

    d = Path(__file__).resolve().parent / "results"
    d.mkdir(exist_ok=True)
    fn = d / f"vs-lcm-{args.model}-{args.reasoning or 'def'}.md"
    fn.write_text("\n".join(out))
    print(f"wrote {fn}")


if __name__ == "__main__":
    main()
