"""LOCOMO eval gate — cmx on a published multi-session conversational QA benchmark.

Ingests a LOCOMO conversation (multiple dated sessions between two speakers) into
cmx, then runs its QA through the cmx grounding pipeline against real models.
Headline metrics for cmx's claims:
  • refusal-on-adversarial: LOCOMO category 5 questions are UNANSWERABLE (gold=None)
    — cmx must refuse, not confabulate.
  • hallucination rate: any shipped answer asserting a fact absent from the
    ingested conversation.
  • answerable accuracy (loose token match) — secondary (LOCOMO is hard).
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cmx.config import CmxConfig            # noqa: E402
from cmx.engine import CmxEngine            # noqa: E402
from cmx.retrieval import HybridRetriever   # noqa: E402
from cmx.store import VerbatimStore         # noqa: E402
import run_eval                              # noqa: E402
import run_real_eval                         # noqa: E402

DATASET = str(Path(__file__).resolve().parent / "locomo10.json")
_TOK = re.compile(r"[0-9A-Za-z]{3,}")


def ingest_conversation(eng, sample, session="locomo"):
    conv = sample["conversation"]
    n = 0
    for i in range(1, 40):
        skey, dkey = f"session_{i}", f"session_{i}_date_time"
        if skey not in conv:
            continue
        date = conv.get(dkey, "")
        for turn in conv[skey]:
            text = f"[{date}] {turn['speaker']}: {turn['text']} ({turn['dia_id']})"
            eng.ingest(session, "user", text)
            n += 1
    return n, conv.get("speaker_a", "A"), conv.get("speaker_b", "B")


def _gold_str(ans):
    return "" if ans is None else (", ".join(map(str, ans)) if isinstance(ans, list) else str(ans))


def _loose_match(gold: str, answer: str) -> bool:
    g = [t.lower() for t in _TOK.findall(gold)]
    if not g:
        return False
    a = answer.lower()
    hit = sum(1 for t in g if t in a)
    return hit >= max(1, int(0.6 * len(g)))   # ≥60% of gold tokens present


# LLM-judge accuracy scorer — the FIELD STANDARD (Dakera uses GPT-4o; WMB-100K uses a
# GPT-4o-mini/Claude/Gemini majority). Token-match scores a semantically-correct answer
# phrased differently as WRONG; the judge credits it. Temperature 0, binary CORRECT/WRONG.
_JUDGE_CACHE: dict = {}


def _llm_judge_correct(question: str, gold: str, answer: str, judge_client, judge_model: str) -> bool:
    if not gold or not answer:
        return False
    key = (question, gold, answer, judge_model)
    if key in _JUDGE_CACHE:
        return _JUDGE_CACHE[key]
    sys = ("You are grading whether an ANSWER correctly answers a QUESTION, given the GOLD answer. "
           "Reply with exactly CORRECT or WRONG. The answer is CORRECT if it conveys the gold fact, "
           "even if phrased differently, paraphrased, or with extra words; ignore citations like "
           "[id=N]. It is WRONG if it states a different value, contradicts the gold, or refuses. "
           "Do not be lenient about the actual fact (a wrong date/name/number is WRONG).")
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": f"QUESTION: {question}\nGOLD: {gold}\nANSWER: {answer}"}]
    try:
        out = (judge_client.complete(msgs, model=judge_model).content or "").strip().upper()
        verdict = out.startswith("CORRECT") or ("CORRECT" in out and "WRONG" not in out)
    except Exception:
        verdict = False
    _JUDGE_CACHE[key] = verdict
    return verdict


def sample_qas(sample, n_answerable=8, n_adversarial=8):
    ans = [q for q in sample["qa"] if q.get("category") != 5 and q.get("answer") is not None]
    adv = [q for q in sample["qa"] if q.get("category") == 5]
    return ans[:n_answerable] + adv[:n_adversarial]


def run_locomo(label, model, provider="", n_answerable=8, n_adversarial=8,
               use_embeddings=False, tools=False, hyde=False, rerank=False,
               strict_inject_k=0, semantic_verifier=False, verifier_model="",
               extra_body=None, verifier_provider="", verifier_extra_body=None,
               window=0, reserve_output_tokens=0, sufficiency_gate=False, self_consistency=0,
               sufficiency_threshold=0, llm_rerank=False, sufficiency_votes=1,
               sufficiency_reasoning=False, strict_value_grounding=False,
               sufficiency_consensus="unanimous", judge_backend="single",
               council_mode="fast", council_verdict_only=True, council_panel="",
               council_preset="", council_peer_review=False, assume_factual_history=False,
               council_personas=None, sample_idx=0,
               force_memory_directives=False, prompted_tool_protocol=False,
               parse_tool_dialects=False, strip_reasoning=False, agentic_search=False,
               judge_accuracy=False, accuracy_judge_model="claude-opus-4.7",
               temporal_rerank=False, importance_rerank=False,
               temporal_weight=0.15, importance_weight=0.15, multihop=False,
               teach_retrieval=True):
    data = json.load(open(DATASET))
    sample = data[sample_idx]
    cfg = CmxConfig()
    cfg.forced_gate = "auto" if tools else "off"
    cfg.max_regenerations = 1
    cfg.recent_window_turns = 4
    cfg.use_embeddings = use_embeddings
    cfg.rerank = rerank
    cfg.strict_inject_k = strict_inject_k
    cfg.semantic_verifier = semantic_verifier
    cfg.sufficiency_gate = sufficiency_gate
    cfg.self_consistency = self_consistency
    cfg.sufficiency_threshold = sufficiency_threshold
    cfg.llm_rerank = llm_rerank
    cfg.sufficiency_votes = sufficiency_votes
    cfg.sufficiency_reasoning = sufficiency_reasoning
    cfg.strict_value_grounding = strict_value_grounding
    cfg.sufficiency_consensus = sufficiency_consensus
    cfg.teach_retrieval = teach_retrieval
    # iter4: judge backend (single-model Verifier vs Hermes Council multi-judge).
    cfg.judge_backend = judge_backend
    cfg.council_mode = council_mode
    cfg.council_verdict_only = council_verdict_only
    cfg.council_panel = council_panel
    cfg.council_preset = council_preset
    cfg.council_peer_review = council_peer_review
    cfg.council_personas = list(council_personas) if council_personas else []
    # LOCOMO/OOLONG QA is entirely about conversation history → gate every turn.
    cfg.assume_factual_history = assume_factual_history
    # forcing layer (prompt-level grounding-enforcement blocks). All flag-gated, default off.
    cfg.force_memory_directives = force_memory_directives
    cfg.prompted_tool_protocol = prompted_tool_protocol
    cfg.parse_tool_dialects = parse_tool_dialects
    cfg.strip_reasoning = strip_reasoning
    cfg.agentic_search = agentic_search
    # iter4 retrieval accuracy levers (deterministic, Dakera-style).
    cfg.temporal_rerank = temporal_rerank
    cfg.temporal_weight = temporal_weight
    cfg.importance_rerank = importance_rerank
    cfg.importance_weight = importance_weight
    cfg.multihop = multihop
    if reserve_output_tokens:
        cfg.reserve_output_tokens = reserve_output_tokens
    if verifier_model:
        cfg.verifier_model = verifier_model
    # model registry: optionally pin a (small) window to simulate a long-conversation /
    # constrained-context regime shared with the LCM arm.
    _m = {}
    if window:
        _m["window"] = window
    if "mini" not in model:
        _m["follows_instructions"] = "high"
    cfg.models = {model: _m} if _m else {}
    store = VerbatimStore(run_eval._tmpdb(), use_trigram=True, use_graph=True)
    embedder = None
    if use_embeddings or rerank:
        from cmx.embeddings import HermesEmbedder
        embedder = HermesEmbedder()
    expander = None
    if hyde:
        from cmx.expand import keyword_expander, HydeExpander, ChainExpander
        expander = ChainExpander(keyword_expander,
                                 HydeExpander(run_real_eval.RealModelClient("gpt-5-mini"),
                                              model="gpt-5-mini"))
    client = (run_real_eval.ToolModelClient(model, provider=provider, extra_body=extra_body) if tools
              else run_real_eval.RealModelClient(model, provider=provider, extra_body=extra_body))
    # an independent judge is needed for the semantic verifier (lever 6) and to keep the
    # tool loop (lever 2) honest; always a SEPARATE cheap model from the foreground one.
    verifier_client = (run_real_eval.RealModelClient(cfg.verifier_model, provider=verifier_provider,
                                                     extra_body=verifier_extra_body)
                       if (semantic_verifier or tools or sufficiency_gate
                           or judge_backend == "council") else None)
    reranker = None
    if cfg.llm_rerank:
        from cmx.rerank import LLMReranker
        reranker = LLMReranker(run_real_eval.RealModelClient(cfg.llm_rerank_model, provider=provider),
                               cfg.llm_rerank_model, pool=20)
    eng = CmxEngine(cfg, store,
                    HybridRetriever(store, cfg, embedder=embedder, expander=expander,
                                    reranker=reranker),
                    client, verifier_client=verifier_client)
    _, spk_a, spk_b = ingest_conversation(eng, sample)

    tool_line = (" If the [CMX EVIDENCE] is insufficient, FIRST call cmx_grep with a reworded "
                 "query (use synonyms/specifics) to search the full history; only refuse after searching."
                 if tools else "")
    contract = {"role": "system", "content":
                f"You answer questions about a conversation between {spk_a} and {spk_b}, using ONLY "
                "the [CMX EVIDENCE] verbatim excerpts shown. Cite the source like [id=N] using the "
                "number shown in the evidence for any historical fact you state; never invent an id." + tool_line +
                " If the answer is not in the evidence, reply EXACTLY: I don't have that in my record. "
                "One short answer."}

    res = {"answerable": 0, "correct": 0, "correct_judge": 0, "adversarial": 0, "refused_adv": 0,
           "hallucinations": 0, "shipped": 0}
    _judge_client = (run_real_eval.RealModelClient(accuracy_judge_model, provider="")
                     if judge_accuracy else None)
    for q in sample_qas(sample, n_answerable, n_adversarial):
        adv = q.get("category") == 5
        resp = eng.respond("locomo", q["question"], model=model, system_prompt=contract["content"],
                           persist=False)
        refused = resp.refused or run_real_eval.run_eval._is_refusal(resp.text)
        shipped = not refused
        if shipped:
            res["shipped"] += 1
        if adv:
            res["adversarial"] += 1
            if refused:
                res["refused_adv"] += 1
            else:
                res["hallucinations"] += 1            # confabulated an unanswerable question
        else:
            res["answerable"] += 1
            gold = _gold_str(q["answer"])
            tok_ok = shipped and _loose_match(gold, resp.text)
            if tok_ok:
                res["correct"] += 1
            elif shipped and run_eval._unsupported_by_store(resp.text, store, "locomo"):
                res["hallucinations"] += 1            # asserted a fact not in the conversation
            # LLM-judge accuracy (field standard) tallied in parallel — credits paraphrases.
            if judge_accuracy and shipped and _llm_judge_correct(
                    q["question"], gold, resp.text, _judge_client, accuracy_judge_model):
                res["correct_judge"] += 1
    return label, res


def report(label, r):
    p = lambda a, b: (100.0 * a / b) if b else 0.0
    return (f"### {label}\n"
            f"- answerable accuracy:    {p(r['correct'], r['answerable']):5.1f}%  ({r['correct']}/{r['answerable']})\n"
            f"- refusal on adversarial: {p(r['refused_adv'], r['adversarial']):5.1f}%  ({r['refused_adv']}/{r['adversarial']})\n"
            f"- hallucination rate:     {p(r['hallucinations'], max(1, r['shipped'])):5.1f}%  ({r['hallucinations']}/{r['shipped']} shipped)\n")


def ab_council_main():  # pragma: no cover
    """iter-4 A/B: single-model sufficiency judge vs the Hermes Council multi-judge.

    Holds the maxed keeper bundle (embeddings + HyDE + cosine-rerank + semantic-verifier +
    L2 sufficiency gate) FIXED and varies ONLY the judge backend, so the delta isolates the
    judge. The Council arm uses its own lane — set, e.g.:

        COUNCIL_SRC=$COUNCIL_SRC \
        COUNCIL_WORKSPACE=$COUNCIL_WORKSPACE \
        COUNCIL_PROVIDER=hermes COUNCIL_MODEL=claude-opus-4.7 COUNCIL_DAEMON=0 \
        PYTHONPATH=$COUNCIL_SRC/libs \
        python benchmarks/run_locomo_eval.py --ab-council --n=12

    One capable model (opus-4.7) wears every Council soul (single-model, many-souls).
    """
    import os
    n_ans, n_adv = 12, 12
    for a in sys.argv:
        if a.startswith("--n="):
            n_ans = int(a.split("=")[1])
        if a.startswith("--adv="):
            n_adv = int(a.split("=")[1])
    council_mode = "fast"
    council_peer = "--peer" in sys.argv
    if "--deep" in sys.argv:
        council_mode = "deep"
    models = [("gpt-5-mini", "gpt-5-mini", ""), ("opus-4.7", "claude-opus-4.7", "")]
    if "--mini-only" in sys.argv:
        models = [("gpt-5-mini", "gpt-5-mini", "")]
    # maxed keeper bundle, held fixed across both arms
    maxed = dict(use_embeddings=True, rerank=True, hyde=True, semantic_verifier=True,
                 sufficiency_gate=True, sufficiency_reasoning=True, assume_factual_history=True,
                 n_answerable=n_ans, n_adversarial=n_adv)
    out = [f"# iter-4 A/B — single judge vs Hermes Council (maxed bundle, {n_ans} ans + {n_adv} adv)\n",
           f"_Council lane: {os.environ.get('COUNCIL_PROVIDER','?')}/{os.environ.get('COUNCIL_MODEL','?')} "
           f"· mode={council_mode}{' +peer' if council_peer else ''} · single-model, many-souls_\n"]
    for label, model, provider in models:
        out.append(f"\n## foreground = {label}\n")
        for backend in ("single", "council"):
            t0 = time.time()
            try:
                lbl, r = run_locomo(f"{label} · judge={backend}", model, provider,
                                    judge_backend=backend, council_mode=council_mode,
                                    council_peer_review=council_peer, **maxed)
                out.append(report(lbl, r) + f"  _(elapsed {time.time()-t0:.0f}s)_\n")
            except Exception as e:  # noqa: BLE001
                out.append(f"### {label} · judge={backend}\n- ERROR {type(e).__name__}: {e}\n")
            print(out[-1])
    d = Path(__file__).resolve().parent / "results"
    d.mkdir(exist_ok=True)
    (d / "iter4-council-ab.md").write_text("\n".join(out))
    print("wrote benchmarks/results/iter4-council-ab.md")


def main():  # pragma: no cover
    if "--ab-council" in sys.argv:
        return ab_council_main()
    models = [("opus-4.7", "claude-opus-4.7", ""), ("gpt-5-mini", "gpt-5-mini", "")]
    use_emb = "--embeddings" in sys.argv
    use_tools = "--tools" in sys.argv
    n_ans = 16
    for a in sys.argv:
        if a.startswith("--n="):
            n_ans = int(a.split("=")[1])
    tag = ("+embeddings" if use_emb else "lexical") + ("+tools" if use_tools else "")
    out = [f"# cmx LOCOMO eval gate ({tag}) — {n_ans} answerable + 6 adversarial (real models)\n"]
    for label, model, provider in models:
        t0 = time.time()
        try:
            lbl, r = run_locomo(label, model, provider, n_answerable=n_ans, n_adversarial=6,
                                use_embeddings=use_emb, tools=use_tools)
            out.append(report(lbl, r) + f"  _(elapsed {time.time()-t0:.0f}s)_\n")
        except Exception as e:
            out.append(f"### {label}\n- ERROR {type(e).__name__}: {e}\n")
        print(out[-1])
    d = Path(__file__).resolve().parent / "results"
    d.mkdir(exist_ok=True)
    fname = f"locomo-gate-{tag.replace('+','_')}.md"
    (d / fname).write_text("\n".join(out))
    print(f"wrote benchmarks/results/{fname}")


if __name__ == "__main__":
    main()
