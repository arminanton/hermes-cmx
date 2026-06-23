# 04 — Evaluation

> The rule (carried from the user's standing practice): **no default switch without numbers.** cmx replaces LCM only after it wins the suite below on **both** a strong and a weak model, then survives ≥ 3 days of real daily-workflow dogfooding.

## 1. What we measure (and why it maps to the complaint)

| Metric | Definition | Targets the failure |
|---|---|---|
| **Hallucination rate** | % of factual-history answers containing an unsupported claim (judged vs. the verbatim store) | the core "hallucinates a lot" |
| **Refusal-to-guess accuracy** | on questions whose answer is *not* in history: % correctly refused (vs. confabulated) | confident wrong answers |
| **Citation validity** | % of cited claims whose quote is actually in the cited verbatim row (deterministic) | provable grounding (Layer 3) |
| **Recall@k** | planted-fact retrievable in top-k slices | "forgets everything" (retrieval floor) |
| **Multi-hop accuracy** | N-hop questions ("file changed after X decided Y about Z") | scattered-fact recall |
| **Cross-session recall** | facts planted across sessions, asked after `/new` ×3 | continuity / "knows all" |
| **Identifier recall** | exact recall of code symbols / IDs / paths (trigram's job) | code/log heavy work |
| **Tokens/turn** | active-prompt tokens at turn T | efficiency bound (non-goal N2) |
| **Latency p50/p95** | per-turn wall-clock incl. retrieval + verify | usability |
| **Failure-under-outage** | behavior when summarizer/verifier/embeddings are down | must degrade to "refuse," never to confabulation |

## 2. The strong-vs-weak matrix (the model-agnosticism proof)

Every metric is run on **both**:
- **Strong**: `claude-opus-4.7` (or `gpt-5.5`).
- **Weak**: `gpt-5-mini` (and a `_default`/small-window stand-in for the ≤32k regime).

**The defining success criterion:** the *gap* between strong and weak on **hallucination rate** and **refusal-to-guess** must be **small** — that is the empirical proof that grounding is engine-enforced, not model-supplied. A design that only works on opus has failed the user's requirement.

## 3. Pass gates (cmx vs. LCM baseline)

| Test | LCM baseline (expected) | cmx gate |
|---|---|---|
| Hallucination rate (strong) | high | **≤ ½ of LCM** |
| Hallucination rate (weak) | very high | **≤ LCM-strong** (weak cmx ≥ strong LCM) |
| Refusal-to-guess (both) | low | **≥ 90%** |
| Citation validity (both) | n/a | **≥ 98%** |
| Recall@5 (identifier set) | moderate (FTS only) | **≥ 95%** |
| Multi-hop | low | **≥ 80%** |
| Cross-session after /new×3 | low (purges) | **≥ 90%** |
| Tokens/turn | low | **≤ 1.5× LCM** (we pay for evidence+audit, bounded) |
| Latency p95 | low | **≤ 2× LCM** (verification cost, bounded; sampled on strong) |
| Failure-under-outage | confabulates (pre-patch) | **0 confabulations**; refuses or cites-only |

Strong↔weak hallucination gap: **≤ 10 percentage points** (the agnosticism proof).

## 4. Datasets

- **Synthetic planted-fact corpus** (primary, deterministic): generate 1k–N-turn conversations with K planted facts (decisions, identifiers, numbers, reversals), ask each back verbatim and multi-hop at the end and across `/new`. Fully labelled → exact scoring. Includes *reversal* traps (a decision later overturned) to test that cmx surfaces the *current* truth, not a stale one.
- **OOLONG** — the benchmark the LCM paper itself uses; gives a directly comparable long-context aggregation/reasoning number against LCM/Claude-Code.
- **LOCOMO** (Mem0's methodology) — multi-session long-conversation recall.
- **RULER / NIAH** — needle-in-haystack across window sizes, to validate the window strategies in `03-ADAPTIVITY.md` (esp. lost-in-the-middle on the 1M model).
- **A held-out slice of the user's real `lcm.db`** — replay genuine long sessions that previously hallucinated; the most honest signal.

## 5. Harness

- `cmx eval --suite <name> --model <m> --baseline lcm` → writes `benchmarks/results/<date>-<model>.md` + machine-readable JSON.
- Deterministic where possible (planted facts, citation validity = exact). LLM-as-judge only for fuzzy hallucination scoring, with the judge ≠ the model under test.
- Every gate result is checked into `benchmarks/results/` and shown to the user **before** any default-switch recommendation.

## 6. Ablations (to justify each layer's cost)

Run the suite with each enforcement layer toggled off, to prove each earns its keep:
- − Layer 1 (no proactive injection) → expect hallucination ↑, refusal ↓.
- − Layer 2 (no forced gate) → expect weak-model hallucination ↑ sharply (the agnosticism gap widens).
- − Layer 3 (no citation check) → expect citation validity ↓, phantom-cite hallucinations.
- − Layer 4 (no verifier) → expect uncited-claim hallucinations ↑.
- − trigram / − embeddings → expect identifier-recall and recall@k ↓ (quantifies the retrieval floor).

These numbers tell us where to spend effort and what we may simplify.
