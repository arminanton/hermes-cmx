# 02 — Grounding Enforcement (the heart of cmx)

> **Premise the whole design rests on:** *A model with retrieval tools but freedom to skip them will skip them.* Therefore no layer here may depend on the model **choosing** to behave. Each layer is an **engine-side forcing function**. Stacked, they make grounding a property of the system, not a hope about the model.

This directly answers the user's question — *"will the model actually obey and retrieve instead of guessing?"* — with: **it doesn't have to, because the engine doesn't give it the option.**

---

## The five layers (defense in depth)

| # | Layer | Removes the failure | Depends on model? |
|---|---|---|---|
| 1 | **Proactive injection** | "model didn't bother to retrieve" | **No** — engine retrieves *for* it |
| 2 | **Forced retrieval gate** | "model answered without looking" | **No** — protocol/tool_choice forced |
| 3 | **Deterministic citation check** | "model cited something that doesn't exist / doesn't say that" | **No** — SQLite lookup |
| 4 | **Independent verification** | "model made an uncited claim not in evidence" | **No** — a *different* process checks |
| 5 | **Refuse-to-guess default** | "model confabulated rather than admit ignorance" | **No** — enforced by 4 |

No single layer is sufficient; together they cover the gaps each leaves.

---

## Layer 1 — Proactive injection (the real "force the model")

The engine does the retrieval the model is supposed to do, **before** the model sees the turn, and places the result in front of it.

```
on_assemble(turn):
    q = extract_query_signals(turn.user_text, recent_window)   # entities, identifiers, quoted strings
    evidence = retrieval_core.hybrid(q, k = profile.k, budget = window_budget_for(2c))
    block = render_evidence(evidence)   # labelled, engine-role, each slice tagged [id=NNN]
    context.insert(block, role = TOOL_OR_SYSTEM)               # NEVER assistant role
```

- Converts *"hope the model retrieves"* → *"the exact text is already on screen."* Most "guessing" disappears because the answer is usually right there.
- `k` and slice size come from the **enforcement profile** and the **window budget** (small window → fewer, higher-precision slices).
- The block is labelled `"[CMX EVIDENCE — verbatim from your history. Cite [id=N] for any claim you draw from these.]"` so Layer 3 has something to check.
- **Limitation it leaves:** retrieval might miss the right slice, or the model might claim something *beyond* the evidence → handled by Layers 2–4.

## Layer 2 — Forced retrieval gate

When injected evidence is insufficient (low fusion scores, or the model signals uncertainty), the model must go *through* retrieval before it can assert a fact.

Mechanisms, in order of preference by provider capability:
- **`tool_choice = "required"`** (or provider equivalent): the next model action *must* be a `cmx_*` retrieval call; it cannot emit a free-form factual answer.
- **Structured two-step**: turn 1 must return either `{retrieve: query}` or `{answer_needs_no_history: true}`; only after retrieval may it answer. Engine rejects malformed/short-circuit responses.
- **Stop-sequence interception**: if the model starts emitting a factual answer without a prior retrieval this turn, the engine cuts generation and re-prompts with the forcing instruction.

- **Applies to factual-history turns only** (a "what's the capital of France" turn needs no history) — classified cheaply by the engine, biased toward forcing when unsure.
- **Limitation it leaves:** the model can still retrieve and then *misread* it → Layers 3–4.

## Layer 3 — Deterministic citation check (provable grounding)

Require factual claims to carry citations `[id=N]`. The engine verifies each **with SQLite, no LLM judgment**:

```
for claim in parse_claims(answer):
    for id in claim.cited_ids:
        row = db.get_message(id)                       # exists?
        if row is None: fail(claim, "phantom_citation")
        if not span_supported(claim.quote, row.content):   # quoted span actually present?
            fail(claim, "citation_mismatch")           # trigram/substring/fuzzy-window match
    record(answer_claims, claim, verified = not failed, method="citation_exact")
on any fail → Layer 2 regenerate (bounded) with: "Citation [id=N] does not support 'X'. Re-answer using only cited evidence or say you don't know."
```

- This is **provable** grounding for anything cited: a claim survives only if a real verbatim row contains the quoted content.
- `span_supported` uses normalized substring + trigram-window fuzzy match (tolerates whitespace/case, not meaning drift).
- **Limitation it leaves:** the model can make a claim and simply *not cite it* → Layer 4.

## Layer 4 — Independent verification pass (catches uncited claims)

A **separate** checker — *not the model that wrote the answer* — audits factual assertions that lack citations or evidence support.

```
claims = extract_factual_history_claims(answer)            # cheap NLI-style or rules
unsupported = [c for c in claims if not supported_by(c, injected_evidence ∪ cited_rows)]
if unsupported:
    if profile.verifier == "deterministic":   # claim references an id/identifier? check store
        resolve_or_fail(unsupported)
    else:                                      # cheap LLM-as-judge (e.g. gpt-5-mini) with the evidence
        verdicts = verifier_model.judge(unsupported, evidence)   # supported / unsupported / needs-retrieval
    on unsupported → Layer 2 regenerate, or Layer 5 downgrade
```

- The verifier is **engine-chosen and may be cheaper/different** from the foreground model — so a strong *or* weak foreground model gets the same audit.
- Runs **only on factual-history turns**; can be async/spot-check on strong-model profiles for latency.
- **Limitation it leaves:** a subtle claim could slip past an imperfect verifier → residual risk, bounded and measured (see `04-EVALUATION.md`). This is why we say *hallucination crushed, not eliminated.*

## Layer 5 — Refuse-to-guess default

The standing contract, enforced by Layer 4 rather than the model's manners:

> If a factual claim about history is neither in retrieved evidence nor verifiable against the store, the answer is **downgraded** to: *"I don't have that in my record — want me to search for it?"* (optionally auto-triggering a deeper `cmx_grep`).

This makes "I don't know" the **safe, enforced** outcome instead of a confident wrong answer.

---

## Worked scenarios (end-to-end)

### Scenario A — "What did we decide about the CI4 migration two days ago?" (strong model, large window)
1. **Inject**: trigram surfaces the exact verbatim turns mentioning `CI4`/`migration`; injected as evidence with ids.
2. Model answers, citing `[id=4821]`.
3. **Citation check**: id 4821 exists, contains the quoted decision → ✔ verified.
4. Verifier spot-check passes. → Ship, audit row written. *No guessing possible — the decision text was on screen and checked.*

### Scenario B — same question, **gpt-5-mini, 32k window** (the hard case)
1. **Window-aware inject**: only top-2 *highest-precision* trigram slices (budget is tiny). Recent window shortened.
2. Profile = **strict**: `tool_choice=required` for this factual turn. Mini tries to answer from memory → engine **rejects**, forces `cmx_grep("CI4 migration decision")`.
3. Engine serves the verbatim slice; mini answers citing `[id=4821]`.
4. **Citation check** ✔. **Mandatory verifier** (separate cheap model) confirms support ✔. → Ship.
   *The weak model could not skip retrieval, could not cite a phantom, and was independently audited — same grounding as the strong model.*

### Scenario C — the answer genuinely isn't in history
1. Inject finds only weak matches (low fusion scores).
2. Forced gate → model retrieves → still nothing relevant.
3. Verifier flags the drafted claim as unsupported.
4. **Layer 5**: answer downgraded to *"That isn't in my record — want me to search wider or check a file?"* → No confabulation.

### Scenario D — retrieval engine returns the wrong slice (precision miss)
1. Injected evidence is off-topic; model answers from it or guesses.
2. **Citation check**: the model's quote isn't in the cited row → fail → regenerate with `cmx_grep` forced (different query expansion).
3. If still unresolved after bounded retries → Layer 5 downgrade. *A precision miss degrades to "I don't know," never to a confident wrong answer.* (This is why retrieval precision is the floor we invest in — see Risks.)

### Scenario E — verifier outage / cost ceiling hit
1. Verifier model unavailable. Engine falls back to **deterministic-only** checks (Layer 3) + Layer 5 for anything uncited.
2. Behavior degrades to "cited-and-checked, otherwise refuse" — strictly safe, never to "summary fragments as truth" (LCM's old failure). Measured by the *failure-under-outage* eval.

---

## Why this beats Self-RAG / "let the model critique itself"

Self-RAG and MemGPG-style self-paging put the retrieve/critique decision **inside the model**. On a strong model that mostly works; on gpt-5-mini, or a strong model under load, it doesn't. cmx moves every one of those decisions **into the engine**: evidence is injected *for* the model, retrieval is *forced*, citations are checked *deterministically*, and answers are audited by a *different* process. The foreground model's cooperation is never on the critical path to grounding — which is exactly the property the user asked for.
