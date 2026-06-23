# 03 — Adaptivity: Model Profiles & Context-Window Strategies

Two of the user's hard constraints:
1. *"a mechanism that … double checks the model is actually doing what it needs, regardless if it is a gpt-5.5 model or gpt-5-mini"* → **enforcement profiles**.
2. *"some models have lower context window, so it is important to have this handling right otherwise we are wasting the implementation"* → **window-aware budgeting**.

Both are solved in the engine. The **grounding contract never changes**; only *how hard* the engine pushes and *how much* it can show.

---

## 1. The model capability registry

A config-driven table the engine consults to pick a profile. Resolved per active model (foreground) and per verifier model.

```yaml
cmx:
  models:
    "gpt-5.5":            { window: 400000, follows_instructions: high,  forced_tools: yes, cost: high,   tokenizer: o200k }
    "claude-opus-4.7":    { window: 200000, follows_instructions: high,  forced_tools: yes, cost: high,   tokenizer: anthropic }
    "gemini-3.1-pro":     { window: 1000000,follows_instructions: high,  forced_tools: yes, cost: med,    tokenizer: gemini }
    "gpt-5-mini":         { window: 128000, follows_instructions: low,   forced_tools: yes, cost: low,    tokenizer: o200k }
    "gpt-5.4-mini":       { window: 128000, follows_instructions: low,   forced_tools: yes, cost: low,    tokenizer: o200k }
    "_default":           { window: 32000,  follows_instructions: low,   forced_tools: no,  cost: low,    tokenizer: heuristic }
  verifier_model: "gpt-5-mini"        # the independent auditor (Layer 4); cheap on purpose
```

`window` and `tokenizer` drive budgeting; `follows_instructions`, `forced_tools`, `cost` drive the enforcement profile. `_default` is deliberately **paranoid** — an unknown model is treated as weak and small.

## 2. Enforcement profiles (Layer strictness)

| Profile | Triggered when | L1 inject | L2 forced gate | L3 citations | L4 verifier | L5 refuse |
|---|---|---|---|---|---|---|
| **Strict** | weak instruction-following OR small window OR unknown model | **max** within budget, highest-precision | **on** for all factual-history turns | **required** | **mandatory** (separate model) | on |
| **Balanced** | mid models | on | **on** for factual-history turns | required | **spot-check** (sampled) | on |
| **Light** | strong + large window | on (bigger budget) | optional (only on low-confidence retrieval) | required for factual claims | **async/sampled** | on |

Key properties:
- **gpt-5-mini → Strict**: it *cannot* skip retrieval (forced gate), *cannot* cite a phantom (deterministic check), and is *always* independently audited. That is how a weak model gets strong-model grounding.
- **gpt-5.5 → Light**: same guarantees, less overhead (verification sampled/async) for speed and cost — because a strong model needs the engine to *catch* mistakes less often, not because it's trusted.
- Profiles are **overridable** per session/task (e.g., force Strict during high-stakes work regardless of model).

## 3. Context-window strategies (budget allocation)

The engine computes a **real** token budget using the **provider-aware tokenizer** (the LCM single-`cl100k` bug is fixed here — wrong counts mean wrong budgets means a wasted implementation, exactly the user's worry). It then allocates in strict priority order:

```
budget = model.window − reserve_for_output − safety_margin
allocate, in order, trimming the lowest priority first to fit:
   1. Pinned sinks        (system + IDENTITY + active task)        — never trimmed
   2. Retrieved evidence  (top-K verbatim slices)                  — trimmed by lowering K
   3. Recent verbatim     (last N turns)                           — trimmed by lowering N
   4. Headnote pointers   (navigation)                             — dropped first
```

> Note the ordering choice: **evidence outranks the recent window**. For grounding, the *retrieved-for-the-question* slice matters more than raw recency — recency is already partly covered by the model's own short-term coherence, while the specific historical fact is not.

### 3.1 Tiny window (≤ 32k, e.g. small/local models)
- **Cannot show much** → retrieval **precision is everything**. K = 1–3 *highest-fusion* slices; aggressive de-dup; chunk-level (not whole-message) slices.
- Recent window kept short; big tool outputs **externalized** to disk with refs (reuse LCM's externalizer).
- Lean **harder on Layers 3–4**: since you can't display much evidence, you *verify* more. Forced gate always on.
- This is the regime where **trigram + embeddings precision** pays for itself most — you get essentially one shot at the right chunk.

### 3.2 Medium window (128k–200k, e.g. gpt-5-mini, opus-4.7)
- Comfortable evidence injection (K ~ 5–10) + a real recent window.
- Balanced/Strict profile depending on `follows_instructions`.

### 3.3 Large window (≥ 1M, e.g. gemini-3.1-pro)
- More evidence + larger recent window, **but still verify**: large windows suffer **"lost in the middle"** (Liu et al. 2023) and context rot — a fact present at position 400k may be ignored. So Layer 1 still *places* the evidence near the prompt edges (sink-adjacent), and Layer 3 still checks citations. A big window is **not** a substitute for grounding.

## 4. Verifier model selection (Layer 4)

- The verifier is **engine-chosen, independent of the foreground model**, and intentionally cheap (`gpt-5-mini` default). A cheap auditor checking a strong model's factual claims against provided evidence is a *much* easier task than original generation, so it is both affordable and reliable.
- If the foreground model *is* the cheap model, the verifier still differs in **role and prompt** (it sees only the claim + evidence, judged in isolation) — reducing correlated errors.
- Deterministic verification (claim references an id/identifier present in the store) is preferred whenever the claim is checkable without judgment.

## 5. What "getting it right" prevents (the user's "wasting the implementation")

| Mistake | Consequence | cmx guard |
|---|---|---|
| Wrong token count (single tokenizer) | budget mis-set → overflow or under-fill | provider-aware tokenizer per model |
| Showing too much on a small model | evidence pushed out / lost-in-middle | priority budget + precision-first retrieval |
| Trusting a weak model to retrieve | guessing returns | Strict profile: forced gate + mandatory verify |
| Trusting a big window to "just work" | lost-in-middle hallucination | sink-adjacent placement + citation check |
| Verifier as strong/expensive as the model | cost/latency blowup | cheap independent verifier; sampled on strong models |

The result: **the same grounding guarantee from a 32k local model up to a 1M frontier model**, with overhead scaled to need.
