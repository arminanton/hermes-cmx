# 07 — Host Integration (H2): wiring the grounding hooks into Hermes

> **STATUS (2026-06-14): Option A is PROMOTED and LIVE in `src`.** The ABC hooks
> (`agent/context_engine.py:237-258`, no-op defaults) and the Option-A enforcement
> wiring (`agent/conversation_loop.py:~3874-3892`) are in live `src`. Verified: with
> `context.engine: cmx`, the host selects the cmx plugin engine and calls
> `enforce_response` after every reply; an ungrounded factual-history answer is
> replaced in-host with the refuse-to-guess message (activation check, 2026-06-14).
> Strict no-op for the built-in compressor and LCM (guarded by `capabilities()`).
> Remaining: **Option B** (corrective regenerate loop) is the only unwired refinement.

This is the **final, review-gated** integration step. The additive `ContextEngine`
hooks are already implemented and committed on the host worktree branch
`feat/cmx-host-hooks` (`agent/context_engine.py`), with **no-op defaults so
ContextCompressor and LCM are unaffected** (verified). What remains is calling them
from `agent/conversation_loop.py`. It is documented rather than promoted because
that file drives every live turn across all providers and must be validated in live
multi-provider sessions before promotion — consistent with the standing
isolated-worktree / review-before-promote workflow.

## The two hooks

| Hook | When | cmx behavior |
|---|---|---|
| `engine.pre_send(messages, model)` | immediately before the model API call | (optional) inject verbatim evidence every turn; cmx currently injects in `compress()`, so this can stay a no-op initially |
| `engine.enforce_response(answer, messages, model, final=?)` | immediately after the assistant reply is finalized | runs Layer 3 (deterministic citation check) + Layer 4 (verify vs assembled verbatim) and returns a directive |

`enforce_response` returns one of:
- `{"action": "accept"}` — ship as-is
- `{"action": "regenerate", "message": <correction>}` — re-prompt the model
- `{"action": "replace", "text": <refusal>}` — ship a safe replacement

All calls are **guarded by `engine.capabilities().get("enforce_response")`**, so the
wiring is a strict no-op for engines that don't advertise it (compressor, LCM).

## Option A — minimal, safe (recommended first): replace-only

A single contained insertion at the response-finalization point (near
`agent/conversation_loop.py:3697`, where `assistant_message` is normalized and its
string content is resolved). No control-flow loop-back, so lowest risk:

```python
# after the assistant reply's string content is finalized (≈ conversation_loop.py:3697+)
_eng = getattr(agent, "context_compressor", None)
if _eng is not None and _eng.capabilities().get("enforce_response") and isinstance(content, str) and content:
    _verdict = _eng.enforce_response(content, messages, model=agent.model, final=True)
    if _verdict.get("action") == "replace":
        content = _verdict["text"]            # ungrounded answer → safe refusal
    # 'regenerate' is treated as 'replace' in Option A (no loop-back yet)
```

Effect: any ungrounded factual-history answer is replaced in-host by the
refuse-to-guess text. Strong safety property, no loop-back complexity. This is the
right first promotion.

## Option B — full loop: regenerate then refuse

Wrap the model call + response-finalization in a bounded loop so a corrective
re-prompt is possible before refusing. Pseudocode around the existing API-call site:

```python
correction = None
for attempt in range(cfg_max_regen + 1):                  # e.g. 2
    msgs = list(messages)
    if _eng and _eng.capabilities().get("pre_send"):
        msgs = _eng.pre_send(msgs, agent.model)
    if correction:
        msgs.append({"role": "system", "content": correction})
    response = call_model(msgs, ...)                       # existing call path
    content = finalize_assistant_content(response)         # existing normalization
    if not (_eng and _eng.capabilities().get("enforce_response")):
        break
    final = attempt == cfg_max_regen
    v = _eng.enforce_response(content, msgs, model=agent.model, final=final)
    if v["action"] == "accept":
        break
    if v["action"] == "replace":
        content = v["text"]; break
    correction = v["message"]                              # regenerate
```

This reproduces the fully-tested standalone `CmxEngine.respond()` loop in-host.
Higher value (correct-then-ship), higher integration surface — promote only after
Option A is proven and live multi-provider sessions are clean.

## Promotion checklist (per standing workflow)
1. Land hooks in live `agent/context_engine.py` (already on branch; trivially safe — additive no-ops).
2. Wire **Option A** at `conversation_loop.py:~3697` in an isolated worktree.
3. Run the existing host test suite (no regressions for compressor/LCM — they don't advertise the capability).
4. Live `hermes -z` smoke on each provider (opus-4.7 / sonnet / gpt-5.5 / gpt-5-mini) with `context.engine: cmx`.
5. Only then promote; revisit Option B as a follow-up.

## What is already done (no host changes needed)
- cmx is selectable as `context.engine: cmx` via a plugin (`__init__.register`) — **live-tested**: it registered, injected verbatim evidence on compaction, and a real opus-4.7 grounded its answer with a citation (`benchmarks/live_inhost_test.py`).
- The full enforcement loop is implemented and proven in `CmxEngine` (standalone) and on 4 real models (`benchmarks/run_real_eval.py` → hallucination 0% on all).
- `CmxContextEngine.enforce_response` / `capabilities` are implemented and unit-tested — the engine is ready for whichever wiring option is promoted.
