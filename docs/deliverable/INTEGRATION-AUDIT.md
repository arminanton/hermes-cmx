# cmx ↔ Hermes integration audit (2026-06-14)

Triggered by the user's session-id-rotation concern. Compared the host's actual calls
(`agent/context_engine.py` ABC + `conversation_compression.py` call sites) against the cmx shim
(`src/cmx/hermes_engine.py`). Found and FIXED three real bugs; the rest of the surface is sound.

## FIXED

### 1. Session lineage lost on compaction rollover  — CRITICAL (install-blocking)
Hermes rotates `session_id` on compaction (`conversation_compression.py:509`) and notifies the
engine: `on_session_start(new, boundary_reason="compression", old_session_id=old)`. cmx **ignored**
those kwargs and all retrieval is `WHERE session_id=?`, so after the first compaction cmx queried a
brand-new id with no rows → **the entire prior conversation went dark, exactly when the retrieval
store matters most.** Fix: `session_lineage` table + `root_session()`/`link_session()`; the shim
normalizes every child id to its lineage ROOT and stores/retrieves under it. Survives restarts
(a resumed child resolves to its root). Tests: `test_session_lineage.py`.

### 2. Incremental-ingest positional cursor breaks across compaction  — HIGH
cmx ingested via `non_system[self._ingested:]` — a positional cursor that assumes `messages` is an
ever-growing list. But the host hands back the **compacted** set (which cmx itself produced) after a
compaction, so post-compaction new turns were **missed** (cursor > list length) or **duplicated**
(`add_message` never deduped). Fix: ingest is now **content-hash dedup** (`content_hashes()` +
`hash_content()`), idempotent and rollover-safe. Trade-off: byte-identical short turns collapse —
acceptable for a fact-retrieval store.

### 3. Short sessions never captured; model-switch staleness  — LOW
- cmx only ingested inside `compress()`/tool calls, so a conversation that never hit the compaction
  threshold stored nothing. Added `on_session_end()` → final flush (dedup-safe).
- `update_model` (ABC default) refreshed `context_length` but not `self.model`, leaving cmx's
  tokenizer stale after a model switch. Added a 2-line override.

### 4. SQLite connection pinned to its creating thread  — CRITICAL (crash + silent memory loss)
- `db.connect()` opened the store's single long-lived connection without `check_same_thread=False`,
  so the moment the host used the engine from a different thread than the one that built it (hosts
  run each turn via `asyncio.to_thread` on a *pooled* worker thread; compaction can fire on another),
  any `self.conn` call raised `pysqlite3.ProgrammingError: SQLite objects created in a thread can only
  be used in that same thread`. Reproduced directly (cross-thread `content_hashes()` crashed). Effect:
  the turn crashed (e.g. deepseek-r1) or, where an outer `try/except` swallowed it, retrieval silently
  degraded to empty → long conversations "lost all memory". Found via another agent running cmx under
  a threaded host.
- FIX: `db.connect()` now sets `check_same_thread=False` and returns a `_LockedConnection` wrapper that
  serializes every statement with a reentrant lock (append-only + WAL ⇒ this is sufficient and
  race-free). Verified: original repro passes; 8–10 threads doing 500 concurrent writes+reads → 0
  errors, all rows present. Regression-tested in `tests/test_thread_safety.py`.

## REVIEWED — OK (safe inherited defaults)
- `should_compress_preflight` / `should_defer_preflight_to_real_usage` — default False; cmx uses
  real-usage `should_compress`. Fine.
- `has_content_to_compress` — default guard for the `/compress` command. Fine.
- `pre_send` — H2 hook; cmx injects in `compress()` instead. Fine (full H2 still pending, by design).
- `on_session_reset` (/new, /reset) — cmx correctly resets counters and never purges verbatim.

## STILL PENDING (known, by design — not a regression)
- **Full enforcement loop (H2) — Option A is now LIVE**, promoted to `src`
  (`conversation_loop.py:~3874-3892` + `context_engine.py` ABC hooks): after every reply the host
  calls `enforce_response`, which for cmx refuses/replaces an ungrounded answer (strict no-op for
  LCM/compressor). Only **Option B** (corrective *regenerate* loop) remains unwired — a value-add,
  not a safety gap, since Option A already refuses rather than ship a guess.

## Net
4 bugs found and fixed (2 continuity + 1 capture/staleness + 1 thread-affinity); 123 tests pass
(incl. `tests/test_thread_safety.py`). The continuity bugs that would have made cmx "forget" long
conversations, and the thread-affinity bug that crashed turns / silently dropped memory under a
threaded host, are fixed and regression-tested — cmx is safe to install for the storage+retrieval
dogfood with in-host grounding enforcement (Option A) live.

---
## Bug 5 — H2 over-refusal on legitimate prompts (global-engine regression, 2026-06-14)
**Symptom:** copilot/opus-4.8 replaced a good persona-setup answer with the cmx REFUSAL text.
**Root cause:** `classify_factual_history` matched bare "you"/"we" (in nearly every prompt) and
was "biased toward True" on the now-false assumption that a false positive only costs "overhead".
Once cmx became the GLOBAL engine with live H2 answer-replacement, a false positive REPLACES a
legitimate answer with a refusal. The 137-line persona block tripped it → sufficiency gate found
no evidence → refusal.
**Fix:** (a) classifier now requires a genuine PAST-REFERENCE signal (not bare you/we) + a 600-char
length guard (instruction/persona/paste blocks are never recall questions); (b) enforce_response
grounds the verifier against cmx's actual retrieved knowledge (session + durable), so a correct
durable answer isn't refused for evidence the verifier couldn't see. Refuse-to-guess still fires on
genuine no-evidence recall. +2 regression tests; 130 pass. Benchmark numbers unchanged
(benchmark uses assume_factual_history, which bypasses the classifier). NEEDS RESTART to apply.
