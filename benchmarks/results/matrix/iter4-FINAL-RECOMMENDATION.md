# cmx iter-4 — FINAL RECOMMENDATION (empirically grounded)

Answering the original question — *which combination of iteration 1, 2, 3 (+ the multi-judge idea)
gives the best result, as a deliverable final product* — now backed by live LOCOMO benchmarks on
real models (gpt-5-mini + gemini-2.5-pro foreground, opus-4.7 / gpt-5-mini judges).

## The final-product config

```
RETRIEVAL (iter1 enforcement + iter2 ship levers, held fixed):
  verbatim SQLite store (pysqlite3: FTS5 + trigram)  ·  embeddings (on)  ·  cosine-rerank (on)
  ·  HyDE (on)  ·  semantic-verifier (on)  ·  inject cap=4  ·  provider-aware tokenizer  ·  profiles
GROUNDING (iter1 5-layer enforcement):
  proactive injection → FORCED pre-answer gate → deterministic citation check → independent
  verify → refuse-to-guess
JUDGE  (iter3 L2 sufficiency gate)  ==  a SINGLE capable model, reasoning/decomposition ON,
  refuse-to-guess.  ❌ NOT the multi-persona Council.
FOREGROUND:  any fast/cheap model (gpt-5-mini, gemini-2.5-pro). Small context window is FINE —
  the DB holds verbatim memory and retrieval re-injects it each turn.
```

## What the benchmarks decided (evidence)

**1. The pre-answer gate is mandatory.** Removing it (verify-only) doubled hallucination to ~44%.
A post-hoc verifier cannot catch a confident wrong answer to an unanswerable question — the model
must be stopped *before* it confabulates.

**2. The multi-persona Council is the WRONG judge — rejected.** At n=24 adversarial,
`council:m5cc` as the gate over-refused to **10% answerable accuracy** (vs 70% for a single opus
gate), gained no hallucination reduction, and cost **15× more** (4127s vs 271s). An adversarial
panel's job is to manufacture doubt → it refuses answerable questions too. As a verifier it was no
better than the single judge. → CouncilVerifier stays in the tree **flag-gated OFF**, a
researched-and-rejected path (same rigor as the rejected iter-2/3 levers). (The Council remains
excellent for its actual job — plan/diff/decision review — just not as the cmx grounding judge.)

**3. A CHEAP gate is viable.** gpt-5-mini as the sufficiency judge matched (sometimes beat) the
opus gate at n=16 (e.g. gpt-5-mini fg + gpt-5-mini gate = 50% acc / 93.8% adv-refusal /
**14.3% halluc** — best halluc in the sweep). The earlier "weak judge over-refuses" was small-n
noise; with the reasoning/decomposition gate a cheap judge holds. (Confirm at higher n.)

**4. The "small-context model via the DB" thesis HOLDS.** Pinning a 16k foreground window matched
native across models (e.g. gpt-5-mini @16k = 62.5% acc; gemini @16k = 62.5%) — because cmx stores
everything verbatim in sqlite and re-injects the relevant slices. A fast, small-context, cheap
model can run an arbitrarily long conversation. **Efficiency thesis confirmed.**

**5. Cross-model robustness.** gpt-5-mini and gemini-2.5-pro foreground behaved comparably
(~40–62% acc, ~87–94% adv-refusal, ~17–25% halluc) — the grounding contract is model-agnostic.

**6. The on-topic-but-unanswerable residual (~15–25%) is a HARD LIMIT.** No config — single or
Council, cheap or capable, native or small-window — reached 0% hallucination at n≥16 adversarial.
This empirically **confirms iter-3's honest ceiling** and **refutes** the "multi-judge gets us to
0%" hypothesis. The earlier "0%" was small-n noise.

## Honest statement of the goal

- **"Infinite-context feeling": delivered.** Verbatim-forever + forced retrieval + DB re-injection
  means no felt memory loss, even for a small-context/cheap foreground model.
- **"0% errors": not achievable as an absolute.** Hallucination is *crushed and bounded* (~15–25%
  only on the deliberately-adversarial, on-topic-but-unanswerable slice; normal answerable QA
  grounds well at ~50–70%), but a hard residual remains. Ship with the gate mandatory, a capable
  refuse-to-guess judge, optional L5 self-consistency, and report the residual — never claim 0%.

## Cost/latency (single-judge final config)
~190–245 s per LOCOMO QA set (n=8+16) — ~1/15th of the Council gate. Production-viable.

## Caveats / to lock before production
- Numbers are single-run at n=8 ans / 16–24 adv (each question ≈ 6–12%) → directional, noisy per
  cell. Re-run the chosen config at n≥20 ans + ≥30 adv across 2–3 LOCOMO samples/seeds to finalize.
- Then: merge the bundle to cmx master, one clean LOCOMO gate number, 3-day dogfood vs LCM, switch
  default only on a clean win.

## Artifacts
- Wiring (flag-gated): `src/cmx/council_judge.py`, `config.py`, `engine.py`; tests `tests/test_council_judge.py` (94 green, pysqlite3).
- Benchmarks: `benchmarks/{run_finalists,run_gate_sweep,analyze_matrix}.py` (plus internal council-matrix harness, not shipped);
  results under `benchmarks/results/matrix/` (`iter4-council-matrix.md`, `iter4-FINALISTS.md`, `iter4-GATE-SWEEP.md`).
- CC personas: `council/workspace/personas/persona-roster.yaml` (`ccverifier`, `groundingauditor`).
- Interpreter: always `python3` (pysqlite3 + trigram).

---

## LOCKED — high-n multi-seed confirmation (gpt-5-mini foreground, 60 ans + 84 adv, seeds 0/1/2)

| gate (pooled across 3 LOCOMO conversations) | answer_acc | adv_refusal | hallucination |
|---|---:|---:|---:|
| **opus-4.7 (capable) — RECOMMENDED** | **58.3%** | **95.2%** | **8.9%** |
| gpt-5-mini (cheap) | 41.7% | 89.3% | 22.2% |

**Decision locked:** the **capable gate wins on all three axes** — +16.6pts accuracy, +5.9pts
adversarial-refusal, and **2.5× lower hallucination** (8.9% vs 22.2%). The small-n gate-sweep hint
that "a cheap gate is fine" did NOT survive high-n multi-seed confirmation — it was noise. iter-3's
"the judge must be CAPABLE" is robustly confirmed.

**Revised residual:** with a capable gate the on-topic-but-unanswerable hallucination pools to
**~9%** (per-seed 0% / 6.2% / 17.6%), not the ~20% a single noisy seed implied. So the honest locked
envelope on hard LOCOMO is: **~58% answerable accuracy · ~95% adversarial refusal · ~9% residual
hallucination**, with a fast/cheap foreground model backed by the DB. Hallucination crushed and
bounded to single digits — still not a literal 0%.

**LOCKED FINAL CONFIG (ready to merge):**
- retrieval: iter-1 enforcement + iter-2 embeddings+HyDE+cosine-rerank+cap4+semantic-verifier (pysqlite3 FTS5+trigram)
- gate: iter-3 L2 sufficiency, **single CAPABLE judge (opus-tier), reasoning/decomposition ON, refuse-to-guess** — pre-answer (mandatory), single (NOT Council)
- foreground: any fast/cheap model (gpt-5-mini / gemini-2.5-pro), small context OK (DB holds memory)
- Council multi-judge: rejected as the grounding judge (flag-gated OFF)
- next: set as cmx defaults → merge iter2/iter3 bundle to master → 3-day dogfood vs LCM → switch default on a clean win.
