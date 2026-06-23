# Investigation: external LOCOMO/memory-benchmark claims (2026-06-13)

Investigated three sources the user flagged. Verdict: **they largely VINDICATE cmx's approach and
expose two fixable gaps in OUR benchmark — and they show "90–100% on LOCOMO" is mostly a
scoring/methodology artefact, not a real bar we're failing to clear.**

## Source 1 — MemPalace "100% on LoCoMo" (Reddit r/ML) → DEBUNKED
- The "100% R@10" was an **evaluation artefact**: they set `top_k=50` > the total sessions per
  conversation (19–32), so the retriever returned **every** session → all gold evidence is present
  by construction. It measured the reranker/LLM, not retrieval. (Confirmed by community audit +
  the team's own README correction; their LongMemEval "100%" was overfit — held-out 98.4%, raw
  mode 96.6% on default ChromaDB embeddings.)
- **Lesson for us:** anyone reporting ~100% on LOCOMO is almost certainly gaming the metric or
  overfitting. cmx's honest ~58–66% **with measured 0–9% hallucination** is more credible than a
  headline "100%". Do not chase 100%; it's a red flag, not a target.

## Source 2 — Dakera (88.2% on LoCoMo) → reveals OUR two methodology gaps
- LOCOMO = 50 conversations, **1,540 questions**, 4 categories (direct / multi-hop / temporal /
  open-domain). Dakera scores **88.2% overall with a GPT-4o LLM judge**, default config, full set.
- They explicitly warn: **"two percentage points on a 100-question eval is statistically
  meaningless."** We ran 8–24 adversarial questions per cell → our per-cell numbers ARE noise (we
  saw this directly: single-seed 0% vs 20% halluc).
- Their architecture = **hybrid retrieval (vector + BM25) + temporal re-ranking + importance
  weighting.** Temporal is their hardest category (73.9%).

## Source 3 — WMB-100K (Wontopos) → INDEPENDENTLY VALIDATES cmx's thesis
- Critiques the whole field: **"every vendor scores themselves using different evaluation methods,
  then claims #1."** Key fact: **LOCOMO's official metric (Token-Overlap F1) gives GPT-4 with FULL
  CONTEXT only 32.1%** — yet vendors self-report 60–90% using their own lenient methods.
- **WMB-100K penalizes false memory at −0.25 (2.5× the +0.1 reward)** because "a false memory =
  confidently returning wrong info — dangerous… a missing memory = 'I don't know' — inconvenient
  but safe." **This is cmx's refuse-to-guess / 0%-hallucination thesis, verbatim.** The most
  rigorous new benchmark rewards exactly what cmx was built to do.
- Adds a **speed penalty** (latency tiers) — which is why the Council judge (15× slower) was the
  wrong call, now externally corroborated.
- Finds accuracy **collapses toward zero at 100K turns**; "systems that score 66% at 600 turns
  flatline at 100K." cmx's verbatim-store-never-summarize architecture is the design meant to
  resist this — untested at scale so far (LOCOMO is only 600 turns / ~50K tokens).

---

## What this means for cmx (reframing the user's "stuck at 60%, wanted 90%")

1. **The 90–100% bar is largely a SCORING-METHOD illusion.** Official LOCOMO F1 = 32% for
   GPT-4-full-context. The 88–90% figures use **lenient LLM judges** over the **full 1,540-question
   set**. We use **harsh loose-token-match** over a **tiny sample**. We are not "behind by 30
   points" — we are measuring with a stricter ruler on fewer questions.
2. **cmx's real edge (0% hallucination / refuse-to-guess) is now the field's frontier metric**
   (WMB-100K). The gamed "100%" claims have **no false-memory test at all**.
3. **Two concrete, high-value fixes to our benchmark (change measurement, not cmx):**
   - **(A) LLM-judge scorer.** Replace loose-token-match with a GPT-4o/4o-mini judge (the field
     standard). A semantically-correct answer phrased differently currently scores 0 for us; the
     judge would credit it. **This likely moves our measured accuracy up substantially** — possibly
     most of the gap to ~80%+ — with zero change to cmx. **Highest-value next experiment.**
   - **(B) Full / large sample.** Score the full LOCOMO QA set (1,540 across the 10 conversations),
     not 8–24, to kill the noise. Then per-cell deltas become trustworthy.
4. **Two concrete ACCURACY levers cmx is missing (from Dakera's winning recipe):**
   - **Temporal re-ranking** (down-weight stale memories by recency signals in the query) — the
     hardest LOCOMO category; cmx has none.
   - **Importance weighting** (rank critical decisions/config above incidental mentions).

## Recommended next steps (re-prioritised by this evidence)
1. **Build an LLM-judge scorer for the harness** and RE-SCORE the existing locked config — this is
   the apples-to-apples number vs Dakera/Mem0/others. (cmx unchanged; just honest measurement.)
2. Re-run the locked single-gate config on the **full LOCOMO set** with the LLM judge → the real,
   publishable cmx number (answerable accuracy + adversarial-refusal + hallucination + latency).
3. THEN resume forcing/accuracy work with temporal re-ranking + importance weighting as the levers
   (more promising than the prompt-forcing, which the ablation can de-prioritise).
4. Frame cmx's headline as the field is now framing it: **safety-first memory** (low/zero false
   memory) at competitive recall + low latency — exactly WMB-100K's scoring philosophy.

## Citations
- MemPalace debunk: vectorize.io/articles/mempalace-benchmarks; community audit (r/ML); MemPalace
  README correction (raw 96.6% LongMemEval, top_k=50 artefact on LOCOMO).
- Dakera methodology: dakera.ai/blog/dakera-benchmark-methodology (1,540 Q, GPT-4o judge, 88.2%).
- WMB-100K: github.com/Irina1920/WMB-100K (README v2.1) + dev.to/wontopos post — LOCOMO official F1
  GPT-4-full-context 32.1%; FM penalty −0.25; speed penalty; 100K-turn collapse.

---

## DEFINITIVE re-score result (the honest number) — 2026-06-13

Ran the locked config (maxed retrieval + single capable opus gate, gpt-5-mini foreground) across
**5 LOCOMO conversations, 75 answerable + 75 adversarial**, scoring with BOTH our token-match AND
the field-standard LLM judge (opus-4.7):

| metric | token-match | LLM-judge |
|---|---:|---:|
| answerable accuracy | **46.7%** | **45.3%** |
| adversarial refusal | 94.7% | — |
| hallucination | 7.7% | — |

Per-conversation accuracy swung **20% → 80%** (Dakera's "single conversations are noisy" warning,
confirmed hard).

### The two conclusions (both important, one humbling)
1. **Our measurement is HONEST.** The LLM judge (the exact method Dakera/others use to report 88%)
   agrees with our token-match — actually **1.3pt STRICTER**, not more lenient. So switching to the
   field-standard scorer does NOT close the gap. We are genuinely at ~46%, not secretly at ~88%.
2. **The gap to "88%" is real but is mostly APPLES-TO-ORANGES, not cmx being broken:**
   - **Different thing measured.** Dakera scores *memory RECALL* (did the store return the right
     memory) leniently. We score *end-to-end ANSWER correctness* through a small model under a
     strict refuse-to-guess gate.
   - **Different question mix.** We weight unanswerable/adversarial at 50% (15+15) and lean on the
     harder answerable categories; the 1,540-question self-reports are mostly answerable.
   - **The over-refusal trade is deliberate.** cmx refuses borderline-answerable questions to hold
     hallucination near zero — those score as "not correct" but are SAFE, not wrong. cmx optimises
     WMB-100K's frontier metric (low false-memory), not raw answer-rate.

### Where the real accuracy gain is (honest, evidence-backed)
- **NOT** a scoring change (confirmed: judge ≈ token-match).
- **NOT** prompt-forcing (the ablation showed it HURT these already-cooperative models).
- **YES** retrieval levers cmx still lacks — **temporal re-ranking** + **importance weighting**
  (Dakera's two винning levers; temporal is the hardest LOCOMO category for everyone).
- **YES** a measured relaxation of the sufficiency gate's over-refusal (accepting a small,
  bounded hallucination rise) — the iter-2/3 precision/recall trade, re-opened deliberately.

### cmx's externally-validated strength
~8% hallucination + ~95% adversarial refusal = the safety-first profile WMB-100K explicitly rewards
(−0.25 false-memory penalty) and the debunked "100%" systems don't even test.

---

## Retrieval-lever result (temporal + importance) — another honest negative

Built temporal re-ranking + importance weighting (Dakera's two levers), deterministic, flag-gated.
Pooled 3 LOCOMO conversations (45 answerable), locked config, LLM-judge:

| arm | acc (LLM-judge) | acc (token) | adv_refusal | halluc |
|---|---:|---:|---:|---:|
| baseline | 57.8% | 57.8% | 93.3% | 5.9% |
| temporal | 55.6% | 60.0% | 93.3% | 6.1% |
| importance | 55.6% | 62.2% | 93.3% | 9.1% |
| both | 57.8% | **66.7%** | 93.3% | 6.1% |

**The trap, caught:** by token-match the levers look like +8.9pt (57.8→66.7). By the LLM judge they
are FLAT (+0.0). The levers surface evidence containing more gold *tokens* but not more *correct*
answers — and importance alone raised hallucination (5.9→9.1) by promoting fact-dense
confabulation-bait. **No real, judge-verified accuracy gain on LOCOMO.** (Also vindicates building
the LLM judge — token-match alone would have falsely "shipped" this lever.)

Why they don't help here: LOCOMO's retrieval bottleneck isn't temporal/importance ORDERING (the
embeddings+rerank already surface the relevant turn). The residual misses are multi-hop synthesis,
genuine temporal *reasoning* (date arithmetic — the model's job, not retrieval), and the deliberate
over-refusal — none of which a re-rank prior fixes.

## Honest standing after the full sweep of levers
Across every lever tried this session — Council-judge, prompt-forcing (F1–F5), temporal &
importance re-ranking — **none produced a judge-verified accuracy gain**; forcing and Council
actively hurt. cmx sits at a genuine, well-measured **~55–58% answerable / ~6–9% hallucination /
~93% adversarial-refusal** on hard LOCOMO with a cheap model. The remaining real levers are
harder and carry explicit tradeoffs:
  1. **Over-refusal relaxation** (the sufficiency gate refuses borderline-answerable questions) —
     the one untested lever that directly targets answerable accuracy, but the finalists showed
     removing the gate entirely doubles hallucination (→44%); needs a *measured partial* relaxation.
  2. **Multi-hop answer synthesis** (combine 2-3 evidence turns) — real engineering, real risk.
  3. **A stronger foreground model** (opus instead of gpt-5-mini) — buys accuracy at cost/latency.
The cheap, safe wins are exhausted; the honest position is to keep cmx's locked, validated
safety-first config and choose the next lever deliberately with the user.

---

## Lever 2 — MULTI-HOP synthesis result (the FIRST real accuracy win)

Diagnostic (opus-4.8, 8k window) located the gap: single-hop 76.5% vs multi-hop 34.6% (mostly
safe refusals). Built multi-hop mode (gate allows derivation-by-combining-cited-slices + synthesis
instruction + inject_k 4→8; citation-check/verifier unchanged). Benchmark (3 LOCOMO convs, opus-4.8,
LLM judge):

| arm | single-hop | multi-hop | adv-refusal | halluc |
|---|---:|---:|---:|---:|
| baseline | 76.5% | 34.6% | 97.2% | 2.2% |
| **multihop** | **82.4%** | **46.2%** | **97.2%** | 9.3% |

**Δ: multi-hop +11.5pt · single +5.9pt · adversarial-refusal +0.0pt · hallucination +7.0pt.**

### Honest reading (more positive than the REJECT heuristic)
1. **Real, judge-verified accuracy gain** — multi-hop +11.5pt and single-hop +5.9pt. The FIRST
   lever this whole session that the LLM judge confirms (forcing, council, temporal, importance all
   failed). The diagnostic correctly predicted multi-hop was the gap.
2. **Adversarial safety PRESERVED** — adversarial-refusal unchanged at 97.2%. Multi-hop did NOT
   re-open the on-topic-but-unanswerable confabulation the strict gate was built to stop. The
   "must cite every sub-fact" rule held the adversarial line.
3. **The +7pt hallucination is ANSWERABLE-side** (since adversarial refusal didn't move). Two
   sources, mixed: (a) genuine confabulation — the model combining facts and stating a wrong derived
   value; (b) a MEASUREMENT ARTIFACT — the deterministic `_unsupported_by_store` check flags a
   CORRECT derived value (a duration, a comparison) that isn't verbatim in any single slice, even
   though the LLM judge scored the same answer correct. So the true confabulation cost is < 7pt.

### Verdict: a TUNABLE OPERATING POINT (not a free win, not a reject)
multi-hop mode buys ~+10pt answerable accuracy for ≤+7pt answerable-hallucination, with the
adversarial guardrail intact. It is the precision/recall trade, quantified and adversarial-safe:
- **safety-first deployments:** keep multihop OFF (2.2% halluc).
- **recall-first deployments:** turn multihop ON (≈65%+ pooled answerable, ≤9% halluc).

### Clear refinement (next, to cut the cost): multi-hop-aware verification
Replace the deterministic verbatim-token check with the LLM verifier FOR multi-hop answers only —
it can accept a correct DERIVATION (5 months from two dates) while still rejecting a wrong
inference. Expected: keep the +11.5pt accuracy, cut the +7pt hallucination toward baseline.

---

## Lever 2 — multi-hop: the fair-metric verdict (and a noise lesson)

Re-ran with a FAIR judge-based hallucination metric (counts shipped answers the judge rules WRONG,
not correct-but-non-verbatim derivations). Two runs of the SAME config:

| run | Δ multi-hop acc | Δ adv-refusal | Δ halluc |
|---|---:|---:|---:|
| v1 (strict halluc) | +11.5pt | +0.0pt | +7.0pt (strict) |
| v2 (fair halluc) | **+0.0pt** | +0.0pt | **+2.1pt** (judge) |

### Conclusions (honest)
1. **Multi-hop accuracy: NO reliable change.** The same config gave +11.5pt then +0.0pt — the
   signal is **dominated by noise** at ~26 multi-hop questions pooled (~8/seed). We cannot claim the
   lever lifts multi-hop. (Dakera's "small samples are statistically meaningless," demonstrated on
   our own work — exactly why they use 1,540 questions.)
2. **The v1 "+7pt hallucination" was a measurement artifact** — the fair judge metric shows only
   +2.1pt (noise). The strict verbatim-token check was penalizing CORRECT multi-hop derivations
   (durations, comparisons not verbatim in any slice). So multi-hop mode is **NOT unsafe**.
3. **Consistent across both runs:** single-hop +6pt (the synthesis prompt + higher inject_k modestly
   help even single-hop), and **adversarial-refusal unchanged at 97.2%** (the guardrail holds).
4. **Verdict: SAFE but UNPROVEN for its purpose.** Keep multihop flag-gated, default OFF. A reliable
   verdict needs the full multi-hop set (~400 questions) — a multi-hour eval, not 3 seeds.

## SESSION-WIDE honest conclusion (all levers)
Every accuracy lever attempted — Council-judge, prompt-forcing (F1–F5), temporal & importance
re-ranking, multi-hop synthesis — is **negative or lost in the noise**. What is ROBUST and
REPRODUCIBLE is the baseline:
- **single-hop ~76%, multi-hop ~31%, adversarial-refusal ~95–97%, real hallucination low**, honestly
  measured (LLM judge == token-match), **model-agnostic**, and **window-decoupled** (opus-4.8 @ 8k
  window answered over 600-turn conversations from the SQLite DB — the context-size-match problem is
  solved).
The cheap accuracy levers are exhausted. Multi-hop is the real remaining gap but is a hard problem
that needs a large-sample eval to tune or a fundamentally different mechanism — not a prompt/rerank
tweak. The defensible deliverable is the validated, safety-first locked config.
