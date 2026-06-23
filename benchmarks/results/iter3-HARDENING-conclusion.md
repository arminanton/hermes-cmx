# Iteration 3 — HARDENING conclusion (larger n): the residual is REAL; small-n 0% was noise

## L2-only (binary sufficiency gate, gpt-5.4 judge), mild contract, n = 22 answerable + 14 adversarial
| model | answerable acc | adversarial refusal | hallucination |
|---|---|---|---|
| opus-4.7 medium | 63.6% | 78.6% | **20.0%** |
| opus-4.6 fast (no-reasoning) | 72.7% | 85.7% | **18.2%** |
| gpt-5-mini low | 59.1% | 92.9% | **5.6%** |
| gpt-5-mini medium | 54.5% | 85.7% | 11.1% |
| sonnet-4.6 (no-reasoning) | 81.8% | 35.7% | **31.0%** |
| gemini-3.1-pro | 27.3% | 64.3% | 21.7% |

## The honest finding (this is why we hardened)
- The earlier **0% / 100% on opus & mini-low was small-sample NOISE** (n=8 adversarial). At
  n=14 adversarial, EVERY config hallucinates **5-31%** on LOCOMO category-5 questions.
- **L2 reduces but does NOT eliminate** adversarial confabulation: the sufficiency judge
  (gpt-5.4) still gets fooled by on-topic-but-insufficient evidence and passes a chunk of
  unanswerable questions, which the foreground model then answers.
- Model ranking is counter-intuitive and noisy: **gpt-5-mini-low is among the BEST (5.6%)**;
  **sonnet-4.6 with reasoning OFF is the WORST (31%)**. Signal: disabling reasoning hurts
  unanswerability detection; raw model "strength" does not predict grounding here.

## What this means
1. cmx is still a clear improvement over LCM (verbatim retrieval, no lossy-summary
   forgetting; head-to-head cmx 0% vs LCM 10% in the constrained regime) — but it is **not a
   0% hallucination guarantee on deliberately-adversarial, on-topic-but-unanswerable
   questions** (the hardest, rarest question type; normal answerable QA grounds well).
2. The on-topic-unanswerable case is a **hard, likely-fundamental limit** of retrieval +
   enforcement: similarity can't gate it (L1), a single sufficiency judge only partially
   gates it (L2), and a stronger judge doesn't help (judge ablation). 
3. Stacking more levers made it WORSE, not better (L4 feeds the gate misleading evidence;
   aggressive citation contracts push cite-and-answer). Less is more.

## Revised recommendation
- SHIP the conservative, validated core: iter-1 enforcement + iter-2 retrieval (embeddings/
  HyDE/cosine-rerank) + **L2 sufficiency gate** (it nets-reduces adversarial confabulation).
- Treat hallucination as **mitigated, not solved**. For critical work: enable reasoning,
  prefer answerable-only flows, and keep refuse-to-guess strict.
- Do NOT claim 0%. Next research (iter-4): an ensemble/multi-judge unanswerability vote, or a
  fundamentally different abstention signal — the single-judge sufficiency gate has a ceiling.
