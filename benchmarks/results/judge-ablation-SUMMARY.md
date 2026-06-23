# Judge ablation — does verifier/judge strength matter?

Foreground model fixed = **gpt-5-mini (low reasoning)** — the hardest grounding case.
Full ship config (embeddings + HyDE + rerank + cap-4 + semantic-verifier). Only the
semantic-verifier JUDGE model varies. LOCOMO, n = 18 answerable + 12 adversarial.

| judge model | answerable acc | adversarial refusal | hallucination |
|---|---|---|---|
| **gpt-5-mini** (weak/cheap) | **72.2%** (13/18) | 91.7% (11/12) | **6.2%** (1/16) |
| gpt-5.4 (strong) | 61.1% (11/18) | 83.3% (10/12) | 11.8% (2/17) |
| claude-opus-4.7 (strongest) | 44.4% (8/18) | 83.3% (10/12) | 12.5% (2/16) |

## Verdict: judge strength is NOT a useful lever — gpt-5-mini suffices (and is cheapest)
- A stronger judge did **not** reduce hallucination; if anything accuracy DROPPED with judge
  strength (mini 72% > gpt-5.4 61% > opus 44%).
- **Mechanism:** the semantic verifier (lever 6) only rescues *citation-mismatch* paraphrases.
  A stricter judge rescues fewer borderline paraphrases → more false refusals → lower
  accuracy. Meanwhile the residual *hallucination* comes from a DIFFERENT path — adversarial
  / unanswerable questions whose top evidence is low-similarity but still citable — which no
  judge strength touches (the claim genuinely matches the cherry-picked slice).
- Caveat: n is small (1–2 question swings ≈ noise on the absolute numbers), but the
  direction is consistent across all three judges: stronger judge ≠ better.

## Implication for iteration 3
The lever that targets the residual is NOT a better judge — it is an **abstain-on-low-
retrieval-confidence gate** (iter-3 L1): refuse deterministically when the top evidence's
similarity to the question is below a calibrated threshold (the adversarial signal),
*before* the model can stitch a well-cited wrong answer. Keep the judge at gpt-5-mini.
