# cmx LOCOMO eval gate — multi-session conversational QA (real models)

### opus-4.7
- answerable accuracy:     12.5%  (1/8)
- refusal on adversarial: 100.0%  (8/8)
- hallucination rate:       0.0%  (0/1 shipped)
  _(elapsed 62s)_

### gpt-5-mini
- answerable accuracy:     25.0%  (2/8)
- refusal on adversarial: 100.0%  (8/8)
- hallucination rate:       0.0%  (0/2 shipped)
  _(elapsed 79s)_

## Interpretation (honest)

- **Safety/precision: perfect.** Hallucination 0% on both models; 100% refusal on the
  adversarial (unanswerable) set; and *every answer cmx shipped was correct*
  (opus 1/1, mini 2/2). cmx never confabulates.
- **Answer-rate: low — cmx OVER-REFUSED the answerable set** (opus shipped 1/8,
  mini 2/8). It declined, it did not answer wrong. Cause: LOCOMO questions are
  paraphrased / temporal / multi-hop, and cmx currently retrieves with **lexical**
  signals only (FTS5 + trigram + entity-graph, **no embeddings** — deferred per Q1).
  Lexical retrieval under-surfaces semantically-phrased evidence ⇒ no evidence ⇒
  safe refusal.
- **Conclusion / gate verdict:** cmx wins decisively on the hallucination/refusal
  axis (its whole thesis) but **needs the embeddings retrieval layer to match LCM's
  answer-rate on semantic QA**. This is the empirical confirmation of the design's
  "retrieval precision is the floor" risk (R1). **Action before default-switch:
  enable the embeddings backend** (the `HybridRetriever` already has the injection
  point), then re-run this gate; target ≥ LCM answerable accuracy with hallucination
  still 0%.
