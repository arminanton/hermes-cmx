# Iteration 2 — Lever 6: Semantic-support verifier (+ the lever-2 retest)

## What shipped
An independent **strict NLI judge** rescues a CITED sentence whose fact tokens are not
verbatim in its cited evidence (paraphrase/synonym) — reducing FALSE refusals on
correctly-grounded free-form answers — while:
- **never** softening a phantom citation (cited id does not exist),
- defaulting to UNSUPPORTED on judge outage/absence (safe),
- being prompted to flag substitution/contradiction ("chose Postgres" vs evidence
  "chose MySQL" → UNSUPPORTED), the exact failure mode a bare embedding-cosine check
  would miss.
Config: `cmx.semantic_verifier` (default **off**), optional `semantic_prefilter_sim`
embedding pre-filter. Gated; LCM/default behavior unchanged.

## A/B gate — gpt-5-mini, full retrieval stack (embeddings + HyDE + rerank)

### Semantic verifier alone (no tools)
| semantic_verifier | answerable acc | adversarial refusal | hallucination |
|---|---|---|---|
| OFF | 58.3% | 66.7% | 16.7% |
| ON  | 58.3–66.7% | 83.3–100% | **0–9%** |

- **No LOCOMO answerable-accuracy lift** from the rescue path: LOCOMO answers are mostly
  verbatim-extractable (names/dates/values), so correct answers already pass the
  deterministic check; the paraphrase-rescue rarely decides. Its accuracy value is in
  real, paraphrase-heavy dialogue, not this benchmark.
- **Safe**: keeps hallucination ≈0% and adversarial refusal high. Provides the Layer-4
  judge for uncited claims (defense in depth). **Verdict: CONDITIONAL KEEP (default-off).**

### The decisive test — does lever 6 make lever 2 (tools) safe?
| full stack + verifier | answerable acc | adversarial refusal | hallucination |
|---|---|---|---|
| **tools OFF** | **66.7%** | **100.0%** ✅ | **0.0%** ✅ |
| tools ON | 50.0% | **0.0%** ❌ | **38.9%** ❌ |

**Lever 2 = REJECT, even on top of lever 6.** With a tool loop + forced retrieval the
model fishes up *some* tangentially-related slice for every unanswerable question, cites
it, and BOTH the citation check and the semantic verifier rubber-stamp it — because they
verify "evidence supports claim," never "this is the RIGHT evidence for the question."
Adversarial refusal collapses 100%→0%, hallucination 0%→38.9%. This reproduces and
deepens the original lever-2 finding: the semantic verifier cannot recover it.

## The core principle this iteration proves
**Engine-driven proactive injection + refuse-to-guess is safe; model-driven retrieval
(tools) is not.** Let the engine decide what verbatim evidence to inject and force the
model to ground on it or refuse. Giving the model a retrieval tool loop hands it the rope
to confabulate, and no downstream check can distinguish a well-cited wrong answer from a
correct one.

## Tests
77 unit tests green (incl. 6 new lever-6 tests: paraphrase rescued, substitution caught,
phantom never rescued, outage-safe).
