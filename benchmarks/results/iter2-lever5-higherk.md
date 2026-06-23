# Iteration 2 — Lever 5: Higher injection-k for strict (weak) models — REJECTED

## Hypothesis
The strict profile caps `inject_k=4`. gpt-5-mini is *strict* (low-follows heuristic) yet
has a **128k window**, so the cap looked like evidence starvation conflating "small window"
with "weak instruction-following." Hypothesis: raise inject_k to `k_default` (8) for
large-window strict models → more evidence coverage → higher answer accuracy. recall_diag
supported the premise: full-stack recall **@4=75% < @8=80% < @12=85%**.

## A/B model gate (gpt-5-mini, full retrieval stack = embeddings + HyDE + rerank)
12 answerable + 6 adversarial, two independent runs:

| inject_k | answerable acc | adversarial refusal | hallucination |
|---|---|---|---|
| **4** (cap) run 1 | 58.3% | 100.0% | **0.0%** (0/10) |
| **4** (cap) run 2 | 58.3% |  83.3% | **9.1%** (1/11) |
| **8** (raised) run 1 | 50.0% | 83.3% | **18.2%** (2/11) |
| **8** (raised) run 2 | 58.3% | 66.7% | **16.7%** (2/12) |
| **4 combined** | ~58% | **~92%** | **~4.5%** (1/21 shipped) |
| **8 combined** | ~54% | ~75% | **~17%** (4/23 shipped) ❌ |

## Verdict: REJECT (keep the cap=4)
- Raising evidence for a weak model gave **no accuracy gain** and **reproducibly broke the
  hallucination guardrail** (~4.5% → ~17%) and dropped adversarial refusal (~92% → ~75%).
- Mechanism: a weak model turns extra retrieved slices into *false syntheses* — especially
  on adversarial (unanswerable) questions, where more "material" tempts a confabulated
  answer. This is exactly the risk the original `inject_k=4` cap was guarding against.
- **The cap is a feature, not a bug.** For strict/weak models, **precision** (the right
  evidence in the top-4, delivered by levers 1/3/4), not **recall** (more slices), is what
  lifts accuracy. recall@k is the wrong proxy for weak-model inject_k.

## The real win this lever surfaced
With the full retrieval stack (levers 1+3+4) and the cap kept at 4, **gpt-5-mini reaches
~58% answerable accuracy** on LOCOMO — up from v1's 12.5–25%. The v1 "over-refusal" was a
**retrieval-precision** problem (wrong/absent evidence in the top-k), now fixed — NOT an
inject-count problem. Confirms iter2's core finding: improve what lands in the top-k, don't
just inject more.

## Shipped change (minimal)
- Decoupled the knob in code: enforcement strictness stays tied to `follows_instructions`;
  evidence count is `_strict_inject_k()`. Default keeps the conservative cap for ALL strict
  models regardless of window. `cfg.strict_inject_k > 0` is an opt-in override for future
  well-behaved small models. No behavioral change vs v1 default (cap still 4).
- 71 unit tests green.
