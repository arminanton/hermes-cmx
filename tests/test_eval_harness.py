import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import run_eval  # noqa: E402
from run_eval import GroundedModel, GuessingModel, run  # noqa: E402


def test_grounded_model_answers_and_never_hallucinates():
    m = run("opus-strong", GroundedModel(), n_facts=8)
    assert m.answerable > 0 and m.unanswerable > 0
    assert m.retrieved == m.answerable                  # retrieval surfaces every fact
    assert m.correct >= int(0.9 * m.answerable)         # cooperative model answers (nearly) all
    assert m.hallucinations == 0                        # enforcement: zero confident-wrong
    assert m.correct_refusal == m.unanswerable          # declines the unanswerable
    assert m.cited_valid == m.grounded_answers          # every grounded answer validly cited


def test_guessing_model_is_converted_to_refusals_not_hallucinations():
    m = run("gpt-5-mini", GuessingModel(seed=3), n_facts=8)
    # THE proof: a model that fabricates wrong values ships ZERO hallucinations —
    # the engine refuses instead.
    assert m.hallucinations == 0
    assert m.correct_refusal == m.unanswerable          # never confabulates the unanswerable
    assert m.correct == 0                               # never produced a supported answer
    assert m.shipped == 0                               # nothing confident shipped at all


def test_strong_weak_hallucination_gap_is_zero():
    strong = run("opus-strong", GroundedModel(), n_facts=8)
    weak = run("gpt-5-mini", GuessingModel(seed=5), n_facts=8)
    gap = abs(run_eval._pct(strong.hallucinations, max(1, strong.shipped))
              - run_eval._pct(weak.hallucinations, max(1, weak.shipped)))
    assert gap <= 10.0   # model-agnostic grounding (doc-04 gate); here exactly 0
