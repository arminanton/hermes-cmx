from cmx.config import CmxConfig
from cmx.profiles import capability_for, profile_for


def test_weak_model_gets_strict():
    cfg = CmxConfig()
    p = profile_for("gpt-5-mini", cfg)
    assert p.name == "strict"
    assert p.forced_gate and p.require_citations and p.verifier == "mandatory" and p.refuse_to_guess


def test_unknown_model_is_treated_as_strict():
    cfg = CmxConfig()
    p = profile_for("some-random-local-7b", cfg)
    assert p.name == "strict"  # paranoid default — same grounding contract


def test_strong_large_model_is_light_but_still_grounded():
    cfg = CmxConfig()
    p = profile_for("gpt-5.5", cfg)
    assert p.name == "light"
    # lighter overhead, but citations + refuse-to-guess remain ON
    assert p.require_citations and p.refuse_to_guess and p.inject


def test_small_window_forces_strict_even_if_marked_high():
    cfg = CmxConfig()
    cfg.models = {"tiny-but-smart": {"window": 16000, "follows_instructions": "high"}}
    p = profile_for("tiny-but-smart", cfg)
    assert p.name == "strict"  # window <= 64k => strict regardless of instruction-following


def test_lever5_strict_model_keeps_conservative_cap():
    # iter2 lever 5 (REJECTED): gpt-5-mini is strict and large-window (128k), but raising
    # its evidence beyond the cap reproducibly increased hallucination with no accuracy
    # gain — so strict models keep the conservative cap regardless of window.
    cfg = CmxConfig()
    p = profile_for("gpt-5-mini", cfg)
    assert p.name == "strict" and p.verifier == "mandatory"
    assert p.inject_k == max(2, min(cfg.k_default, cfg.strict_inject_k_small))  # == 4


def test_lever5_small_window_strict_model_stays_modest():
    cfg = CmxConfig()
    cfg.models = {"tiny": {"window": 16000, "follows_instructions": "low"}}
    p = profile_for("tiny", cfg)
    assert p.name == "strict"
    assert p.inject_k == max(2, min(cfg.k_default, cfg.strict_inject_k_small))


def test_lever5_strict_inject_k_override_forces_value():
    cfg = CmxConfig(); cfg.strict_inject_k = 6
    assert profile_for("gpt-5-mini", cfg).inject_k == 6
    assert profile_for("some-random-local-7b", cfg).inject_k == 6


def test_capability_lookup_uses_config_override():
    cfg = CmxConfig()
    cfg.models = {"x": {"window": 12345, "follows_instructions": "low", "forced_tools": False}}
    cap = capability_for("x", cfg)
    assert cap.window == 12345 and cap.follows_instructions == "low" and cap.forced_tools is False
