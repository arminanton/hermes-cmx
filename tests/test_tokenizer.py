from cmx.tokenizer import get_tokenizer, provider_of, window_for


def test_provider_detection():
    assert provider_of("claude-opus-4.7") == "anthropic"
    assert provider_of("gpt-5-mini") == "openai"
    assert provider_of("gemini-3.1-pro-preview") == "google"
    assert provider_of("some-local-llama") == "_default"


def test_counts_are_positive_and_scale():
    tk = get_tokenizer("gpt-5-mini")
    assert tk.count("") == 0
    short = tk.count("hello world")
    long = tk.count("hello world " * 50)
    assert 0 < short < long


def test_provider_specific_counts_differ():
    # Different providers use different calibration → not a single cl100k estimate.
    text = "def migrate(): return CI4_migrate_v2(payload)"
    a = get_tokenizer("claude-opus-4.7").count(text)
    o = get_tokenizer("gpt-5-mini").count(text)
    g = get_tokenizer("gemini-3.1-pro-preview").count(text)
    assert len({a, o, g}) >= 2  # at least two providers disagree


def test_message_overhead():
    tk = get_tokenizer("gpt-5-mini")
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    assert tk.count_messages(msgs) >= tk.count("hi") + tk.count("yo")


def test_window_lookup():
    assert window_for("gpt-5.5") == 400_000
    assert window_for("gemini-3.1-pro-preview") == 1_000_000
    assert window_for("totally-unknown-model") == 32_000  # paranoid default
    assert window_for("custom", {"custom": {"window": 12345}}) == 12345
