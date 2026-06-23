from cmx.expand import ChainExpander, keyword_expander


def test_keyword_expander_strips_question_words():
    out = keyword_expander("When did Caroline go to the support group?")
    assert out and "caroline" in out[0].lower() and "support" in out[0].lower()
    assert "when" not in out[0].lower() and "did" not in out[0].lower()


def test_keyword_expander_noop_when_all_stopwords():
    assert keyword_expander("what did we do?") == []


def test_chain_expander_merges():
    e = ChainExpander(lambda q: ["a"], lambda q: ["b"])
    assert e("x") == ["a", "b"]


def test_chain_expander_tolerates_failure():
    def boom(q):
        raise RuntimeError("x")
    e = ChainExpander(boom, lambda q: ["ok"])
    assert e("x") == ["ok"]
