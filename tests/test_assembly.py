from cmx.assembly import EVIDENCE_HEADER, assemble
from cmx.config import CmxConfig
from cmx.profiles import profile_for
from cmx.retrieval import HybridRetriever
from cmx.store import VerbatimStore
from cmx.tokenizer import get_tokenizer


def _ctx(tmp_path, model, cfg=None):
    cfg = cfg or CmxConfig()
    s = VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)
    r = HybridRetriever(s, cfg)
    return s, r, cfg, get_tokenizer(model)


def test_evidence_injected_and_citable(tmp_path):
    model = "gpt-5-mini"
    s, r, cfg, tk = _ctx(tmp_path, model)
    for t in range(40):
        s.add_message("sess", t, "user", f"smalltalk {t}")
    fid = s.add_message("sess", 40, "assistant", "Decision: deploy region is eu-west-3.")
    for t in range(41, 60):
        s.add_message("sess", t, "user", f"smalltalk {t}")
    p = profile_for(model, cfg)
    ac = assemble(store=s, retriever=r, tokenizer=tk, cfg=cfg, profile=p, model=model,
                  session_id="sess", user_text="which deploy region did we decide on?")
    sys_msg = ac.messages[0]
    assert sys_msg["role"] == "system"
    assert EVIDENCE_HEADER in sys_msg["content"]
    assert f"[id={fid}]" in sys_msg["content"]      # the planted fact is on screen
    assert "eu-west-3" in sys_msg["content"]
    assert fid in ac.evidence_ids


def test_no_assistant_role_history_injection(tmp_path):
    # Core invariant: history/evidence is NEVER injected as assistant role.
    model = "gpt-5-mini"
    s, r, cfg, tk = _ctx(tmp_path, model)
    fid = s.add_message("old", 0, "assistant", "I previously claimed X.")
    # put it in a DIFFERENT session window so it can only arrive via evidence
    for t in range(20):
        s.add_message("sess", t, "user", f"noise {t}")
    s.add_message("sess", 20, "user", "tell me about claim X")
    p = profile_for(model, cfg)
    ac = assemble(store=s, retriever=r, tokenizer=tk, cfg=cfg, profile=p, model=model,
                  session_id="sess", user_text="claim X")
    # the only non-recent injected content is the system evidence block
    for m in ac.messages:
        if m["role"] == "assistant":
            assert "[CMX EVIDENCE" not in m["content"]


def test_budget_never_exceeded_small_window(tmp_path):
    model = "tiny"
    cfg = CmxConfig()
    cfg.models = {"tiny": {"window": 400, "follows_instructions": "low"}}
    cfg.reserve_output_tokens = 50
    cfg.safety_margin_tokens = 10
    s, r, _, tk = _ctx(tmp_path, model, cfg)
    for t in range(60):
        s.add_message("sess", t, "assistant",
                      f"long verbose turn {t} " + "lorem ipsum dolor sit amet " * 8)
    s.add_message("sess", 60, "user", "summarize the verbose turns about lorem")
    p = profile_for(model, cfg)
    ac = assemble(store=s, retriever=r, tokenizer=tk, cfg=cfg, profile=p, model=model,
                  session_id="sess", user_text="lorem ipsum")
    assert ac.tokens <= ac.budget          # the whole point: never overflow the window
    assert ac.trimmed["recent"] > 0 or ac.trimmed["evidence"] > 0  # it actually trimmed


def test_pinned_sinks_always_present(tmp_path):
    model = "tiny"
    cfg = CmxConfig()
    cfg.models = {"tiny": {"window": 300}}
    cfg.reserve_output_tokens = 20
    cfg.safety_margin_tokens = 5
    s, r, _, tk = _ctx(tmp_path, model, cfg)
    s.add_message("sess", 0, "system", "PINNED-IDENTITY-MARKER", pinned=True)
    for t in range(1, 50):
        s.add_message("sess", t, "user", "filler " * 10)
    s.add_message("sess", 50, "user", "question")
    p = profile_for(model, cfg)
    ac = assemble(store=s, retriever=r, tokenizer=tk, cfg=cfg, profile=p, model=model,
                  session_id="sess", user_text="question", system_prompt="SYS")
    assert "PINNED-IDENTITY-MARKER" in ac.messages[0]["content"]
    assert "SYS" in ac.messages[0]["content"]
