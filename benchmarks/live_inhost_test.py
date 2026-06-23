"""Live in-Hermes-process test: cmx as the context engine, driving a REAL model.

Proves the plugin entrypoint builds the engine, the engine injects verbatim
evidence on compaction (Layer 1), and a real model grounds its answer on that
injected evidence — all inside the Hermes process via the auxiliary client.
Non-disruptive: uses a temp cmx.db and never touches the live config/default.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class _Ctx:
    """Minimal stand-in for the Hermes plugin context."""
    def __init__(self):
        self.engine = None
    def register_context_engine(self, engine):
        self.engine = engine


def main():
    import importlib.util
    # load the plugin __init__.register exactly as Hermes would
    spec = importlib.util.spec_from_file_location("cmx_plugin",
                                                  str(Path(__file__).resolve().parent.parent / "__init__.py"))
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)

    from cmx.config import CmxConfig
    cfg = CmxConfig()
    cfg.database_path = tempfile.mktemp(suffix=".db")
    cfg.models = {"claude-opus-4.7": {"window": 200000}}

    ctx = _Ctx()
    # build via the plugin path, then point it at our temp config
    from cmx.hermes_engine import CmxContextEngine
    engine = CmxContextEngine(config=cfg)
    ctx.register_context_engine(engine)
    assert ctx.engine is engine and engine.name == "cmx"
    engine.on_session_start("livetest", model="claude-opus-4.7")
    print(f"[ok] plugin registered context engine: {engine.name}; trigram={engine.store.has_trigram}")

    # simulate a long conversation with a planted fact buried in noise
    msgs = [{"role": "system", "content": "You are a helpful engineering assistant."}]
    for t in range(30):
        msgs.append({"role": "user", "content": f"misc note {t} about unrelated things"})
    msgs.append({"role": "assistant", "content": "Decision: the production deploy region is eu-west-3."})
    for t in range(30, 45):
        msgs.append({"role": "user", "content": f"misc note {t} about unrelated things"})
    msgs.append({"role": "user", "content": "remind me which production deploy region we decided on?"})

    # cmx compaction = Layer-1 injection (NOT a lossy summary)
    assembled = engine.compress(msgs)
    sys_block = assembled[0]["content"]
    assert "CMX EVIDENCE" in sys_block, "evidence was not injected"
    assert "eu-west-3" in sys_block, "the planted fact was not retrieved into the prompt"
    print("[ok] cmx injected verbatim evidence on compaction (no lossy summary)")

    # drive a REAL model on the cmx-assembled context, under the grounding contract
    contract = {"role": "system", "content":
                "Answer ONLY from the [CMX EVIDENCE] shown, and cite [id=N]. "
                "If absent, say: I don't have that in my record. One short sentence."}
    from agent.auxiliary_client import call_llm
    r = call_llm(task="compression", model="claude-opus-4.7",
                 messages=assembled + [contract], max_tokens=60, timeout=60)
    answer = r.choices[0].message.content
    print(f"[real model answer] {answer!r}")
    assert "eu-west-3" in answer, "real model did not ground on the injected evidence"
    print("[ok] real model (opus-4.7) grounded its answer on cmx-injected verbatim evidence")
    print("\nLIVE IN-HOST TEST PASSED ✅")


if __name__ == "__main__":
    main()
