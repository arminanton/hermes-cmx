"""Example 3 — a real model grounds its answer on retrieved verbatim evidence (needs model access).

End-to-end inside the Hermes process: cmx injects the relevant verbatim slice at compaction
(Layer 1), and a real model answers the question *citing* that evidence — instead of guessing
from a lossy summary. Mirrors benchmarks/live_inhost_test.py.

Run (uses the Hermes auxiliary client, so it makes one real model call):
    PYTHONPATH=src \
      python3 examples/03_grounded_answer.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cmx.config import CmxConfig
from cmx.hermes_engine import CmxContextEngine

MODEL = "claude-opus-4.7"


def main():
    cfg = CmxConfig()
    cfg.database_path = tempfile.mktemp(suffix=".db")
    cfg.models = {MODEL: {"window": 200000}}
    engine = CmxContextEngine(config=cfg)
    engine.on_session_start("grounding-demo", model=MODEL)

    msgs = [{"role": "system", "content": "You are a helpful engineering assistant."}]
    for t in range(30):
        msgs.append({"role": "user", "content": f"misc note {t} about unrelated things"})
    msgs.append({"role": "assistant", "content": "Decision: the production deploy region is eu-west-3."})
    for t in range(30, 45):
        msgs.append({"role": "user", "content": f"misc note {t} about unrelated things"})
    msgs.append({"role": "user", "content": "remind me which production deploy region we decided on?"})

    assembled = engine.compress(msgs)          # cmx injects verbatim evidence (not a summary)
    sys_block = assembled[0]["content"]
    assert "CMX EVIDENCE" in sys_block and "eu-west-3" in sys_block
    print("[ok] cmx injected the verbatim evidence into the prompt\n")

    contract = {"role": "system", "content":
                "Answer ONLY from the [CMX EVIDENCE] shown, and cite [id=N]. "
                "If absent, say: I don't have that in my record. One short sentence."}
    from agent.auxiliary_client import call_llm
    r = call_llm(task="compression", model=MODEL,
                 messages=assembled + [contract], max_tokens=60, timeout=60)
    answer = r.choices[0].message.content
    print(f"model answer: {answer!r}")
    assert "eu-west-3" in answer, "model failed to ground on the injected evidence"
    print("\n[ok] the model answered from real history, with a citation ✅")


if __name__ == "__main__":
    main()
