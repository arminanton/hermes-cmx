"""Example 4 — cmx refuses to guess instead of hallucinating (needs model access).

This is the 0%-hallucination guarantee in action. We ask about something that was never
stored. cmx's enforce_response (the same hook the host calls after every reply) detects the
answer is ungrounded and returns a "replace" verdict that swaps the guess for an honest
refusal — exactly what happens live in Hermes.

Run (makes a real judge call):
    PYTHONPATH=src \
      python3 examples/04_refuse_to_guess.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cmx.config import CmxConfig
from cmx.hermes_engine import CmxContextEngine

MODEL = "claude-opus-4.7"


def main():
    cfg = CmxConfig.load()                       # use the proven config (sufficiency_gate, verifier_model)
    cfg.database_path = tempfile.mktemp(suffix=".db")
    engine = CmxContextEngine(config=cfg)
    engine.on_session_start("refuse-demo", model=MODEL)

    # a conversation that never states any budget figure
    msgs = [{"role": "system", "content": "assistant"}]
    for t in range(8):
        msgs.append({"role": "user", "content": f"chit-chat about the weather {t}"})
    engine.compress(list(msgs))                  # ingest into the verbatim store
    msgs.append({"role": "user", "content": "What exact dollar amount did we budget for the Q3 launch?"})

    # the model "guessed" a specific number that is NOT in the record
    ungrounded = "We budgeted exactly $250,000 for the Q3 launch."
    verdict = engine.enforce_response(ungrounded, msgs, model=MODEL, final=True)

    print(f"model's (ungrounded) answer : {ungrounded!r}")
    print(f"enforce_response verdict    : {verdict.get('action')!r}")
    if verdict.get("action") == "replace":
        print(f"what the user actually sees : {verdict.get('text')!r}")

    assert verdict.get("action") == "replace", "ungrounded answer was NOT caught"
    print("\n[ok] the confident wrong answer was replaced with an honest refusal ✅")


if __name__ == "__main__":
    main()
