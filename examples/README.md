# cmx examples

Four small, runnable programs that demonstrate what cmx does. The first two need **no model
access** and run instantly; the last two make a real model call.

> Run under the Hermes interpreter (it has `pysqlite3` with the trigram tokenizer):
> ```bash
> cd hermes-cmx
> PYTHONPATH=src python3 examples/01_store_and_retrieve.py
> ```
> The two model-driven examples also need the Hermes package on the path:
> ```bash
> PYTHONPATH=src python3 examples/03_grounded_answer.py
> ```

| # | File | Needs a model? | Shows |
|---|---|---|---|
| 1 | `01_store_and_retrieve.py` | no | every message stored **verbatim**; the exact fact recovered from 80 turns of noise (no lossy summary) |
| 2 | `02_survives_session_rotation.py` | no | memory **survives Hermes session-id rotation** (compaction/checkpoints) via lineage normalization — the bug that made this robust |
| 3 | `03_grounded_answer.py` | yes | cmx injects the verbatim slice and a **real model answers with a citation** |
| 4 | `04_refuse_to_guess.py` | yes | an ungrounded "guess" is **replaced with an honest refusal** (the 0%-hallucination mechanism, exactly as the live host applies it) |

Each example is self-contained, uses a throwaway temp database, and **never touches your live
`~/.hermes/cmx.db` or config**. They print a clear `[ok] … ✅` on success.

### What to read next
- `../README.md` — the project overview and the cmx-vs-LCM comparison.
- `../benchmarks/README.md` — every accuracy/safety lever we tried, with verdicts and numbers.
- `../docs/deliverable/cmx-DELIVERABLE.md` — the proven config + how it's installed on Hermes.
