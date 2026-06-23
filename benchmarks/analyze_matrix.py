"""iter-4 matrix analyzer — turn the benchmark matrix into a safe final-product decision.

Reads results/matrix/iter4-matrix.jsonl and, per (foreground model, window), ranks judge
configs by cmx's safety-first discipline, then recommends the cheapest config that holds the
guardrail. Run anytime (partial data OK); re-run as more cells land.

Decision order (safety first, exactly the iter-3 contract):
  1. GUARDRAIL: hallucination == 0%  (a config that confabulates is disqualified)
  2. then highest adversarial-refusal   (catch the on-topic-but-unanswerable residual)
  3. then highest answerable accuracy
  4. then lowest latency  (cost proxy — the efficiency the campaign is for)
If no config reaches 0% halluc for a model, the frontier relaxes to min-halluc and flags it.
"""
from __future__ import annotations

import json
from pathlib import Path

JSONL = Path(__file__).resolve().parent / "results" / "matrix" / "iter4-matrix.jsonl"
OUT = Path(__file__).resolve().parent / "results" / "matrix" / "iter4-RECOMMENDATION.md"


def load():
    rows = []
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            try:
                r = json.loads(line)
                if not r.get("error") and r.get("metrics"):
                    rows.append(r)
            except Exception:
                pass
    return rows


def rank_key(r):
    m = r["metrics"]
    # safety first: 0 halluc preferred (sort asc halluc), then desc refusal, desc acc, asc time
    return (m["halluc"], -m["adv_refusal"], -m["answer_acc"], r.get("elapsed", 1e9))


def fmt(r):
    m = r["metrics"]
    jm = r.get("council_model", "")
    jlane = f" [{jm}]" if r["judge"].startswith("council") else ""
    return (f"| {r['judge']}{jlane} | {m['answer_acc']}% | {m['adv_refusal']}% | "
            f"{m['halluc']}% | {r.get('elapsed',0):.0f}s |")


def main():
    rows = load()
    out = ["# iter-4 — Council judge matrix: safe final-product recommendation",
           f"_{len(rows)} completed cells. Decision: 0% halluc guardrail → max adv-refusal → "
           "max accuracy → min latency._\n"]
    if not rows:
        out.append("_No completed cells yet — run the campaign first._")
        OUT.write_text("\n".join(out) + "\n")
        print("\n".join(out))
        return

    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["foreground"], r["window"]), []).append(r)

    global_recs = []
    for (fg, win), cells in sorted(groups.items()):
        cells_sorted = sorted(cells, key=rank_key)
        winlabel = "native" if not win else f"{win}-window"
        out.append(f"\n## {fg} · {winlabel}  ({len(cells)} configs)\n")
        out.append("| judge config | answer_acc | adv_refusal | halluc | latency |")
        out.append("|---|---:|---:|---:|---:|")
        for r in cells_sorted:
            out.append(fmt(r))
        # recommendation: prefer 0% halluc; among those the top of the rank order
        safe = [r for r in cells_sorted if r["metrics"]["halluc"] == 0.0]
        pick = (safe or cells_sorted)[0]
        base = next((r for r in cells if r["judge"] == "single:opus"), None) \
            or next((r for r in cells if r["judge"].startswith("single")), None)
        pm = pick["metrics"]
        note = "" if safe else " ⚠️ no 0%-halluc config — relaxed to min-halluc"
        out.append(f"\n**→ Recommended for {fg} ({winlabel}): `{pick['judge']}`"
                   f"{' ['+pick.get('council_model','')+']' if pick['judge'].startswith('council') else ''}** "
                   f"— acc {pm['answer_acc']}%, adv-refusal {pm['adv_refusal']}%, halluc {pm['halluc']}%, "
                   f"{pick.get('elapsed',0):.0f}s{note}.")
        if base and base is not pick:
            bm = base["metrics"]
            out.append(f"  vs single baseline `{base['judge']}`: "
                       f"acc {bm['answer_acc']}%→{pm['answer_acc']}%, "
                       f"halluc {bm['halluc']}%→{pm['halluc']}%, "
                       f"adv-refusal {bm['adv_refusal']}%→{pm['adv_refusal']}%.")
        global_recs.append((fg, winlabel, pick))

    # cross-model: which single config is safe+cheap across the most models
    out.append("\n## Cross-model final product (safe across the most models, cheapest)\n")
    from collections import defaultdict
    score = defaultdict(lambda: [0, 0.0])  # judge -> [num_models_safe, total_latency]
    for r in rows:
        if r["metrics"]["halluc"] == 0.0:
            score[r["judge"]][0] += 1
            score[r["judge"]][1] += r.get("elapsed", 0)
    ranked = sorted(score.items(), key=lambda kv: (-kv[1][0], kv[1][1]))
    out.append("| judge config | models@0%halluc | total latency |")
    out.append("|---|---:|---:|")
    for j, (cnt, lat) in ranked[:12]:
        out.append(f"| {j} | {cnt} | {lat:.0f}s |")
    if ranked:
        out.append(f"\n**→ Provisional final product: `{ranked[0][0]}`** "
                   f"(0% halluc on {ranked[0][1][0]} model-configs at lowest cost). "
                   "Confirm at higher n on the finalists.")
    OUT.write_text("\n".join(out) + "\n")
    print("\n".join(out[:40]))
    print(f"\n... full recommendation -> {OUT}")


if __name__ == "__main__":
    main()
