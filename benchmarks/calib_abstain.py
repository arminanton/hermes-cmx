"""Iteration-3 L1 calibration: is retrieval-confidence separable for answerable vs
adversarial (unanswerable) LOCOMO questions?

Deterministic (embeddings only, NO foreground model). For each question we retrieve the
top-k evidence and compute the retrieval confidence = max cosine(query, evidence_slice).
If adversarial (category 5, answer=None) questions have systematically LOWER max-cosine
than answerable ones, a threshold τ can deterministically refuse the adversarial case
BEFORE the model can confabulate — the headline iter-3 lever.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from cmx.config import CmxConfig                # noqa: E402
from cmx.retrieval import HybridRetriever       # noqa: E402
from cmx.store import VerbatimStore             # noqa: E402
from cmx.embeddings import HermesEmbedder       # noqa: E402
import recall_diag as RD                        # noqa: E402


def _maxcos(emb, qtext, slices, vecs_by_id):
    qv = np.asarray(emb.embed([qtext])[0], "float32")
    qn = qv / (np.linalg.norm(qv) + 1e-8)
    best = -1.0
    for s in slices:
        v = vecs_by_id.get(s.id)
        if v is None:
            continue
        vn = v / (np.linalg.norm(v) + 1e-8)
        best = max(best, float(vn @ qn))
    return best


def main(k=8):
    qa = json.load(open(RD.DATASET))[0]["qa"]
    answerable = [q for q in qa if q.get("category") != 5 and q.get("answer") is not None][:40]
    adversarial = [q for q in qa if q.get("category") == 5][:40]

    st = VerbatimStore(tempfile.mktemp(suffix=".db"), use_trigram=True, use_graph=True)
    RD._ingest(st)
    emb = HermesEmbedder()
    cfg = CmxConfig(); cfg.use_embeddings = True; cfg.rerank = True
    r = HybridRetriever(st, cfg, embedder=emb)
    # ensure embeddings exist, then snapshot id->vector
    r._embed_search("warmup", 5, "loc")
    ids, mat = st.get_session_vectors("loc")
    vecs = {i: mat[j] for j, i in enumerate(ids)}

    def confidences(qs):
        out = []
        for q in qs:
            sl = r.retrieve(q["question"], k=k, session_id="loc")
            out.append(_maxcos(emb, q["question"], sl, vecs))
        return np.array(out)

    ca = confidences(answerable)
    cv = confidences(adversarial)
    pct = lambda a, p: float(np.percentile(a, p)) if len(a) else float("nan")
    print(f"answerable  (n={len(ca)}): mean={ca.mean():.3f} p10={pct(ca,10):.3f} p25={pct(ca,25):.3f} p50={pct(ca,50):.3f}")
    print(f"adversarial (n={len(cv)}): mean={cv.mean():.3f} p50={pct(cv,50):.3f} p75={pct(cv,75):.3f} p90={pct(cv,90):.3f}")

    # sweep τ: refuse if max-cos < τ. Want: keep answerable (above τ), drop adversarial (below τ).
    print("\nτ      answerable-kept   adversarial-refused")
    best = None
    for tau in np.arange(0.20, 0.61, 0.05):
        keep_ans = float((ca >= tau).mean())
        ref_adv = float((cv < tau).mean())
        # objective: maximize (answerable kept + adversarial refused)
        score = keep_ans + ref_adv
        print(f"{tau:.2f}   {keep_ans*100:5.1f}%            {ref_adv*100:5.1f}%   (sum={score:.2f})")
        if best is None or score > best[1]:
            best = (float(tau), score, keep_ans, ref_adv)
    print(f"\nbest τ={best[0]:.2f}: answerable-kept {best[2]*100:.1f}%, adversarial-refused {best[3]*100:.1f}%")


if __name__ == "__main__":
    main()
