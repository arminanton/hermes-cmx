"""Deterministic retrieval-recall diagnostic (iteration-2 primary metric).

Measures evidence-recall@k on LOCOMO answerable questions for different retrieval
configs — NO model calls, so it's cheap, fast, and noise-free. This is the primary
signal for the retrieval levers (1 embeddings, 3 query-expansion, 4 rerank, 5 chunk+k).
Reports lexical-only, embedding-only, and fused, at multiple k.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cmx.config import CmxConfig
from cmx.retrieval import HybridRetriever
from cmx.store import VerbatimStore

DATASET = str(Path(__file__).resolve().parent / "locomo10.json")


def _ingest(store):
    conv = json.load(open(DATASET))[0]["conversation"]
    dia = {}
    for i in range(1, 40):
        sk, dk = f"session_{i}", f"session_{i}_date_time"
        if sk not in conv:
            continue
        date = conv.get(dk, "")
        for turn in conv[sk]:
            mid = store.add_message("loc", 0, "user",
                                    f"[{date}] {turn['speaker']}: {turn['text']} ({turn['dia_id']})")
            dia[turn["dia_id"]] = mid
    return dia


def _questions(n):
    qa = json.load(open(DATASET))[0]["qa"]
    return [q for q in qa if q.get("category") != 5 and q.get("answer") is not None
            and q.get("evidence")][:n]


def recall_at(r, questions, dia, k, source_filter=None):
    hit = 0
    for q in questions:
        sl = r.retrieve(q["question"], k=k, session_id="loc")
        if source_filter:
            ids = {s.id for s in sl if source_filter in s.sources}
        else:
            ids = {s.id for s in sl}
        gold = {dia.get(e) for e in q["evidence"] if dia.get(e)}
        if ids & gold:
            hit += 1
    return hit, len(questions)


def main(n=40, ks=(5, 8, 15)):
    from cmx.embeddings import HermesEmbedder
    qs = _questions(n)
    print(f"LOCOMO evidence-recall on {len(qs)} answerable questions\n")
    # lexical (no embeddings)
    cfg = CmxConfig(); cfg.use_embeddings = False
    st = VerbatimStore(tempfile.mktemp(suffix=".db"), use_trigram=True, use_graph=True)
    dia = _ingest(st)
    rlex = HybridRetriever(st, cfg)
    # embeddings build
    cfg2 = CmxConfig(); cfg2.use_embeddings = True
    st2 = VerbatimStore(tempfile.mktemp(suffix=".db"), use_trigram=True, use_graph=True)
    dia2 = _ingest(st2)
    remb = HybridRetriever(st2, cfg2, embedder=HermesEmbedder())
    for k in ks:
        hl, n1 = recall_at(rlex, qs, dia, k)
        hf, _ = recall_at(remb, qs, dia2, k)
        he, _ = recall_at(remb, qs, dia2, k, source_filter="emb")
        print(f"k={k:2}:  lexical {100*hl//n1:3d}% ({hl}/{n1})   "
              f"fused(+emb) {100*hf//n1:3d}% ({hf}/{n1})   "
              f"emb-only {100*he//n1:3d}% ({he}/{n1})")


if __name__ == "__main__":
    main()
