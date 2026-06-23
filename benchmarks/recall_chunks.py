"""Iteration-3 L3 test: do CHUNK embeddings beat whole-turn embeddings when a fact is
buried in a long turn? (Real embeddings.) Builds a corpus of long filler turns plus
target turns where one distinctive fact sentence is buried mid-paragraph, then measures
recall@k of the target turn with use_chunks OFF vs ON.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cmx.config import CmxConfig            # noqa: E402
from cmx.retrieval import HybridRetriever   # noqa: E402
from cmx.store import VerbatimStore         # noqa: E402
from cmx.embeddings import HermesEmbedder   # noqa: E402

FILLER = (
    "The team discussed scheduling and logistics for the upcoming offsite. "
    "Lunch options were debated at length and nobody could agree on catering. "
    "Several people shared weekend plans and weather observations. "
    "There was a tangent about commute times and parking availability. "
    "Someone mentioned a documentary they watched and recommended it. "
    "The group reviewed generic status updates with no concrete decisions. "
    "A long aside covered office plants, coffee machines, and desk ergonomics. ")

FACTS = [
    ("the internal codename for the Q3 data migration was Bluefin-7",
     "what was the internal codename for the Q3 data migration?"),
    ("the production database failover region was set to ap-southeast-2",
     "which region was the production database failover set to?"),
    ("the agreed API rate limit for partners was 4200 requests per minute",
     "what API rate limit did we agree on for partners?"),
    ("the security audit was scheduled with the vendor Krollmark on 14 March",
     "which vendor was the security audit scheduled with?"),
    ("the rollback budget cap was fixed at 18000 dollars per incident",
     "what was the rollback budget cap per incident?"),
    ("the feature flag for the new checkout was named chk_v3_canary",
     "what was the feature flag name for the new checkout?"),
    ("the on-call rotation handoff time moved to 9:45 am every Tuesday",
     "what time did the on-call handoff move to?"),
    ("the data retention window for raw logs was shortened to 47 days",
     "how many days is the raw log retention window?"),
    ("the preferred embedding model for the search prototype was minilm-x2",
     "which embedding model was preferred for the search prototype?"),
    ("the contract renewal deadline with the cloud vendor was 30 November",
     "what is the contract renewal deadline with the cloud vendor?"),
]


def _buried(fact: str) -> str:
    # bury the single fact sentence in a very long turn (~12x filler) so the whole-turn
    # average is dominated by filler — the regime where chunking should matter.
    pad = " ".join([FILLER] * 6)
    return pad + " " + fact.capitalize() + ". " + pad


def main(k_list=(3, 5, 8)):
    st = VerbatimStore(tempfile.mktemp(suffix=".db"), use_trigram=True, use_graph=False)
    # 60 long pure-filler turns (distractors), each as long as the targets
    for i in range(60):
        st.add_message("s", i, "user", " ".join([FILLER] * 12) + f" (note {i})")
    target = {}
    for j, (fact, _q) in enumerate(FACTS):
        mid = st.add_message("s", 100 + j, "user", _buried(fact))
        target[j] = mid
    emb = HermesEmbedder()

    def recall(use_chunks):
        cfg = CmxConfig(); cfg.use_embeddings = True; cfg.rerank = False
        cfg.use_chunks = use_chunks; cfg.chunk_min_chars = 400; cfg.chunk_size_chars = 220
        r = HybridRetriever(st, cfg, embedder=emb)
        res = {}
        for k in k_list:
            hit = 0
            for j, (_fact, q) in enumerate(FACTS):
                # isolate the embedding effect (FTS keyword-match would mask dilution):
                # whole-turn embedding kNN vs chunk-embedding kNN, directly.
                ids = (r._chunk_embed_search(q, k, "s") if use_chunks
                       else r._embed_search(q, k, "s"))
                if target[j] in ids:
                    hit += 1
            res[k] = (hit, len(FACTS))
        return res

    off = recall(False)
    on = recall(True)
    print("buried-fact recall@k — whole-turn emb-kNN vs chunk emb-kNN (FTS isolated out)")
    for k in k_list:
        ho, no = off[k]; hc, nc = on[k]
        print(f"  k={k:2}:  whole-turn {100*ho//no:3d}% ({ho}/{no})   chunk {100*hc//nc:3d}% ({hc}/{nc})")


if __name__ == "__main__":
    main()
