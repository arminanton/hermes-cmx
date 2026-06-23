import numpy as np

from cmx.config import CmxConfig
from cmx.embeddings import MockEmbedder, cosine_topk, pack, unpack
from cmx.retrieval import HybridRetriever
from cmx.store import VerbatimStore


def test_pack_unpack_roundtrip():
    v = [0.1, 0.2, 0.3]
    assert np.allclose(unpack(pack(v)), np.asarray(v, dtype="float32"))


def test_cosine_topk_orders_by_similarity():
    ids = [10, 11, 12]
    M = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype="float32")
    out = cosine_topk([1.0, 0.0], ids, M, k=2)
    assert out[0] == 10 and out[1] == 11   # most aligned first


def test_store_vectors_roundtrip(tmp_path):
    s = VerbatimStore(str(tmp_path / "cmx.db"))
    a = s.add_message("sess", 0, "user", "alpha")
    b = s.add_message("sess", 1, "user", "beta")
    assert {i for i, _ in s.messages_missing_embeddings("sess")} == {a, b}
    s.store_embeddings([(a, [1.0, 0.0]), (b, [0.0, 1.0])], model="m", dim=2)
    assert s.messages_missing_embeddings("sess") == []
    ids, mat = s.get_session_vectors("sess")
    assert ids == [a, b] and mat.shape == (2, 2)


def test_retrieval_uses_embeddings_source(tmp_path):
    cfg = CmxConfig()
    cfg.use_embeddings = True
    s = VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True, use_graph=True)
    r = HybridRetriever(s, cfg, embedder=MockEmbedder())
    fid = s.add_message("sess", 0, "assistant", "the cache retention window is ninety days")
    for t in range(1, 20):
        s.add_message("sess", t, "user", f"noise {t}")
    hits = r.retrieve("cache retention window ninety days", session_id="sess")
    hit = next((h for h in hits if h.id == fid), None)
    assert hit is not None and "emb" in hit.sources   # embeddings contributed


def test_graceful_without_embedder(tmp_path):
    cfg = CmxConfig()
    cfg.use_embeddings = True            # enabled, but no embedder injected
    s = VerbatimStore(str(tmp_path / "cmx.db"))
    r = HybridRetriever(s, cfg, embedder=None)
    s.add_message("sess", 0, "user", "content here")
    assert isinstance(r.retrieve("content", session_id="sess"), list)   # no crash
