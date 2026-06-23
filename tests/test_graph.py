from cmx.config import CmxConfig
from cmx.engine import CmxEngine
from cmx.graph import extract
from cmx.retrieval import HybridRetriever
from cmx.store import VerbatimStore


def test_extraction_entities_and_relations():
    ents, rels = extract("db-7 connects to cache-3.")
    assert "db-7" in ents and "cache-3" in ents
    assert ("db-7", "connects to", "cache-3") in rels


def _chain(tmp_path):
    s = VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True, use_graph=True)
    s.add_message("sess", 0, "assistant", "The primary db is db-7.")        # id 1
    mid2 = s.add_message("sess", 1, "assistant", "db-7 connects to cache-3.")  # id 2
    port = s.add_message("sess", 2, "assistant", "cache-3 uses port 6379.")    # id 3 (2 hops)
    for t in range(3, 30):
        s.add_message("sess", t, "user", f"noise {t}")
    return s, port


def test_multihop_reaches_far_node(tmp_path):
    s, port = _chain(tmp_path)
    assert s.graph is not None
    ids = s.graph.multihop_messages(["db-7"], hops=2)
    assert port in ids   # the port message is 2 hops from db-7


def test_graph_retrieval_surfaces_what_fts_cannot(tmp_path):
    # "cache-3 uses port 6379" never mentions db-7 → only the graph can connect them.
    s, port = _chain(tmp_path)
    cfg = CmxConfig()
    r = HybridRetriever(s, cfg)
    # baseline: pure lexical search for db-7 cannot return the port message
    assert port not in [h["id"] for h in s.search_fts("db-7", 20, "sess")]
    assert port not in [h["id"] for h in s.search_trgm("db-7", 20, "sess")]
    # hybrid retrieval WITH the graph does
    hits = r.retrieve("what about db-7", session_id="sess")
    hit = next((h for h in hits if h.id == port), None)
    assert hit is not None and "graph" in hit.sources


def test_cmx_graph_query_tool(tmp_path):
    s, port = _chain(tmp_path)
    eng = CmxEngine(CmxConfig(), s, HybridRetriever(s, CmxConfig()), model_client=None)
    out = eng._serve_tool("sess", "cmx_graph_query", {"entity": "db-7"})
    assert f"[id={port}]" in out and "6379" in out


def test_graph_degrades_when_disabled(tmp_path):
    s = VerbatimStore(str(tmp_path / "cmx.db"), use_graph=False)
    assert s.graph is None
    cfg = CmxConfig()
    cfg.use_graph = False
    r = HybridRetriever(s, cfg)
    s.add_message("sess", 0, "user", "plain content here")
    # still works (FTS only), just no graph source
    assert isinstance(r.retrieve("plain content", session_id="sess"), list)
