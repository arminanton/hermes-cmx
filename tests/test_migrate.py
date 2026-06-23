from cmx import db
from cmx.migrate import import_lcm
from cmx.store import VerbatimStore

LCM_DDL = """
CREATE TABLE messages (
  store_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL, source TEXT DEFAULT '', role TEXT NOT NULL,
  content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT,
  timestamp REAL NOT NULL, token_estimate INTEGER DEFAULT 0, pinned INTEGER DEFAULT 0);
"""


def _fake_lcm(tmp_path):
    p = str(tmp_path / "lcm.db")
    c = db.connect(p)
    c.executescript(LCM_DDL)
    rows = [
        ("s1", "system", "IDENTITY anchor", 1, 5),
        ("s1", "user", "We use port 6432 for staging.", 0, 9),
        ("s1", "assistant", "", 0, 0),            # empty → skipped
        ("s1", "assistant", "Acknowledged: 6432.", 0, 4),
        ("s2", "user", "different session content zeta", 0, 3),
    ]
    for sid, role, content, pinned, tok in rows:
        c.execute("INSERT INTO messages(session_id,role,content,timestamp,token_estimate,pinned)"
                  " VALUES(?,?,?,?,?,?)", (sid, role, content, 1.0, tok, pinned))
    c.close()
    return p


def test_import_lcm_round_trip(tmp_path):
    lcm = _fake_lcm(tmp_path)
    store = VerbatimStore(str(tmp_path / "cmx.db"))
    summary = import_lcm(lcm, store)
    assert summary["imported"] == 4 and summary["skipped_empty"] == 1
    assert summary["sessions"] == 2
    # content recoverable + session-scoped + searchable
    assert store.count("s1") == 3
    assert store.count("s2") == 1
    hits = store.search_fts("staging port", k=5, session_id="s1")
    assert hits and "6432" in store.get_message(hits[0]["id"])["content"]
    # pinned preserved
    assert any(m["content"] == "IDENTITY anchor" for m in store.pinned("s1"))


def test_import_assigns_per_session_turn_order(tmp_path):
    lcm = _fake_lcm(tmp_path)
    store = VerbatimStore(str(tmp_path / "cmx.db"))
    import_lcm(lcm, store)
    s1 = store.recent("s1", 10)
    assert [m["turn_index"] for m in s1] == [0, 1, 2]  # contiguous, empty row excluded
