"""Thread-safety regression tests.

cmx holds one long-lived SQLite connection per store, but the host uses the engine from
different worker threads (e.g. asyncio.to_thread runs each turn on a pooled thread, and
compaction can fire on another). A default sqlite3/pysqlite3 connection is pinned to its
creating thread and raises ``ProgrammingError`` when touched from another — which crashed
turns and silently dropped memory in long conversations. db.connect() opens with
check_same_thread=False and serializes statements with a reentrant lock; these tests lock
that behaviour in.
"""
import threading

from cmx.store import VerbatimStore


def _store(tmp_path):
    return VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)


def test_connection_usable_from_another_thread(tmp_path):
    """A store built on the main thread must be usable from a worker thread."""
    store = _store(tmp_path)
    store.add_message("s1", 0, "user", "hello world")

    result = {}

    def worker():
        try:
            # the exact call that crashed during compaction ingest
            result["hashes"] = store.content_hashes("s1")
            store.add_message("s1", 1, "assistant", "reply from worker thread")
            result["count"] = store.count("s1")
        except Exception as e:  # pragma: no cover - failure path
            result["error"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert "error" not in result, result.get("error")
    assert result["count"] == 2


def test_concurrent_writes_are_serialized(tmp_path):
    """Many threads writing/reading the same store must not error or lose rows."""
    store = _store(tmp_path)
    errors = []

    def writer(tid):
        try:
            for i in range(40):
                store.add_message(f"s{tid}", i, "user", f"thread {tid} note {i} about widgets")
                store.content_hashes(f"s{tid}")
                store.search_fts("widgets", k=3, session_id=f"s{tid}")
        except Exception as e:  # pragma: no cover - failure path
            errors.append(f"T{tid}: {type(e).__name__}: {e}")

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:3]
    assert store.count() == 8 * 40
