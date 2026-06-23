#!/usr/bin/env python3
"""SQLite concurrency baseline — same workload as spike_verify.py's Postgres bench,
so we have an apples-to-apples PG-vs-SQLite number. Run with Hermes venv (pysqlite3)."""
import os, time, threading, tempfile, statistics
import sys
# use the same custom pysqlite3 cmx uses (trigram build)
try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3

N_WRITERS = 50
ROWS_EACH = 200
TOTAL = N_WRITERS * ROWS_EACH

DDL = """
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT, turn_index INTEGER, role TEXT, content TEXT,
  content_hash TEXT, created_at REAL
);
"""

def bench(label, path, mode):
    """mode: 'shared_locked' (cmx's _LockedConnection model: one conn + global lock)
             'conn_per_thread' (best-case SQLite: each thread own conn, WAL)."""
    if os.path.exists(path):
        os.remove(path)
    setup = sqlite3.connect(path)
    setup.execute("PRAGMA journal_mode=WAL")
    setup.execute("PRAGMA synchronous=NORMAL")
    setup.executescript(DDL)
    setup.commit(); setup.close()

    errors, latencies = [], []
    barrier = threading.Barrier(N_WRITERS)

    shared_conn = None
    lock = threading.RLock()
    if mode == "shared_locked":
        shared_conn = sqlite3.connect(path, check_same_thread=False)
        shared_conn.execute("PRAGMA journal_mode=WAL")
        shared_conn.execute("PRAGMA busy_timeout=30000")

    def writer(wid):
        try:
            if mode == "conn_per_thread":
                conn = sqlite3.connect(path, timeout=30)
                conn.execute("PRAGMA busy_timeout=30000")
            barrier.wait()
            t0 = time.perf_counter()
            for r in range(ROWS_EACH):
                content = f"worker {wid} row {r}"
                row = ("w-%d" % wid, r, "user", content, "h", time.time())
                if mode == "shared_locked":
                    with lock:
                        shared_conn.execute(
                            "INSERT INTO messages (session_id,turn_index,role,content,content_hash,created_at)"
                            " VALUES (?,?,?,?,?,?)", row)
                        shared_conn.commit()
                else:
                    conn.execute(
                        "INSERT INTO messages (session_id,turn_index,role,content,content_hash,created_at)"
                        " VALUES (?,?,?,?,?,?)", row)
                    conn.commit()
            latencies.append(time.perf_counter() - t0)
            if mode == "conn_per_thread":
                conn.close()
        except Exception as e:
            errors.append(f"w{wid}: {type(e).__name__}: {e}")

    t0 = time.perf_counter()
    threads = [threading.Thread(target=writer, args=(i,)) for i in range(N_WRITERS)]
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - t0
    if shared_conn:
        got = shared_conn.execute("SELECT count(*) FROM messages").fetchone()[0]
        shared_conn.close()
    else:
        c = sqlite3.connect(path); got = c.execute("SELECT count(*) FROM messages").fetchone()[0]; c.close()

    print(f"  [{label}]")
    print(f"    inserted={got}/{TOTAL}  errors={len(errors)}  wall={wall:.2f}s  "
          f"throughput={got/wall:,.0f} rows/s")
    if latencies:
        print(f"    per-worker latency: min={min(latencies):.2f}s "
              f"median={statistics.median(latencies):.2f}s max={max(latencies):.2f}s")
    if errors:
        print(f"    ERRORS (first 3): {errors[:3]}")
    return got, wall, len(errors)

if __name__ == "__main__":
    print(f"== SQLite concurrency baseline ==  ({sqlite3.__name__}, sqlite {sqlite3.sqlite_version})")
    print(f"workload: {N_WRITERS} writers x {ROWS_EACH} rows = {TOTAL} inserts\n")
    d = tempfile.mkdtemp(prefix="cmx-sqlite-bench-")
    print("[A] cmx current model — single shared connection + global RLock (fully serialized):")
    bench("shared_locked", os.path.join(d, "a.db"), "shared_locked")
    print("\n[B] best-case SQLite — connection-per-thread + WAL + busy_timeout:")
    bench("conn_per_thread", os.path.join(d, "b.db"), "conn_per_thread")
    print("\n(compare throughput vs Postgres spike_verify.py: 50 writers ~10.7K rows/s, 0 errors)")
