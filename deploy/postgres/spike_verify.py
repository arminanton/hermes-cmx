#!/usr/bin/env python3
"""cmx Postgres spike — prove (1) retrieval parity (FTS + trigram + vector) and
(2) true multi-writer concurrency vs SQLite's serialized writes.

Run with the Hermes venv:
  python3 deploy/postgres/spike_verify.py
"""
import os, time, threading, random, statistics, math, re
import psycopg
from psycopg.rows import dict_row

DSN = os.environ.get("CMX_PG_DSN", "host=127.0.0.1 port=5433 dbname=cmx user=cmx password=cmx_local_dev")
DIM = 1536

def fake_embedding(text: str) -> list[float]:
    """Deterministic pseudo-embedding so we can test vector search without an API.
    Hash words into a sparse-ish vector; normalize. Good enough to prove the SQL path."""
    v = [0.0] * DIM
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        h = hash(w)
        v[h % DIM] += 1.0
        v[(h >> 8) % DIM] += 0.5
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]

def vec_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"

def reset():
    with psycopg.connect(DSN) as c, c.cursor() as cur:
        cur.execute("TRUNCATE messages, message_embeddings, session_lineage RESTART IDENTITY CASCADE")
        c.commit()

# ───────────────────────── retrieval parity ─────────────────────────
SAMPLES = [
    ("s-alpha", "user",      "My project is called Polaris and the deploy port is 8443."),
    ("s-alpha", "assistant", "Noted: Polaris listens on port 8443 in production."),
    ("s-alpha", "user",      "The database password rotation runs every 90 days via cron."),
    ("s-alpha", "user",      "My cat's name is Mittens and she is a tabby."),
    ("s-beta",  "user",      "Unrelated session about quarterly revenue forecasting models."),
]

def seed_samples():
    with psycopg.connect(DSN) as c, c.cursor() as cur:
        for i, (sid, role, content) in enumerate(SAMPLES):
            cur.execute(
                "INSERT INTO messages (session_id, turn_index, role, content, content_hash, created_at) "
                "VALUES (%s,%s,%s,%s,md5(%s),%s) RETURNING id",
                (sid, i, role, content, content, time.time()),
            )
            mid = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO message_embeddings (message_id, session_id, embedding) VALUES (%s,%s,%s)",
                (mid, sid, vec_literal(fake_embedding(content))),
            )
        c.commit()

def test_fts():
    with psycopg.connect(DSN, row_factory=dict_row) as c, c.cursor() as cur:
        cur.execute(
            "SELECT content, ts_rank(fts, plainto_tsquery('simple', %s)) AS rank "
            "FROM messages WHERE session_id=%s AND fts @@ plainto_tsquery('simple', %s) "
            "ORDER BY rank DESC LIMIT 5",
            ("deploy port", "s-alpha", "deploy port"),
        )
        rows = cur.fetchall()
    hit = any("8443" in r["content"] for r in rows)
    print(f"  [FTS]     query='deploy port' -> {len(rows)} hits, found-port={hit}")
    return hit

def test_trigram():
    with psycopg.connect(DSN, row_factory=dict_row) as c, c.cursor() as cur:
        # substring/fuzzy: 'Polar' should match 'Polaris' via WORD trigram similarity.
        # The <% operator (word_similarity) finds the best-matching substring/word, which
        # is the correct analogue of SQLite's trigram tokenizer — plain % (whole-string
        # similarity) is too strict for a short needle inside a long message.
        cur.execute(
            "SELECT content, word_similarity(%s, content) AS sim FROM messages "
            "WHERE session_id=%s AND %s <%% content ORDER BY sim DESC LIMIT 5",
            ("Polar", "s-alpha", "Polar"),
        )
        rows = cur.fetchall()
    hit = any("Polaris" in r["content"] for r in rows)
    print(f"  [TRGM]    query='Polar' -> {len(rows)} hits, found-Polaris={hit}")
    return hit

def test_vector():
    # NOTE: this proves the pgvector MECHANICS (vector storage, cosine <=> operator, HNSW
    # ordering, session filter) using a deterministic toy embedding. Real semantic quality
    # comes from the production embedding model and is measured in the cmx benchmarks, not here.
    # Query with distinctive tokens unique to the target row, and assert (a) it ranks #1 and
    # (b) the returned cosine distances are monotonically non-decreasing (ORDER BY <=> works).
    q = vec_literal(fake_embedding("Mittens tabby"))
    with psycopg.connect(DSN, row_factory=dict_row) as c, c.cursor() as cur:
        cur.execute(
            "SELECT m.content, (e.embedding <=> %s) AS dist FROM message_embeddings e "
            "JOIN messages m ON m.id=e.message_id WHERE e.session_id=%s "
            "ORDER BY e.embedding <=> %s LIMIT 4",
            (q, "s-alpha", q),
        )
        rows = cur.fetchall()
    dists = [r["dist"] for r in rows]
    sorted_ok = dists == sorted(dists)
    top1_cat = rows and "Mittens" in rows[0]["content"]
    hit = bool(top1_cat and sorted_ok)
    print(f"  [VECTOR]  query='Mittens tabby' -> top1-cat={bool(top1_cat)}, "
          f"distances-monotonic={sorted_ok} dists={[round(d,3) for d in dists]}")
    return hit

def test_isolation():
    # s-beta content must never leak into an s-alpha query
    with psycopg.connect(DSN, row_factory=dict_row) as c, c.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM messages WHERE session_id=%s AND content ILIKE %s",
                    ("s-alpha", "%revenue%"))
        leak = cur.fetchone()["n"]
    print(f"  [ISOLATE] s-beta 'revenue' visible under s-alpha filter = {leak} (want 0)")
    return leak == 0

# ───────────────────────── concurrency benchmark ─────────────────────────
def concurrency_bench(n_writers=50, rows_each=200):
    reset()
    errors = []
    latencies = []
    barrier = threading.Barrier(n_writers)

    def writer(wid):
        try:
            with psycopg.connect(DSN) as c:
                c.execute("SET synchronous_commit=off")
                barrier.wait()  # all start together -> real contention
                t0 = time.perf_counter()
                with c.cursor() as cur:
                    for r in range(rows_each):
                        content = f"worker {wid} row {r} token-{random.randint(0,9999)}"
                        cur.execute(
                            "INSERT INTO messages (session_id, turn_index, role, content, content_hash, created_at) "
                            "VALUES (%s,%s,'user',%s, md5(%s), %s)",
                            (f"w-{wid}", r, content, content, time.time()),
                        )
                    c.commit()
                latencies.append(time.perf_counter() - t0)
        except Exception as e:
            errors.append(f"w{wid}: {type(e).__name__}: {e}")

    total = n_writers * rows_each
    t0 = time.perf_counter()
    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_writers)]
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - t0

    with psycopg.connect(DSN) as c, c.cursor() as cur:
        cur.execute("SELECT count(*) FROM messages")
        got = cur.fetchone()[0]

    print(f"\n  {n_writers} concurrent writers x {rows_each} rows = {total} target inserts")
    print(f"  inserted={got}  errors={len(errors)}  wall={wall:.2f}s  "
          f"throughput={got/wall:,.0f} rows/s")
    if latencies:
        print(f"  per-worker commit-batch latency: "
              f"min={min(latencies):.2f}s median={statistics.median(latencies):.2f}s max={max(latencies):.2f}s")
    if errors:
        print("  ERRORS:", errors[:3])
    return got == total and not errors

if __name__ == "__main__":
    print("== cmx Postgres spike ==")
    print(f"DSN: {DSN}\n")
    reset(); seed_samples()
    print("[1] Retrieval parity (FTS / trigram / vector / isolation):")
    r = [test_fts(), test_trigram(), test_vector(), test_isolation()]
    print(f"  parity: {'ALL PASS' if all(r) else 'FAIL'}\n")

    print("[2] Concurrency — true multi-writer:")
    ok = concurrency_bench(n_writers=50, rows_each=200)
    print(f"\n  concurrency: {'PASS (all rows, no errors)' if ok else 'FAIL'}")
    print("\n== done ==")
