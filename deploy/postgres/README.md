# cmx Postgres backend — true multi-writer context storage

Containerized **Postgres 17 + pgvector + pg_trgm**, the concurrent-write backend for
hermes-cmx. Built for highly-parallel workloads: many Hermes tabs × many sub-agents each,
plus MaxAI-style 30× council fan-out — all writing context simultaneously.

## Why Postgres (proven, not assumed)

SQLite (even WAL) allows **many readers but only ONE writer at a time** — parallel writes
are impossible by design. Measured on this box, same workload (50 concurrent writers ×
200 rows = 10,000 inserts), `src/venv/bin/python3`:

| Backend                                         | Throughput      | Wall   | Errors |
| ----------------------------------------------- | --------------- | ------ | ------ |
| **Postgres 17 (this recipe)**                   | **~10,700 rows/s** | 0.93s  | 0      |
| SQLite — shared conn + global lock (cmx today)  | ~970 rows/s     | 10.3s  | 0      |
| SQLite — connection-per-thread + WAL (best case)| ~890 rows/s     | 11.3s  | 0      |

**~11× faster.** Note the connection-per-thread variant is *slower* than the shared lock:
WAL serializes writers at the file level regardless, so per-connection retry/fsync overhead
makes it worse. SQLite write-concurrency tuning is a dead end for this workload.

Reproduce: `sqlite_baseline.py` (SQLite) vs `spike_verify.py` (Postgres).

## Retrieval parity — all three cmx signals work on Postgres

`spike_verify.py` proves each, with session isolation:

- **FTS** (`tsvector`/`tsquery`, GIN) — the FTS5 analogue.
- **Trigram** (`pg_trgm`, GIN) — substring/identifier recall. Use the **`<%` word-similarity
  operator** (not whole-string `%`) — it matches a short needle inside a long message, which
  is what the SQLite trigram tokenizer does. (`'Polar' <% content` → matches "Polaris".)
- **Vector** (`pgvector`, HNSW, cosine `<=>`) — embeddings.

RRF fusion (operates on id lists) is backend-agnostic and is reused unchanged.

## Usage

```bash
./run.sh                      # bring up + health-check + show extensions
# or:
podman compose -f compose.yaml up -d

# verify everything (parity + concurrency):
python3 spike_verify.py
```

- Host port **5433** → container 5432 (5432 / 540x were busy on this box).
- DSN: `host=127.0.0.1 port=5433 dbname=cmx user=cmx password=cmx_local_dev`
- Data persists in the `cmx-pgdata` volume. Schema + extensions auto-apply on first boot
  from `init/01-schema.sql`.
- Driver: `psycopg[binary]` (bundles libpq — no host build, glibc-safe).

## Files

- `compose.yaml` — the service (tuned: `max_connections=400`, `synchronous_commit=off`,
  `wal_compression=on`).
- `init/01-schema.sql` — messages + FTS + trigram + pgvector + session_lineage; extensions.
- `run.sh` — idempotent bring-up + health check.
- `spike_verify.py` — parity (FTS/trigram/vector/isolation) + concurrency benchmark.
- `sqlite_baseline.py` — same-workload SQLite numbers for the comparison above.

## Tuning notes

- **Tuned LEAN for a RAM-constrained host.** This box has ~29GB RAM and runs heavy swap
  (incl. a 32GB zram device); the freezing ceiling is RAM, not the DB engine. So PG is sized
  conservatively (`shared_buffers=256MB`, `work_mem=8MB`) to stay a good citizen alongside the
  user's many Hermes tabs. On a bigger-RAM instance, raise `shared_buffers` (~25% of RAM) and
  `effective_cache_size` (~50-75%).
- **Use PgBouncer for the many-agent workload.** 50-110 agents + 30 council seats should NOT
  each hold a real PG backend (each backend costs RAM). Put **PgBouncer** (transaction pooling)
  in front so hundreds of agent connections multiplex onto ~20-40 real backends. This is the
  standard answer for high client counts on limited RAM and matters more than raising
  `max_connections`.
- `synchronous_commit=off` — for a context store, losing the last few hundred ms of writes on
  a hard crash is acceptable; it's a large write-throughput win. Flip to `on` for strict
  durability.
- **Storage:** data lives on the single EBS volume (network SSD). There is no separate local
  instance-store NVMe to exploit here. If IO-bound, raise the EBS volume's provisioned IOPS
  (AWS-side) or give PG its own EBS volume with a separate IOPS budget. Do **not** put durable
  PG data on zram (ephemeral + consumes RAM).
