"""cmx.substrate — the shared Postgres layer.

The reusable foundation for the unified platform: one connection pool per DSN, shared by
every domain (context, memory, pintel, ...). Connection-per-operation from a bounded pool
gives true concurrent reads AND writes (the whole reason cmx moves off single-writer SQLite).

Kept deliberately small and dependency-light: psycopg3 + psycopg_pool (libpq bundled via
psycopg[binary], so it's glibc-safe — no host build).
"""
from __future__ import annotations

import threading
from typing import Optional

try:
    import psycopg  # noqa: F401
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
    _PG_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    _PG_AVAILABLE = False
    dict_row = None  # type: ignore
    ConnectionPool = None  # type: ignore


def pg_available() -> bool:
    return _PG_AVAILABLE


# One pool per DSN, process-wide. Domains share the pool so a box running many stores
# does not open N independent connection sets.
_pools: dict[str, "ConnectionPool"] = {}
_lock = threading.RLock()


def get_pool(dsn: str, *, min_size: int = 2, max_size: int = 32,
             timeout: float = 30.0) -> "ConnectionPool":
    """Return the shared ConnectionPool for ``dsn`` (created on first use).

    Bounded on purpose: on a RAM-tight host each PG backend costs memory, so the driver
    pool stays modest. For very high agent counts, front the server with PgBouncer and
    point the DSN at it — this pool then multiplexes onto the pooler.
    """
    if not _PG_AVAILABLE:
        raise RuntimeError("psycopg / psycopg_pool not installed — cannot use the Postgres backend")
    with _lock:
        pool = _pools.get(dsn)
        if pool is None:
            pool = ConnectionPool(
                conninfo=dsn,
                min_size=min_size,
                max_size=max_size,
                timeout=timeout,
                kwargs={"row_factory": dict_row, "autocommit": False},
                open=True,
            )
            _pools[dsn] = pool
        return pool


def close_all() -> None:
    """Close every pool (process shutdown / tests)."""
    with _lock:
        for pool in _pools.values():
            try:
                pool.close()
            except Exception:
                pass
        _pools.clear()
