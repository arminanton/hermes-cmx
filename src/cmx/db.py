"""SQLite driver selection.

Prefer ``pysqlite3`` (newer SQLite with FTS5 + trigram tokenizer, as used on the
Hermes host via HERMES_SQLITE_DRIVER=pysqlite3); fall back to stdlib ``sqlite3``.
"""
from __future__ import annotations

import sqlite3 as _stdlib
import threading
from typing import Any


def _select_driver():
    try:
        import pysqlite3 as drv  # type: ignore
        return drv
    except Exception:
        return _stdlib


driver = _select_driver()
sqlite_version: str = getattr(driver, "sqlite_version", _stdlib.sqlite_version)


class _LockedConnection:
    """Thread-safe wrapper around a single SQLite connection.

    cmx keeps ONE long-lived connection per store, but the host uses the engine from
    different worker threads (e.g. ``asyncio.to_thread`` runs each turn on a pooled
    thread; compaction can fire on yet another). Default sqlite3/pysqlite3 connections
    are pinned to their creating thread (``check_same_thread=True``) and raise
    ``ProgrammingError: SQLite objects created in a thread can only be used in that same
    thread`` the moment another thread touches them — which crashed turns / silently
    dropped memory. We open with ``check_same_thread=False`` and serialize every
    statement with a reentrant lock, so cross-thread use is safe and concurrent writers
    cannot race (the store is append-only + WAL, so this is the whole story).
    """

    __slots__ = ("_raw", "_lock")

    def __init__(self, raw: Any):
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_lock", threading.RLock())

    def execute(self, *a, **k):
        # Allocate an EXPLICIT cursor per statement instead of reusing the
        # connection's implicit internal cursor. pysqlite3 keeps result state on
        # the implicit cursor; with one shared connection across threads
        # (check_same_thread=False — see class docstring), an undrained result
        # set from one statement makes the NEXT connection-level .execute() raise
        # ``OperationalError: another row available``. A dedicated cursor isolates
        # each statement's result set, so interleaved turn/compaction/recall
        # threads can't collide on the shared implicit cursor. The RLock still
        # serializes the call; returning the cursor preserves the prior API
        # (callers do ``.fetchone()`` / ``.fetchall()`` / ``.lastrowid``).
        with self._lock:
            cur = self._raw.cursor()
            cur.execute(*a, **k)
            return cur

    def executemany(self, *a, **k):
        with self._lock:
            cur = self._raw.cursor()
            cur.executemany(*a, **k)
            return cur

    def executescript(self, *a, **k):
        with self._lock:
            return self._raw.executescript(*a, **k)

    def commit(self):
        with self._lock:
            return self._raw.commit()

    def rollback(self):
        with self._lock:
            return self._raw.rollback()

    def close(self):
        with self._lock:
            return self._raw.close()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        try:
            return self._raw.__exit__(*exc)
        finally:
            self._lock.release()

    def __getattr__(self, name):
        return getattr(self._raw, name)

    def __setattr__(self, name, value):
        setattr(self._raw, name, value)


def connect(path: str, **kw: Any):
    """Open a thread-safe connection with sane pragmas for an append-only WAL store."""
    kw.setdefault("check_same_thread", False)
    raw = driver.connect(path, isolation_level=None, **kw)
    raw.row_factory = driver.Row
    raw.execute("PRAGMA journal_mode=WAL")
    raw.execute("PRAGMA synchronous=NORMAL")
    raw.execute("PRAGMA foreign_keys=ON")
    return _LockedConnection(raw)


def fts5_tokenizers_available(conn) -> dict[str, bool]:
    """Probe which FTS5 tokenizers the live SQLite supports (R2 guard)."""
    out: dict[str, bool] = {}
    for tok in ("unicode61", "trigram"):
        try:
            conn.execute(
                f"CREATE VIRTUAL TABLE temp._probe_{tok} USING fts5(x, tokenize='{tok}')"
            )
            conn.execute(f"DROP TABLE temp._probe_{tok}")
            out[tok] = True
        except Exception:
            out[tok] = False
    return out
