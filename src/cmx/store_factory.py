"""Backend selector — returns the verbatim store implementation.

cmx defaults to a zero-config **SQLite** verbatim store (FTS5 + trigram +
embeddings), which is the proven path for a drop-in context engine and needs no
external services. **Postgres** (pgvector + pg_trgm) is an opt-in scale backend:
set ``cmx.backend: postgres`` in config (or provide a DSN via ``cmx.pg_dsn`` /
the ``CMX_PG_DSN`` env) and the same contract runs on Postgres instead.
"""
from __future__ import annotations

import os

from .config import CmxConfig


def _resolve_dsn(cfg: CmxConfig) -> str:
    return (
        cfg.resolved_pg_dsn()
        or os.environ.get("CMX_PG_DSN", "")
        or os.environ.get("SESSIONS_PG_DSN", "")
        or os.environ.get("HERMES_SESSIONS_PG_DSN", "")
    )


def make_store(cfg: CmxConfig):
    """Return the cmx verbatim store for the resolved backend.

    SQLite by default (zero-config). Postgres when ``cmx.backend`` is
    ``postgres`` or a DSN is supplied; a postgres backend with no resolvable
    DSN is a hard error rather than a silent SQLite fallback.
    """
    if cfg.resolved_backend() == "postgres":
        dsn = _resolve_dsn(cfg)
        if not dsn:
            raise RuntimeError(
                "cmx backend is 'postgres' but no DSN resolved. Set cmx.pg_dsn "
                "in config.yaml or CMX_PG_DSN / SESSIONS_PG_DSN in the environment "
                "(or use the default SQLite backend by leaving cmx.backend unset)."
            )
        from .pg_store import PostgresVerbatimStore

        return PostgresVerbatimStore(
            dsn, use_trigram=cfg.use_trigram, use_graph=cfg.use_graph
        )

    from .store import VerbatimStore

    return VerbatimStore(
        cfg.resolved_db_path(), use_trigram=cfg.use_trigram, use_graph=cfg.use_graph
    )
