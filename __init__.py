"""Hermes plugin entrypoint for hermes-cmx (context engine).

Mirrors hermes-lcm's __init__.register(ctx): builds the cmx context engine and
registers it so `context.engine: cmx` selects it. Adds the cmx source tree to the
path if installed out-of-tree (dev layout).
"""
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _ensure_cmx_importable():
    try:
        import cmx  # noqa: F401
        return
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "src"),
                 os.path.join(here, "..", "src")):
        if os.path.isdir(os.path.join(cand, "cmx")):
            sys.path.insert(0, os.path.abspath(cand))
            return


def register(ctx):
    """Plugin entry point — register the cmx context engine."""
    _ensure_cmx_importable()
    from cmx.hermes_engine import CmxContextEngine
    engine = CmxContextEngine()
    ctx.register_context_engine(engine)
    logger.info("cmx plugin loaded — retrieval-first context engine active")
    return engine
