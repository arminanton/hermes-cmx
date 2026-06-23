"""hermes-cmx — Context Memory eXchange.

Retrieval-first, no-lossy-summarization context engine with engine-enforced
grounding. See docs/ for the design. This package is import-safe without Hermes
installed: the engine logic is pure-Python and dependency-injects its LLM client
so it can be unit-tested deterministically.
"""

__version__ = "0.1.0-dev"

from .config import CmxConfig

__all__ = ["CmxConfig", "__version__"]
