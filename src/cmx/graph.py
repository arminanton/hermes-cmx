"""Entity/relation graph for multi-hop recall (P5).

Rule-based extraction (deterministic, no LLM required) of identifiers and simple
relations ("A connects to B", "the X is Y"). Multi-hop traversal surfaces verbatim
messages that plain similarity retrieval misses — e.g. "db-7 → cache-3 → port 6379"
answers "what port does the primary db's cache use?" even though the port message
never mentions the db. An LLM extractor can be injected later for richer relations;
the rule-based floor degrades gracefully (never worse than hybrid retrieval).
"""
from __future__ import annotations

import re
from typing import Iterable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS gentities (norm TEXT PRIMARY KEY, name TEXT, kind TEXT);
CREATE TABLE IF NOT EXISTS gmentions (entity_norm TEXT, message_id INTEGER);
CREATE TABLE IF NOT EXISTS grelations (src TEXT, rel TEXT, dst TEXT, message_id INTEGER);
CREATE INDEX IF NOT EXISTS idx_gmentions ON gmentions(entity_norm);
CREATE INDEX IF NOT EXISTS idx_grel_src ON grelations(src);
CREATE INDEX IF NOT EXISTS idx_grel_dst ON grelations(dst);
"""

# identifier-like tokens (compound with _ . / : - , or containing a digit)
_IDENT = re.compile(r"\b[A-Za-z][\w]*[_./:-][\w./:-]+\b|\b[\w]*\d[\w./:-]*\b")
_REL = re.compile(
    r"\b([\w./:-]+)\s+(connects to|connect to|uses|use|runs on|run on|maps to|"
    r"depends on|links to|talks to|points to|is|are)\s+([\w./:-]+)", re.IGNORECASE)
_STOP = {"is", "are", "the", "a", "an", "to", "on", "of", "in"}


def _norm(s: str) -> str:
    return s.strip().strip(".,:;").lower()


def extract(content: str):
    ents: set[str] = set()
    for m in _IDENT.findall(content or ""):
        n = _norm(m)
        if len(n) >= 2 and n not in _STOP:
            ents.add(n)
    rels = []
    for src, rel, dst in _REL.findall(content or ""):
        s, d = _norm(src), _norm(dst)
        if s and d and s not in _STOP and d not in _STOP:
            rels.append((s, rel.lower(), d))
            ents.add(s)
            ents.add(d)
    return ents, rels


class GraphIndex:
    def __init__(self, conn):
        self.conn = conn
        self.conn.executescript(_SCHEMA)

    def index(self, message_id: int, content: str) -> None:
        ents, rels = extract(content)
        for n in ents:
            self.conn.execute("INSERT OR IGNORE INTO gentities(norm,name,kind) VALUES(?,?,?)",
                              (n, n, "identifier"))
            self.conn.execute("INSERT INTO gmentions(entity_norm,message_id) VALUES(?,?)",
                              (n, message_id))
        for s, rel, d in rels:
            self.conn.execute("INSERT INTO grelations(src,rel,dst,message_id) VALUES(?,?,?,?)",
                              (s, rel, d, message_id))

    def entities_in(self, text: str) -> list[str]:
        ents, _ = extract(text)
        return list(ents)

    def neighbors(self, entity_norm: str):
        out = []
        for r in self.conn.execute("SELECT dst, message_id FROM grelations WHERE src=?", (entity_norm,)):
            out.append((r["dst"], r["message_id"]))
        for r in self.conn.execute("SELECT src, message_id FROM grelations WHERE dst=?", (entity_norm,)):
            out.append((r["src"], r["message_id"]))
        return out

    def _mention_msgs(self, entity_norm: str):
        return [r["message_id"] for r in self.conn.execute(
            "SELECT message_id FROM gmentions WHERE entity_norm=?", (entity_norm,))]

    def multihop_messages(self, seeds: Iterable[str], hops: int = 2) -> list[int]:
        """BFS from seed entities; return message ids ordered by increasing hop distance."""
        seen_ent = set(seeds)
        frontier = set(seeds)
        ordered: list[int] = []
        # hop 0: direct mentions of the seeds
        for s in seeds:
            ordered.extend(self._mention_msgs(s))
        for _ in range(max(0, hops)):
            nxt = set()
            for ent in frontier:
                for other, mid in self.neighbors(ent):
                    ordered.append(mid)
                    if other not in seen_ent:
                        seen_ent.add(other)
                        nxt.add(other)
            frontier = nxt
            if not frontier:
                break
        # de-dup preserving first (closest-hop) occurrence
        seen, dedup = set(), []
        for m in ordered:
            if m not in seen:
                seen.add(m)
                dedup.append(m)
        return dedup
