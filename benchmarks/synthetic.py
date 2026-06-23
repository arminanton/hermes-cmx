"""Synthetic planted-fact corpus for cmx evaluation.

Builds a long conversation with K uniquely-identifiable planted facts interleaved
with noise, plus answerable and unanswerable questions. Fully labelled so scoring
is deterministic (no LLM judge needed for the core metrics).
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Fact:
    key: str
    value: str
    sentence: str


@dataclass
class Question:
    text: str
    key: str
    answerable: bool
    expected_value: str = ""


@dataclass
class Corpus:
    turns: list  # (role, content)
    facts: list
    questions: list


_KEYS = ["deploy region", "cache TTL", "staging port", "release tag", "api timeout",
         "db host", "queue name", "feature flag", "retry limit", "log level"]
_VALS = ["eu-west-3", "300 seconds", "6432", "v4.2.0-rc7", "45 ms", "db-staging-7",
         "orders-q9", "FF_NEW_CHECKOUT", "5 attempts", "DEBUG"]


def make_corpus(n_facts: int = 10, noise_between: int = 8, seed: int = 7) -> Corpus:
    rng = random.Random(seed)
    facts, turns = [], []
    keys = _KEYS[:n_facts]
    vals = _VALS[:n_facts]
    for i, (k, v) in enumerate(zip(keys, vals)):
        for j in range(noise_between):
            turns.append(("user", f"random remark {i}-{j} about nothing of consequence"))
        sentence = f"Decision: the {k} is {v}."
        turns.append(("assistant", sentence))
        facts.append(Fact(k, v, sentence))
    questions = [Question(f"what {k} did we decide on?", k, True, v)
                 for k, v in zip(keys, vals)]
    # unanswerable (never planted) → must be refused, not guessed
    for uk in ["smtp relay", "billing cycle", "shard count"]:
        questions.append(Question(f"what {uk} did we decide on?", uk, False))
    rng.shuffle(questions)
    return Corpus(turns=turns, facts=facts, questions=questions)
