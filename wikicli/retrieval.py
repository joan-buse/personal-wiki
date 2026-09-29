"""Retrieval tool: local BM25 keyword search over saved passages.

No model and no network are involved; `wiki search` exposes these results directly.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .sources import Passage

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had
has have having he her here hers herself him himself his how i if in into is it its itself just
me more most my myself no nor not now of off on once only or other our ours ourselves out over own
same she should so some such than that the their theirs them themselves then there these they this
those through to too under until up very was we were what when where which while who whom why will
with would you your yours yourself yourselves tell please say says said
""".split())


def stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("sses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


@dataclass
class Hit:
    passage: Passage
    score: float
    coverage: float      # share of distinct query terms present in the passage
    matched: list[str]


class Index:
    def __init__(self, passages: list[Passage]):
        self.passages = passages
        self.tokens = [tokenize(p.section + " " + p.text) for p in passages]
        self.tf = [Counter(t) for t in self.tokens]
        self.df: Counter = Counter()
        for t in self.tokens:
            self.df.update(set(t))
        self.avg_len = (sum(len(t) for t in self.tokens) / len(self.tokens)) if self.tokens else 0.0

    @classmethod
    def load(cls, path: Path) -> "Index":
        if not path.exists():
            raise FileNotFoundError(f"No retrieval index at {path}. Run `wiki ingest` first.")
        passages = [Passage(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
        return cls(passages)

    def search(self, query: str, k: int = 5, kinds: tuple[str, ...] = ("raw",),
               k1: float = 1.5, b: float = 0.75) -> list[Hit]:
        q_terms = list(dict.fromkeys(tokenize(query)))
        if not q_terms or not self.passages:
            return []
        n = len(self.passages)
        hits = []
        for i, p in enumerate(self.passages):
            if p.kind not in kinds:
                continue
            tf, length = self.tf[i], len(self.tokens[i])
            score, matched = 0.0, []
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                matched.append(term)
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * length / (self.avg_len or 1)))
            if score > 0:
                hits.append(Hit(p, round(score, 3), round(len(matched) / len(q_terms), 2), matched))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


def save_passages(path: Path, passages: list[Passage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
