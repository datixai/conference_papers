"""Reproduce Table 2: BM25 (k1=1.5, b=0.75) on the Northline corpus."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
K1 = 1.5
B = 0.75

TASKS = [
    ("T1 Factoid", "What is the refund window for an unopened item in original packaging?"),
    ("T2 Conflict", "A customer wants a refund on day 20 for an unopened unused item. Can we refund?"),
    ("T3 Stale fee", "What restocking fee do we charge for an opened but unused return?"),
    ("T4 Missing", "A customer in another country asks how long the international warranty lasts."),
    ("T5 Inform others", "Write one sentence I can send to the day-20 customer about their unopened refund."),
    ("T6 Repair", "The assistant said the unopened refund window is 30 days. Is that still correct?"),
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_docs() -> list[tuple[str, list[str]]]:
    docs = []
    for path in sorted(CORPUS.glob("*.txt")):
        tokens = tokenize(path.stem.replace("-", " ") + "\n" + path.read_text(encoding="utf-8"))
        docs.append((path.stem, tokens))
    return docs


def idf(n_docs: int, df: int) -> float:
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def bm25_scores(query: list[str], docs: list[tuple[str, list[str]]]) -> list[tuple[str, float]]:
    n = len(docs)
    df = Counter()
    for _, tokens in docs:
        df.update(set(tokens))
    avgdl = sum(len(tokens) for _, tokens in docs) / n
    ranked = []
    qtf = Counter(query)
    for name, tokens in docs:
        tf = Counter(tokens)
        dl = len(tokens)
        score = 0.0
        for term, _ in qtf.items():
            if term not in tf:
                continue
            freq = tf[term]
            denom = freq + K1 * (1.0 - B + B * dl / avgdl)
            score += idf(n, df[term]) * (freq * (K1 + 1.0)) / denom
        ranked.append((name, round(score, 3)))
    ranked.sort(key=lambda x: (-x[1], x[0]))
    return ranked


def main() -> None:
    docs = load_docs()
    if len(docs) != 18:
        raise SystemExit(f"Expected 18 corpus files, found {len(docs)}")
    results = []
    for task, query in TASKS:
        ranked = bm25_scores(tokenize(query), docs)
        results.append({"task": task, "query": query, "top5": ranked[:5]})
        print(task)
        for i, (name, score) in enumerate(ranked[:3], 1):
            print(f"  {i}. {name} ({score:.3f})")
    out = ROOT / "retrieval_results.reproduced.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
