"""Exact pretraining exposure c(f) per fact and frequency buckets (laptop phase).

  PYTHONPATH=src python -m abst.counts --data data/g0 --config configs/g0.yaml

c(f) = infini-gram CNF count of  subject AND (any accepted answer string)  within max_diff_tokens
tokens, on the OLMo-2 pretraining index (v4_olmo-mix-1124_llama). It is a lower bound on how often
the fact was stated (paraphrases are missed). E1 adds alias/window sensitivity.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import load_yaml, read_jsonl, write_jsonl
from .infinigram import InfiniGram, cnf

BUCKETS = [("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6-20", 6, 20), ("21-100", 21, 100),
           (">100", 101, float("inf"))]
TAIL = ("1", "2")  # k <= 2 among facts with any exposure


def bucket_of(c: int | None) -> str | None:
    if c is None:
        return None
    for name, lo, hi in BUCKETS:
        if lo <= c <= hi:
            return name
    return None


def fact_query(f: dict, max_answers: int = 4) -> str:
    return cnf([f["subj"]], f["answers"][:max_answers])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/g0")
    ap.add_argument("--config", default="configs/g0.yaml")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-interval", type=float, default=0.5, help="seconds between API requests (politeness)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)["counts"]
    data = Path(args.data)
    facts = read_jsonl(data / "facts.jsonl")
    ig = InfiniGram(index=cfg["index"], cache_path=data / "infinigram_cache.jsonl",
                    max_diff_tokens=cfg["max_diff_tokens"], min_interval=args.min_interval)
    ig.ping()
    qs = [fact_query(f, cfg["max_answers"]) for f in facts]
    res = ig.count_many(qs, workers=args.workers)
    rows = [{"fact_id": f["fact_id"], "c": None if r is None else r["count"],
             "c_approx": None if r is None else r["approx"], "bucket": bucket_of(None if r is None else r["count"]),
             "query": q} for f, r, q in zip(facts, res, qs)]
    write_jsonl(data / "counts.jsonl", rows)
    from collections import Counter

    print("bucket sizes:", dict(Counter(r["bucket"] for r in rows)))


if __name__ == "__main__":
    main()
