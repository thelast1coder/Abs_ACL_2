"""Draw the G0 fact sample, balanced across exposure buckets (laptop phase).

  PYTHONPATH=src python -m abst.sample --data data/g0 --config configs/g0.yaml

Takes up to `per_bucket` facts from every bucket (all facts if the bucket is smaller), drops facts
whose subject is used in the few-shot prompt, and facts without counts.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from .common import load_yaml, read_jsonl, write_jsonl
from .prompts import FEWSHOT_SUBJECTS


def draw(facts, counts, per_bucket: int, seed: int = 0):
    cmap = {c["fact_id"]: c for c in counts}
    by = defaultdict(list)
    for f in facts:
        c = cmap.get(f["fact_id"])
        if not c or c["bucket"] is None or f["subj"] in FEWSHOT_SUBJECTS or not f["answers"]:
            continue
        by[c["bucket"]].append({**f, "c": c["c"], "bucket": c["bucket"]})
    rng = random.Random(seed)
    out = []
    for b in sorted(by):
        lst = by[b]
        rng.shuffle(lst)
        out.extend(lst[:per_bucket])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/g0")
    ap.add_argument("--config", default="configs/g0.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)["sample"]
    data = Path(args.data)
    s = draw(read_jsonl(data / "facts.jsonl"), read_jsonl(data / "counts.jsonl"), cfg["per_bucket"], cfg["seed"])
    write_jsonl(data / "g0_sample.jsonl", s)
    print("G0 sample by bucket:", dict(Counter(f["bucket"] for f in s)))


if __name__ == "__main__":
    main()
