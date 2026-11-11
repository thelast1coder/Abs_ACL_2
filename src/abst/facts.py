"""Facts for ABS: PopQA (Mallen et al. 2023), with exact OLMo-2 co-occurrence counts.

  PYTHONPATH=src python -m abst.facts --out data/g0            # download PopQA (Hugging Face)
  PYTHONPATH=src python -m abst.facts --out data/g0 --from-file data/popqa.jsonl

PopQA fields used: id, subj, prop, obj, s_aliases, o_aliases, s_pop, question, possible_answers.
List-valued fields arrive as JSON strings; we parse them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, write_jsonl


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    try:
        v = json.loads(x)
        return v if isinstance(v, list) else [str(v)]
    except Exception:
        return [str(x)]


MIN_ALIAS_CHARS = 4  # a non-canonical alias must normalise to at least this many characters


def filter_answers(obj: str | None, raw: list[str]) -> list[str]:
    """Drop aliases that match ordinary text. PopQA alias lists contain "can"/"ca" (Canada), "in" (India),
    "US" (normalises to the pronoun "us") and emoji flags, so whole-word grading marks "I can't answer"
    as correct and co-occurrence counts are inflated. The canonical answer `obj` is always kept, first.
    A non-canonical alias is kept only if it normalises to >= MIN_ALIAS_CHARS characters and is not an
    English stopword. Decision logged 2026-11-11 in notes/decisions.md (before any G0 data)."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    from .grading import norm

    out = [obj] if obj and norm(obj) else []
    for a in raw:
        na = norm(a)
        if a == obj or not na or len(na) < MIN_ALIAS_CHARS or na in ENGLISH_STOP_WORDS:
            continue
        out.append(a)
    if not out:  # canonical answer missing or empty: fall back to the raw list
        out = list(raw)
    return list(dict.fromkeys(out))


def normalise_row(r: dict) -> dict:
    raw = [a for a in dict.fromkeys(_as_list(r.get("possible_answers"))) if a]
    return {"fact_id": f"popqa:{r['id']}", "subj": r["subj"], "prop": r["prop"], "obj": r.get("obj"),
            "question": r["question"], "answers": filter_answers(r.get("obj"), raw), "answers_raw": raw,
            "s_aliases": _as_list(r.get("s_aliases")), "s_pop": r.get("s_pop")}


def load_popqa(from_file: str | None = None) -> list[dict]:
    if from_file:
        rows = read_jsonl(from_file)
    else:
        from datasets import load_dataset

        rows = list(load_dataset("akariasai/PopQA", split="test"))
    return [normalise_row(r) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/g0")
    ap.add_argument("--from-file", default=None)
    args = ap.parse_args()
    facts = load_popqa(args.from_file)
    out = Path(args.out)
    write_jsonl(out / "facts.jsonl", facts)
    print(f"{len(facts)} facts -> {out / 'facts.jsonl'}")


if __name__ == "__main__":
    main()
