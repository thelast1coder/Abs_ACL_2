"""G0 generation pass (GPU phase).

  PYTHONPATH=src python -m abst.run_g0 --role base --model allenai/OLMo-2-1124-7B --data data/g0 --out results/G0
  PYTHONPATH=src python -m abst.run_g0 --role instruct --model allenai/OLMo-2-1124-7B-Instruct --data data/g0 --out results/G0

role=base     : greedy answer + n_samples pure samples (T=1, top_p=1, top_k=0) for self-consistency confidence
role=instruct : one greedy answer per chat variant (S1_plain, S2_abstain)
Facts are the G0 sample written by `abst.sample` (balanced across exposure buckets).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .common import (append_gpu_ledger, build_prompt, estimate_need_gib, load_model, read_jsonl,
                     require_gpu_memory, run_meta, seed_all, write_json, write_jsonl)
from .generation import generate_hf
from .prompts import CHAT_VARIANTS, base_prompt


def run(model, tok, hf_id: str, role: str, data_dir: Path, out_dir: Path, batch_size: int = 32,
        n_samples: int = 10, limit: int | None = None, progress: bool = True, seed: int = 0) -> Path:
    facts = read_jsonl(data_dir / "g0_sample.jsonl")
    if limit:
        facts = facts[:limit]
    od = out_dir / role
    rows = []
    if role == "base":
        prompts = [base_prompt(f["question"]) for f in facts]
        greedy = generate_hf(model, tok, prompts, max_new_tokens=16, do_sample=False, batch_size=batch_size,
                             stops=["\n"], progress=progress)
        samples = generate_hf(model, tok, prompts, max_new_tokens=16, do_sample=True, temperature=1.0, top_p=1.0,
                              num_return_sequences=n_samples, batch_size=max(1, batch_size // 2), stops=["\n"],
                              seed=seed, progress=progress) if n_samples else [[] for _ in facts]
        for f, g, s in zip(facts, greedy, samples):
            rows.append({"fact_id": f["fact_id"], "variant": "B", "response": g[0].strip(),
                         "samples": [x.strip() for x in s]})
    elif role == "instruct":
        for variant, tmpl in CHAT_VARIANTS.items():
            prompts = [build_prompt(tok, tmpl.format(question=f["question"]), chat=True) for f in facts]
            outs = generate_hf(model, tok, prompts, max_new_tokens=48, do_sample=False, batch_size=batch_size,
                               add_special_tokens=False, progress=progress)
            for f, o in zip(facts, outs):
                rows.append({"fact_id": f["fact_id"], "variant": variant, "response": o[0].strip()})
    else:
        raise ValueError(role)
    write_jsonl(od / "generations.jsonl", rows)
    write_json(od / "meta.json", run_meta(model, hf_id, {"role": role, "n_facts": len(facts),
                                                          "n_samples": n_samples if role == "base" else 0}))
    return od


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["base", "instruct"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--data", default="data/g0")
    ap.add_argument("--out", default="results/G0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    seed_all(0)
    if args.device == "cuda":
        require_gpu_memory(estimate_need_gib(args.model, args.batch_size))
    t0 = time.time()
    model, tok = load_model(args.model, device=args.device, revision=args.revision)
    od = run(model, tok, args.model, args.role, Path(args.data), Path(args.out), args.batch_size,
             args.n_samples, args.limit)
    append_gpu_ledger("abst.run_g0", args.model, time.time() - t0, note=str(od))
    print(f"wrote {od}")


if __name__ == "__main__":
    main()
