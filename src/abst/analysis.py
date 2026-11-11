"""G0 analysis for ABS: who pays for abstention? (CPU)

  PYTHONPATH=src python -m abst.analysis --data data/g0 --results results/G0 --config configs/g0.yaml

Exact bookkeeping per exposure bucket k (an identity, no model):
    C_S(k) - C_B(k) = Gain_S(k) - Loss_S(k)
    Gain = facts B got wrong and S gets right
    Loss = facts B got right and S does not (Loss_abs: S abstained; Loss_flip: S answered wrongly)
Tail tax:
    TT = [Loss_S(tail) / Loss_S(all)] / [C_B(tail) / C_B(all)],  tail = buckets k in {1, 2}
    TT = 1: the tail loses answers in proportion to what it contributes; TT = 2: it pays double.
Matched-coverage reference: threshold the base model's own self-consistency confidence so that its
coverage equals S's coverage; compare tail tax. If training is no better than this, it learned
nothing about the tail.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import load_yaml, read_jsonl, write_json, write_jsonl
from .counts import BUCKETS
from .grading import is_abstention, is_correct, same_answer

ORDER = [b[0] for b in BUCKETS]


def grade(sample: list[dict], gens: list[dict]) -> list[dict]:
    by = defaultdict(dict)
    for g in gens:
        by[g["fact_id"]][g["variant"]] = g
    rows = []
    for f in sample:
        g = by.get(f["fact_id"], {})
        if "B" not in g:
            continue
        b = g["B"]
        conf = (np.mean([same_answer(b["response"], s) for s in b.get("samples", [])])
                if b.get("samples") else None)
        row = {"fact_id": f["fact_id"], "bucket": f["bucket"], "c": f["c"],
               "B_correct": is_correct(b["response"], f["answers"]), "B_conf": conf}
        for v, gv in g.items():
            if v == "B":
                continue
            row[f"{v}_correct"] = is_correct(gv["response"], f["answers"])
            row[f"{v}_abstain"] = is_abstention(gv["response"], f["answers"])
        rows.append(row)
    return rows


def bucket_table(rows: list[dict], variant: str) -> dict:
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[r["bucket"]].append(r)
    groups["all"] = rows
    for k, rs in groups.items():
        cb = sum(r["B_correct"] for r in rs)
        cs = sum(r[f"{variant}_correct"] for r in rs)
        gain = sum((not r["B_correct"]) and r[f"{variant}_correct"] for r in rs)
        loss_abs = sum(r["B_correct"] and (not r[f"{variant}_correct"]) and r[f"{variant}_abstain"] for r in rs)
        loss_flip = sum(r["B_correct"] and (not r[f"{variant}_correct"]) and not r[f"{variant}_abstain"] for r in rs)
        loss = loss_abs + loss_flip
        assert cs - cb == gain - loss, "bookkeeping identity violated"
        out[k] = {"N": len(rs), "C_B": cb, "C_S": cs, "gain": gain, "loss": loss, "loss_abs": loss_abs,
                  "loss_flip": loss_flip, "acc_B": cb / len(rs) if rs else None,
                  "acc_S": cs / len(rs) if rs else None,
                  "abstain_S": float(np.mean([r[f"{variant}_abstain"] for r in rs])) if rs else None}
    return out


def _tt(rows, lost, tail):
    cb = np.array([r["B_correct"] for r in rows])
    t = np.array([r["bucket"] in tail for r in rows])
    if lost.sum() == 0 or cb.sum() == 0 or (cb & t).sum() == 0:
        return None
    return float(((lost & t).sum() / lost.sum()) / ((cb & t).sum() / cb.sum()))


def tail_tax(rows, variant: str, tail=("1", "2"), n_boot: int = 1000, seed: int = 0) -> dict:
    lost = np.array([r["B_correct"] and not r[f"{variant}_correct"] for r in rows])
    tt = _tt(rows, lost, tail)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(rows), len(rows))
        rr = [rows[i] for i in idx]
        v = _tt(rr, lost[idx], tail)
        if v is not None:
            boots.append(v)
    return {"tail_tax": tt, "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None,
            "tail_losses": int(sum(lost[i] and rows[i]["bucket"] in tail for i in range(len(rows))))}


def matched_coverage(rows, variant: str, tail=("1", "2")) -> dict | None:
    if any(r["B_conf"] is None for r in rows):
        return None
    cov_s = float(np.mean([not r[f"{variant}_abstain"] for r in rows]))
    conf = np.array([r["B_conf"] for r in rows])
    best = None
    for tau in np.unique(np.concatenate([conf, [1.01]])):
        cov = float(np.mean(conf >= tau))
        if best is None or abs(cov - cov_s) < abs(best[1] - cov_s):
            best = (float(tau), cov)
    tau, cov = best
    lost = np.array([r["B_correct"] and r["B_conf"] < tau for r in rows])
    return {"coverage_S": cov_s, "tau": tau, "coverage_threshold": cov, "tail_tax_threshold": _tt(rows, lost, tail),
            "loss_threshold": int(lost.sum())}


def decide(tables: dict, tts: dict, mc: dict, counts_ok: bool, gate: dict) -> dict:
    v = {"checks": {}, "notes": []}
    tail = gate["tail_buckets"]
    primary = gate["primary_variant"]
    t = tables[primary]
    for b in tail:
        n = t.get(b, {}).get("N", 0)
        acc = t.get(b, {}).get("acc_B") or 0.0
        v["checks"][f"V1_bucket_{b}_has_>={gate['min_facts_per_tail_bucket']}_facts"] = n >= gate["min_facts_per_tail_bucket"]
        v["checks"][f"V2_bucket_{b}_base_accuracy>={gate['min_base_acc_tail']}"] = acc >= gate["min_base_acc_tail"]
    v["checks"]["V3_tail_losses>=%d" % gate["min_tail_losses"]] = (tts[primary]["tail_losses"] or 0) >= gate["min_tail_losses"]
    v["checks"]["V4_counts_available"] = counts_ok
    if not counts_ok:
        v["verdict"] = "BLOCKED (exact counts missing)"
        return v
    if not all(v["checks"][k] for k in v["checks"] if k.startswith("V1")):
        v["verdict"] = "BLOCKED (too few tail facts: add EntityQuestions or TailCat facts and re-sample)"
        return v
    if not all(v["checks"][k] for k in v["checks"] if k.startswith("V2")):
        v["verdict"] = "FOLD (base model knows too little in the tail: nothing to lose; fold into MB5 as a section)"
        return v
    tp = tts[primary]["tail_tax"]
    ci = tts[primary]["ci"]
    others = [tts[k]["tail_tax"] for k in tts if k != primary and tts[k]["tail_tax"] is not None]
    if tp is not None and tp >= gate["go_tail_tax"] and ci and ci[0] > 1.0 and v["checks"]["V3_tail_losses>=%d" % gate["min_tail_losses"]]:
        v["verdict"] = "GO"
    elif (tp is None or tp < gate["fold_tail_tax"]) and all(o < gate["fold_tail_tax"] for o in others):
        v["verdict"] = "FOLD (tail tax below 1.2 for every variant: abstention is not tail-biased here)"
    else:
        v["verdict"] = "REVIEW (between thresholds: human decision, log it in notes/decisions.md)"
    m = mc.get(primary)
    if m and m.get("tail_tax_threshold") is not None and tp is not None:
        v["notes"].append(f"matched-coverage base threshold tail tax {m['tail_tax_threshold']:.2f} vs trained {tp:.2f}")
    return v


def render(res: dict) -> str:
    L = ["# Gate G0: does abstention take its losses from the long tail?", "",
         f"**Verdict: {res['decision']['verdict']}**", ""]
    for v, tab in res["tables"].items():
        tt = res["tail_tax"][v]
        L += [f"## {v}", "", f"tail tax **{tt['tail_tax']}** (95% CI {tt['ci']}), tail losses {tt['tail_losses']}", "",
              "| bucket | N | acc_B | acc_S | abstain_S | gain | loss_abs | loss_flip |", "|---|---|---|---|---|---|---|---|"]
        for b in ORDER + ["all"]:
            if b in tab:
                r = tab[b]
                L.append(f"| {b} | {r['N']} | {r['acc_B']:.3f} | {r['acc_S']:.3f} | {r['abstain_S']:.3f} | "
                         f"{r['gain']} | {r['loss_abs']} | {r['loss_flip']} |")
        mc = res["matched_coverage"].get(v)
        if mc:
            L.append(f"\nmatched coverage {mc['coverage_S']:.3f}: base threshold tau={mc['tau']:.2f}, "
                     f"tail tax {mc['tail_tax_threshold']}")
        L.append("")
    L += ["Validity checks:"] + [f"- [{'x' if ok else ' '}] {k}" for k, ok in res["decision"]["checks"].items()]
    L += [f"- note: {n}" for n in res["decision"]["notes"]]
    L += ["", "Thresholds: configs/g0.yaml (pre-registered). Signed: ____________________  date: __________"]
    return "\n".join(L)


def analyse(data: Path, results: Path, cfg: dict) -> dict:
    sample = read_jsonl(data / "g0_sample.jsonl")
    gens = []
    for role in ("base", "instruct"):
        p = results / role / "generations.jsonl"
        if p.exists():
            gens += read_jsonl(p)
    rows = grade(sample, gens)
    variants = sorted({k[:-8] for r in rows for k in r if k.endswith("_correct") and not k.startswith("B_")})
    tables = {v: bucket_table(rows, v) for v in variants}
    tail = tuple(cfg["gate"]["tail_buckets"])
    tts = {v: tail_tax(rows, v, tail) for v in variants}
    mc = {v: matched_coverage(rows, v, tail) for v in variants}
    counts_ok = all(f.get("c") is not None for f in sample)
    res = {"tables": tables, "tail_tax": tts, "matched_coverage": mc, "n_rows": len(rows)}
    res["decision"] = decide(tables, tts, mc, counts_ok, cfg["gate"]) if variants else {
        "verdict": "BLOCKED (no instruct generations)", "checks": {}, "notes": []}
    write_jsonl(results / "graded.jsonl", rows)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/g0")
    ap.add_argument("--results", default="results/G0")
    ap.add_argument("--config", default="configs/g0.yaml")
    ap.add_argument("--gate-out", default="results/gates")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = analyse(Path(args.data), Path(args.results), cfg)
    out = Path(args.gate_out)
    write_json(out / "G0.json", res)
    (out / "G0.md").write_text(render(res), encoding="utf-8")
    print(render(res))


if __name__ == "__main__":
    main()
