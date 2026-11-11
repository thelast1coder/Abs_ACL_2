import numpy as np

from abst.analysis import analyse, bucket_table, matched_coverage, tail_tax
from abst.common import read_jsonl, write_jsonl
from abst.counts import bucket_of, fact_query
from abst.facts import normalise_row
from abst.grading import is_abstention, is_correct, same_answer
from abst.prompts import base_prompt
from abst.run_g0 import run
from abst.sample import draw

GATE = {"tail_buckets": ["1", "2"], "primary_variant": "S1_plain", "min_facts_per_tail_bucket": 40,
        "min_base_acc_tail": 0.15, "min_tail_losses": 5, "go_tail_tax": 1.5, "fold_tail_tax": 1.2}


def test_grading():
    ans = ["New York City", "NYC"]
    assert is_correct("He was born in New York City.", ans)
    assert not is_correct("He was born in York.", ans)
    assert is_abstention("I don't know.", ans)
    assert is_abstention("", ans)
    assert not is_abstention("I'm not sure, but probably NYC.", ans)  # hedged but correct
    assert not is_abstention("London", ans)
    assert same_answer("Paris", "paris.") and same_answer("Paris", "Paris, France") and not same_answer("", "x")


def test_buckets_and_query():
    assert [bucket_of(x) for x in (0, 1, 2, 3, 5, 6, 20, 21, 100, 101, None)] == \
        ["0", "1", "2", "3-5", "3-5", "6-20", "6-20", "21-100", "21-100", ">100", None]
    f = normalise_row({"id": 7, "subj": "Rahul Sharma", "prop": "occupation", "obj": "cricketer",
                       "question": "What is Rahul Sharma's occupation?", "possible_answers": '["cricketer", "cricket player"]',
                       "s_aliases": "[]", "s_pop": 12})
    assert f["answers"] == ["cricketer", "cricket player"]
    assert fact_query(f) == "Rahul Sharma AND cricketer OR cricket player"
    assert base_prompt("Q?").endswith("Q: Q?\nA:")


def test_alias_filter():
    f = normalise_row({"id": 8, "subj": "X", "prop": "country", "obj": "Canada", "question": "In what country is X?",
                       "possible_answers": '["Canada", "can", "CA", "CAN", "\U0001F1E8\U0001F1E6", "Dominion of Canada"]'})
    assert f["answers"] == ["Canada", "Dominion of Canada"]
    assert "can" in f["answers_raw"]
    assert not is_correct("I can't answer that.", f["answers"])
    assert is_abstention("I can't answer that.", f["answers"])
    g = normalise_row({"id": 9, "subj": "Y", "prop": "color", "obj": "red", "question": "What colour is Y?",
                       "possible_answers": '["red", "us"]'})
    assert g["answers"] == ["red"]  # the canonical answer is kept even when short


def synthetic_rows(n=1400, seed=0, tail_bias=True):
    """Base knows facts with probability rising with exposure; S loses tail facts more if tail_bias."""
    rng = np.random.default_rng(seed)
    buckets = ["0", "1", "2", "3-5", "6-20", "21-100", ">100"]
    accs = [0.1, 0.25, 0.3, 0.4, 0.55, 0.7, 0.85]
    loss_p = [0.5, 0.6, 0.55, 0.25, 0.15, 0.08, 0.05] if tail_bias else [0.2] * 7
    rows = []
    for i in range(n):
        b = i % 7
        cb = rng.random() < accs[b]
        lost = cb and rng.random() < loss_p[b]
        gained = (not cb) and rng.random() < 0.05
        cs = (cb and not lost) or gained
        abst = (not cs) and rng.random() < 0.7
        conf = float(np.clip((0.8 if cb else 0.3) + rng.normal(0, 0.2), 0, 1))
        rows.append({"fact_id": str(i), "bucket": buckets[b], "c": b, "B_correct": bool(cb), "B_conf": conf,
                     "S1_plain_correct": bool(cs), "S1_plain_abstain": bool(abst)})
    return rows


def test_identity_and_tail_tax():
    rows = synthetic_rows()
    tab = bucket_table(rows, "S1_plain")
    for k, r in tab.items():
        assert r["C_S"] - r["C_B"] == r["gain"] - r["loss"]
    tt = tail_tax(rows, "S1_plain", n_boot=200)
    assert tt["tail_tax"] > 1.5 and tt["ci"][0] > 1.0
    tt0 = tail_tax(synthetic_rows(tail_bias=False), "S1_plain", n_boot=200)
    assert 0.7 < tt0["tail_tax"] < 1.3


def test_matched_coverage_threshold():
    rows = synthetic_rows()
    mc = matched_coverage(rows, "S1_plain")
    assert abs(mc["coverage_threshold"] - mc["coverage_S"]) < 0.05
    assert mc["tail_tax_threshold"] is not None


def test_draw_balances_and_excludes_fewshot():
    facts = [{"fact_id": str(i), "subj": "Osaka" if i == 0 else f"S{i}", "answers": ["x"]} for i in range(50)]
    counts = [{"fact_id": str(i), "c": i % 3, "bucket": bucket_of(i % 3)} for i in range(50)]
    s = draw(facts, counts, per_bucket=5)
    assert len(s) == 15 and all(f["subj"] != "Osaka" for f in s)


def test_end_to_end_tiny(tmp_path, tok, model):
    data = tmp_path / "data"
    facts = []
    for i in range(60):
        facts.append({"fact_id": f"popqa:{i}", "subj": f"Person{i}", "prop": "occupation", "question":
                      f"What is the occupation of Person{i}?", "answers": ["cricketer"], "c": i % 4,
                      "bucket": bucket_of(i % 4)})
    write_jsonl(data / "g0_sample.jsonl", facts)
    res_dir = tmp_path / "results"
    run(model, tok, "local/tiny", "base", data, res_dir, batch_size=8, n_samples=3, progress=False)
    run(model, tok, "local/tiny", "instruct", data, res_dir, batch_size=8, progress=False)
    gens = read_jsonl(res_dir / "base" / "generations.jsonl")
    assert len(gens) == 60 and len(gens[0]["samples"]) == 3
    res = analyse(data, res_dir, {"gate": GATE})
    assert set(res["tables"]) == {"S1_plain", "S2_abstain"}
    assert "verdict" in res["decision"]
