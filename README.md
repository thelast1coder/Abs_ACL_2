# ABS: What Abstention Deletes (ACL 2027)

Anti-hallucination training silences rare knowledge. Exact bookkeeping on OLMo-2, with infini-gram counts.

- Start with `PROJECT.md`, then `CLAUDE.md`.

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m pytest -q tests
bash scripts/run_g0.sh laptop      # PopQA + exact counts + sample (network, no GPU)
SMOKE=1 bash scripts/run_g0.sh gpu # small real-model smoke (DEVICE=cpu works)
bash scripts/run_g0.sh gpu         # G0, about 3 GPU-hours
bash scripts/run_g0.sh analyse     # -> results/gates/G0.md
```
