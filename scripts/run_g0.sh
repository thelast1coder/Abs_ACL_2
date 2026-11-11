#!/usr/bin/env bash
# Gate G0 for ABS. Run from the repository root.
#   bash scripts/run_g0.sh laptop        # tests, PopQA download, exact counts, G0 sample (network, no GPU)
#   SMOKE=1 bash scripts/run_g0.sh gpu   # Qwen2.5-0.5B pair on 40 facts (DEVICE=cpu works)
#   bash scripts/run_g0.sh gpu           # OLMo-2-7B base and instruct (about 3 GPU-hours)
#   bash scripts/run_g0.sh analyse       # -> results/gates/G0.md
set -euo pipefail
export PYTHONPATH=src
STAGE=${1:-laptop}
DATA=${DATA:-data/g0}
DEVICE=${DEVICE:-cuda}

case "$STAGE" in
  laptop)
    python -m abst.check_env
    python -m pytest -q tests
    [ -f "$DATA/facts.jsonl" ] || python -m abst.facts --out "$DATA"
    python -m abst.counts --data "$DATA" --config configs/g0.yaml
    python -m abst.sample --data "$DATA" --config configs/g0.yaml
    ;;
  gpu)
    python -m abst.check_env
    python -m pytest -q tests
    if [ "${SMOKE:-0}" = "1" ]; then
      python -m abst.run_g0 --role base --model Qwen/Qwen2.5-0.5B --data "$DATA" --out results/smoke --limit 40 --n-samples 3 --device "$DEVICE" --batch-size 8
      python -m abst.run_g0 --role instruct --model Qwen/Qwen2.5-0.5B-Instruct --data "$DATA" --out results/smoke --limit 40 --device "$DEVICE" --batch-size 8
      python -m abst.analysis --data "$DATA" --results results/smoke --gate-out results/smoke_gate || true
      echo "Smoke: the run must complete; read 20 instruct responses by eye to check the abstention regex."
      exit 0
    fi
    python -m abst.run_g0 --role base --model allenai/OLMo-2-1124-7B --data "$DATA" --out results/G0 --device "$DEVICE"
    python -m abst.run_g0 --role instruct --model allenai/OLMo-2-1124-7B-Instruct --data "$DATA" --out results/G0 --device "$DEVICE"
    ;;
  analyse)
    python -m abst.analysis --data "$DATA" --results results/G0 --config configs/g0.yaml
    ;;
  *) echo "usage: bash scripts/run_g0.sh [laptop|gpu|analyse]"; exit 2 ;;
esac
