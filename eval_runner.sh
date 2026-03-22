#!/usr/bin/env bash
# eval_runner.sh — Run the full Phase 3 eval pipeline in order.
#
# Prerequisites (must be completed before running this script):
#   python run_simulation.py   # produces results/federated_adapter.npz
#                              # and results/round_metrics.csv
#
# Usage:
#   chmod +x eval_runner.sh
#   bash eval_runner.sh

set -e

echo "=== Phase 3 eval ==="

echo ""
echo "--- Step 1: Train local adapters (compute-budget-equivalent baseline) ---"
python train_local.py

echo ""
echo "--- Step 2: Evaluate all conditions on the shared test set ---"
python eval.py

echo ""
echo "--- Step 3: Generate accuracy and loss-curve plots ---"
python plot.py

echo ""
echo "=== Results written to results/ ==="
