#!/usr/bin/env bash
# Run LOCALLY (on the M2) after eval_all.sh finishes on the VM. Pulls each run's
# eval_results/<run>.json back into the local proj/eval_results/ (same path eval.py
# writes to when run locally), so downstream analysis reads them either way.
#
#   bash gcp/fetch_eval.sh <VM_NAME> <ZONE>
set -euo pipefail

VM="${1:?usage: bash gcp/fetch_eval.sh VM_NAME ZONE}"
ZONE="${2:?usage: bash gcp/fetch_eval.sh VM_NAME ZONE}"

DEST="/Users/shuban_iyer/Diff_Policy_Proj/proj/eval_results"
mkdir -p "$DEST"
RUNS=(pd_10 pd_50 pd_100 pd_200
      noisy_10 noisy_50 noisy_100 noisy_200
      hybrid_10 hybrid_50 hybrid_100 hybrid_200)

for run in "${RUNS[@]}"; do
  echo "fetching $run.json..."
  gcloud compute scp --zone "$ZONE" \
    "$VM:~/diffusion_policy/eval_results/$run.json" \
    "$DEST/$run.json" || echo "  (skip $run -- not found on VM)"
done

echo
echo "done. local results in $DEST/"
