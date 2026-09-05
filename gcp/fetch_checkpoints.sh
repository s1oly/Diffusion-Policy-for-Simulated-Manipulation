#!/usr/bin/env bash
# Run LOCALLY (on the M2) after training finishes on the VM. Pulls each run's
# latest.ckpt back into the local diffusion_policy/outputs/<run>/checkpoints/ so
# eval.py can load it here (eval runs locally, where robosuite/MuJoCo already work).
#
#   bash gcp/fetch_checkpoints.sh <VM_NAME> <ZONE>
set -euo pipefail

VM="${1:?usage: bash gcp/fetch_checkpoints.sh VM_NAME ZONE}"
ZONE="${2:?usage: bash gcp/fetch_checkpoints.sh VM_NAME ZONE}"

DEST="/Users/shuban_iyer/Diff_Policy_Proj/diffusion_policy/outputs"
RUNS=(pd_10 pd_50 pd_100 pd_200
      noisy_10 noisy_50 noisy_100 noisy_200
      hybrid_10 hybrid_50 hybrid_100 hybrid_200)

for run in "${RUNS[@]}"; do
  mkdir -p "$DEST/$run/checkpoints"
  f="$DEST/$run/checkpoints/latest.ckpt"
  # skip anything already fully fetched (>900MB; a full ckpt is ~1GB) so a
  # re-run after a dropped connection only pulls the missing/partial ones.
  if [ -f "$f" ] && [ "$(stat -f%z "$f")" -gt 943718400 ]; then
    echo "skip $run (already have $(du -h "$f" | cut -f1))"
    continue
  fi
  echo "fetching $run/latest.ckpt..."
  gcloud compute scp --zone "$ZONE" \
    "$VM:~/diffusion_policy/outputs/$run/checkpoints/latest.ckpt" \
    "$f"
done

echo
echo "done. eval each locally, e.g.:"
echo "  python peg_insertion/training/eval.py --checkpoint $DEST/pd_100/checkpoints/latest.ckpt --n-rollouts 50"
