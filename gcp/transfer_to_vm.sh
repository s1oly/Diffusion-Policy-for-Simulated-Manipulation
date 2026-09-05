#!/usr/bin/env bash
# Run LOCALLY (on the M2). Packs the diffusion_policy code (with our edits) + the 12
# datasets and copies them, plus the VM-side scripts, to the GCP VM.
#
#   bash gcp/transfer_to_vm.sh <VM_NAME> <ZONE>
#
# Excludes outputs/ (the 1GB crashed checkpoint), .git, wandb, egg-info -- so this ships
# ~18MB of code + ~540MB of data, not 1GB of stale checkpoints.
set -euo pipefail

VM="${1:?usage: bash gcp/transfer_to_vm.sh VM_NAME ZONE}"
ZONE="${2:?usage: bash gcp/transfer_to_vm.sh VM_NAME ZONE}"

ROOT="/Users/shuban_iyer/Diff_Policy_Proj"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$(mktemp -d)"

echo "[1/3] packing repo code (excluding outputs/.git/wandb/egg-info)..."
tar czf "$STAGE/dp_code.tar.gz" -C "$ROOT" \
  --exclude='diffusion_policy/outputs' \
  --exclude='diffusion_policy/.git' \
  --exclude='diffusion_policy/wandb' \
  --exclude='diffusion_policy/*.egg-info' \
  --exclude='*.pyc' --exclude='__pycache__' \
  diffusion_policy

echo "[2/3] packing datasets (proj/data/*.hdf5)..."
tar czf "$STAGE/dp_data.tar.gz" -C "$ROOT/proj" data

echo "[3/3] copying to $VM ($ZONE)..."
gcloud compute scp --zone "$ZONE" --recurse \
  "$STAGE/dp_code.tar.gz" "$STAGE/dp_data.tar.gz" \
  "$HERE/setup_vm.sh" "$HERE/train_all.sh" \
  "$VM:~/"

rm -rf "$STAGE"
echo
echo "done. Now SSH in and run setup:"
echo "  gcloud compute ssh $VM --zone $ZONE"
echo "  bash ~/setup_vm.sh"
