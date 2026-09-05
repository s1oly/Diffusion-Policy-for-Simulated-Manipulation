#!/usr/bin/env bash
# Run ON THE VM (inside tmux). Trains all 12 ablation cells on CUDA, sequentially.
# num_epochs is identical across runs so the ablation stays a fair comparison.
#
#   bash ~/train_all.sh                # default epochs
#   NUM_EPOCHS=800 bash ~/train_all.sh # override
set -euo pipefail
cd ~/diffusion_policy

DATA="$HOME/data"
NUM_EPOCHS="${NUM_EPOCHS:-500}"

# All 12 cells: 3 controllers x N in {10,50,100,200}. Small-N runs finish fast (fewer
# steps/epoch), so the batch is dominated by the four *_200 / *_100 runs.
RUNS=(pd_10 pd_50 pd_100 pd_200
      noisy_10 noisy_50 noisy_100 noisy_200
      hybrid_10 hybrid_50 hybrid_100 hybrid_200)

for run in "${RUNS[@]}"; do
  echo "===================  training $run  (epochs=$NUM_EPOCHS)  ==================="
  python train.py --config-name=train_diffusion_unet_lowdim_workspace \
    task=peg_insertion_lowdim \
    training.device=cuda \
    training.num_epochs="$NUM_EPOCHS" \
    logging.mode=disabled \
    task.dataset_path="$DATA/$run.hdf5" \
    hydra.run.dir="outputs/$run"
done

echo
echo "all 12 runs complete. checkpoints at ~/diffusion_policy/outputs/<run>/checkpoints/latest.ckpt"
echo "fetch them to the M2 with:  bash gcp/fetch_checkpoints.sh <VM_NAME> <ZONE>"
