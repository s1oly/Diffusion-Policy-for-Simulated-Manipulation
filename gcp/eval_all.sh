#!/usr/bin/env bash
# Run ON THE VM (inside tmux), after setup_eval_vm.sh. Evaluates all 12 trained
# checkpoints on CUDA -- the L4 makes the diffusion sampler ~20-40x faster than the
# M2's MPS, where inference was 96% of eval wall time (3.7s per 100-step prediction).
# Writes eval_results/<run>.json for each; fetch them back with fetch_eval.sh.
#
#   bash ~/eval_all.sh                 # 50 rollouts each (default)
#   N_ROLLOUTS=100 bash ~/eval_all.sh  # override
set -euo pipefail

# Make `import diffusion_policy` resolve on the VM (eval.py's own repo-locate assumes
# the local Diff_Policy_Proj/{proj,diffusion_policy} layout, which differs here).
export PYTHONPATH="$HOME/diffusion_policy:${PYTHONPATH:-}"
cd ~/diffusion_policy   # eval.py writes eval_results/<stem>.json relative to cwd

N_ROLLOUTS="${N_ROLLOUTS:-50}"
RUNS=(pd_10 pd_50 pd_100 pd_200
      noisy_10 noisy_50 noisy_100 noisy_200
      hybrid_10 hybrid_50 hybrid_100 hybrid_200)

for run in "${RUNS[@]}"; do
  ckpt="$HOME/diffusion_policy/outputs/$run/checkpoints/latest.ckpt"
  if [ ! -f "$ckpt" ]; then echo "SKIP $run (no checkpoint at $ckpt)"; continue; fi
  echo "===================  evaluating $run  (n=$N_ROLLOUTS)  ==================="
  python ~/peg_insertion/training/eval.py \
    --checkpoint "$ckpt" \
    --n-rollouts "$N_ROLLOUTS" \
    --device cuda
done

echo
echo "all evals complete. results in ~/diffusion_policy/eval_results/*.json"
echo "fetch them to the M2 with:  bash gcp/fetch_eval.sh dp-train <ZONE>"
