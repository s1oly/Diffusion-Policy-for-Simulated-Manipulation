#!/usr/bin/env bash
# Run ON THE VM, after shipping the eval code:
#   gcloud compute scp --recurse peg_insertion gcp/setup_eval_vm.sh gcp/eval_all.sh dp-train:~/ --zone <ZONE>
#
# Installs the eval-only deps and smoke-tests that the custom robosuite env builds
# headless and a checkpoint loads on CUDA. No GL/rendering setup is needed: make_env
# uses has_renderer=has_offscreen_renderer=use_camera_obs=False, so MuJoCo never
# creates an OpenGL context -- it's pure physics.
#
#   bash ~/setup_eval_vm.sh
set -euo pipefail

# eval.py finds the diffusion_policy repo via a path relative to the LOCAL layout;
# on the VM we point PYTHONPATH at the repo so `import diffusion_policy` resolves.
export PYTHONPATH="$HOME/diffusion_policy:${PYTHONPATH:-}"

echo "[1/3] installing system libs cv2 needs (robosuite imports OpenCV at load time,
      even headless -- the opencv wheel needs libGL.so.1 / libglib)..."
sudo apt-get update -qq
sudo apt-get install -y libgl1 libglib2.0-0

echo "[2/3] installing robosuite + mujoco (pinned to the local diffpol env)..."
# Same pins as the local diffpol env so the custom Manipulation_Enviroment registers
# and behaves identically to how the demos were collected.
python -m pip install --quiet robosuite==1.5.2 mujoco==3.8.1

echo "[3/3] smoke test: import chain + custom env (headless) + policy on CUDA..."
cd ~/diffusion_policy
python - <<'PY'
import os, sys, torch
sys.path.insert(0, os.path.expanduser("~/peg_insertion/training"))
import eval as Ev  # triggers the collect_demos + robosuite + Manipulation_Enviroment imports
env = Ev.make_env(600)
print("  env built     :", type(env).__name__)
ckpt = os.path.expanduser("~/diffusion_policy/outputs/pd_10/checkpoints/latest.ckpt")
pol, cfg = Ev.load_policy(ckpt, torch.device("cuda"))
print("  policy on cuda:", next(pol.parameters()).is_cuda,
      "| num_inference_steps:", getattr(pol, "num_inference_steps", "?"),
      "| To/Ta:", pol.n_obs_steps, "/", pol.n_action_steps)
env.close()
print("  OK")
PY
echo
echo "smoke test passed. Run the full sweep in tmux so it survives disconnects:"
echo "  tmux new -s eval"
echo "  bash ~/eval_all.sh            # or: N_ROLLOUTS=100 bash ~/eval_all.sh"
