#!/usr/bin/env bash
# Run ON THE VM (after transfer_to_vm.sh). Unpacks and installs the training deps.
# Assumes a PyTorch Deep Learning VM image (CUDA torch already present in the base env).
#
#   bash ~/setup_vm.sh
set -euo pipefail
cd ~

echo "[1/3] unpacking..."
tar xzf dp_code.tar.gz        # -> ~/diffusion_policy
tar xzf dp_data.tar.gz        # -> ~/data   (the 12 *.hdf5)

echo "[2/3] installing training deps..."
# PyTorch DLVM images ship CUDA torch already. If the chosen image doesn't have torch
# (e.g. a common-cu* image), install a CUDA build now.
if ! python -c "import torch" 2>/dev/null; then
  echo "  torch not found -> installing CUDA torch..."
  pip install --quiet torch
fi
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
# Same pins as the local diffpol env. pytorch3d is intentionally NOT installed -- the repo's
# rotation_conversions_shim.py covers the only functions used, so no compile needed.
# huggingface_hub is pinned old because diffusers 0.11.1 needs HfFolder/cached_download.
pip install --quiet \
  "diffusers==0.11.1" "huggingface_hub==0.11.1" pandas \
  hydra-core omegaconf zarr numba einops dill wandb threadpoolctl h5py scipy tqdm

echo "[3/3] verifying the training import chain + policy build on CUDA..."
cd ~/diffusion_policy
python - <<'PY'
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)
import hydra, os, torch
from hydra import compose, initialize_config_dir
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("diffusion_policy/config")):
    cfg = compose(config_name="train_diffusion_unet_lowdim_workspace",
                  overrides=["task=peg_insertion_lowdim", "training.device=cuda"])
cls = hydra.utils.get_class(cfg._target_)
pol = hydra.utils.instantiate(cfg.policy).to("cuda")
print("OK:", cls.__name__, "| obs_dim", pol.obs_dim, "| device cuda:", next(pol.parameters()).is_cuda)
PY
echo
echo "setup complete. Start training in a tmux session so it survives disconnects:"
echo "  tmux new -s train"
echo "  bash ~/train_all.sh            # or: NUM_EPOCHS=800 bash ~/train_all.sh"
