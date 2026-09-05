# GCP training + eval runbook

Train **and** eval the 12 ablation cells on a GCP L4 GPU (fast, no MPS bugs). Eval was moved to
the GPU too: on the M2 the diffusion sampler was ~96% of eval wall time (a 100-step DDPM chain at
~3.7s per prediction on MPS), so the full 12x50 rollout sweep took ~2 days; on the L4 it's a few
hours. Eval needs no headless-GL setup because `make_env` uses
`has_renderer=has_offscreen_renderer=use_camera_obs=False` -- pure MuJoCo physics, no OpenGL
context (the only extra deps are `libgl1`/`libglib` for the `cv2` module robosuite imports at load).

## Prerequisites: point gcloud at the project (once)

```bash
gcloud config set project diffpolproject
gcloud config set compute/zone us-central1-a
gcloud services enable compute.googleapis.com
```

After this, every command below (and the transfer/fetch scripts) uses `diffpolproject` by default.

## 0. One-time: provision an L4 VM

Uses a PyTorch Deep Learning VM image (CUDA + torch preinstalled). First boot installs the NVIDIA
driver (a few minutes). Add `--provisioning-model=SPOT` for a cheaper preemptible VM (see caveat
at the bottom).

**Why L4 and not T4:** T4 is chronically out of stock on GCP — every zone returns "does not have
enough resources available to fulfill the request" (aka `ZONE_RESOURCE_POOL_EXHAUSTED`), which is a
hardware shortage, not a quota problem. The L4 (in the `g2` machine family) is newer, in stock, and
better for this job (24GB vs T4's 16GB, faster). Same image + driver, so nothing downstream changes.
T4 is kept below as a fallback.

**0a. (optional) Confirm the current image family.** Google versions and retires these over time
(the old `pytorch-latest-gpu` alias is gone). As of 2026-09 the current PyTorch families are
`pytorch-2-9-cu129-ubuntu-2204-nvidia-580` (used below) and the `ubuntu-2404` variant. To re-check:

```bash
gcloud compute images list --project deeplearning-platform-release \
  --filter="family~pytorch" --format="value(family)" | sort -u
```

**0b. (optional) Confirm you have L4 quota in the region.** GPU quota is per-region. As of 2026-09,
`diffpolproject` has `NVIDIA_L4_GPUS` limit 1 in us-central1 (and T4 limit 1 there too — the T4
failures are stockout, not quota). To check a region:

```bash
gcloud compute regions describe us-central1 --project=diffpolproject \
  --flatten="quotas[]" --format="table(quotas.metric,quotas.limit,quotas.usage)" | grep L4_GPUS
```

If a region shows limit `0`, request a bump (Console → IAM & Admin → Quotas → filter
`NVIDIA_L4_GPUS`, pick the region → Edit → 1); usually auto-approved in minutes.

**0c. Create the VM.** `g2-standard-8` bundles 1× L4 — no `--accelerator` flag needed. Sweep the
us-central1 zones (where L4 quota is confirmed) and take the first with capacity — then use that same
zone for steps 1 and 4:

```bash
for ZONE in us-central1-a us-central1-b us-central1-c; do
  echo "trying $ZONE..."
  gcloud compute instances create dp-train \
    --project=diffpolproject --zone="$ZONE" \
    --machine-type=g2-standard-8 \
    --maintenance-policy=TERMINATE \
    --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB --metadata="install-nvidia-driver=True" \
    && { echo "CREATED in $ZONE"; break; }
done
```

200GB boot disk: silences the <200GB I/O-performance warning and leaves room for the checkpoints
(topk keeps 5 per run at ~1GB each). If `setup_vm.sh` finds no torch (e.g. you picked a `common-cu*`
image instead of a `pytorch-*` one), it pip-installs a CUDA torch build automatically.

**T4 fallback.** If you'd rather use a T4 (16GB, cheaper) and can find stock, swap the machine type
back to `n1-standard-8` and re-add the accelerator flag. Widen the zone sweep and add SPOT, since T4
stock is thin:

```bash
for ZONE in us-central1-b us-central1-c us-central1-f us-east1-c us-east1-d us-west1-b \
            us-west4-a europe-west4-b asia-east1-a; do
  echo "trying $ZONE..."
  gcloud compute instances create dp-train \
    --project=diffpolproject --zone="$ZONE" \
    --machine-type=n1-standard-8 --accelerator=type=nvidia-tesla-t4,count=1 \
    --maintenance-policy=TERMINATE --provisioning-model=SPOT \
    --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB --metadata="install-nvidia-driver=True" \
    && { echo "CREATED in $ZONE"; break; }
done
```

## 1. Transfer code + data (LOCAL)

```bash
bash gcp/transfer_to_vm.sh dp-train us-central1-a
```

## 2. Set up the env (ON VM)

```bash
gcloud compute ssh dp-train --zone us-central1-a
bash ~/setup_vm.sh
```

Verifies torch sees CUDA and the policy builds on GPU before you commit to a run.

## 3. Train all 12 (ON VM, in tmux)

```bash
tmux new -s train        # so it survives SSH disconnects
bash ~/train_all.sh      # or: NUM_EPOCHS=800 bash ~/train_all.sh
# detach with Ctrl-b d ; reattach later with: tmux attach -t train
```

`num_epochs` is the same for all 12 (fair ablation). Small-N runs finish fast; the batch is
dominated by the `*_100` / `*_200` runs.

## 4. Fetch checkpoints (LOCAL)

```bash
bash gcp/fetch_checkpoints.sh dp-train us-central1-a
```

Pulls each `latest.ckpt` into `diffusion_policy/outputs/<run>/checkpoints/`.

## 5. Eval all 12 on the GPU (ON VM)

The checkpoints are already on the VM from training (`~/diffusion_policy/outputs/<run>/`), so eval
reuses them in place -- no need to push them back up. `eval.py` builds its own env and reads only
the label + obs_keys from each checkpoint's cfg, so the VM's baked-in dataset paths don't matter.

**5a. Ship the eval code + VM scripts (LOCAL).** `peg_insertion/` carries the custom env +
controllers; the two scripts run on the VM.

```bash
cd /Users/shuban_iyer/Diff_Policy_Proj/proj
gcloud compute scp --recurse peg_insertion gcp/setup_eval_vm.sh gcp/eval_all.sh \
  dp-train:~/ --zone us-central1-a
```

**5b. Install eval deps + smoke-test (ON VM).** Installs `libgl1`/`libglib` (for the `cv2` robosuite
imports at load) then `robosuite==1.5.2` + `mujoco==3.8.1` (pinned to the local `diffpol` env so the
custom `Manipulation_Enviroment` registers identically), and verifies the env builds headless and a
checkpoint loads on CUDA.

```bash
gcloud compute ssh dp-train --zone us-central1-a
bash ~/setup_eval_vm.sh
```

**5c. Run the sweep in tmux (ON VM).** All 12 on CUDA, `--device cuda`, 50 rollouts each; writes
`~/diffusion_policy/eval_results/<run>.json` as each finishes.

```bash
tmux new -s eval
bash ~/eval_all.sh          # or: N_ROLLOUTS=100 bash ~/eval_all.sh
# detach with Ctrl-b d ; progress = `ls ~/diffusion_policy/eval_results/` (fills 1 file per run)
```

**5d. Fetch results to the M2 (LOCAL).**

```bash
bash gcp/fetch_eval.sh dp-train us-central1-a     # -> proj/eval_results/<run>.json
```

**Fallback -- eval locally on the M2** (slow: ~2 days for all 12; fine for a single run):

```bash
cd /Users/shuban_iyer/Diff_Policy_Proj/proj
python peg_insertion/training/eval.py \
  --checkpoint ../diffusion_policy/outputs/pd_100/checkpoints/latest.ckpt --n-rollouts 50
```

## 6. Shut down the VM

Do this once both the checkpoints (step 4) and the eval JSONs (step 5d) are safely on the M2 --
then delete, since everything you need is local.

```bash
gcloud compute instances delete dp-train --project=diffpolproject --zone us-central1-a
# or, to keep the disk for a later re-run (still bills for the disk):
gcloud compute instances stop dp-train --project=diffpolproject --zone us-central1-a
```

## 7. Analyze the ablation (LOCAL)

Reads the 12 `eval_results/*.json` into the 3 (controller) x 4 (N) success-rate table, a
success-vs-N curve (`peg_insertion/analysis/ablation.png`), and a spawn-yaw breakdown.

```bash
python peg_insertion/analysis/analyze_ablation.py
```

---

**Preemptible/SPOT caveat:** a SPOT VM can be reclaimed mid-run. `train_all.sh` runs sequentially,
so a reclaim loses only the in-progress run. If you use SPOT, re-running `train_all.sh` restarts the
whole loop from scratch (it does not auto-resume) -- fine for the fast small-N runs, but for the
`*_200` runs an on-demand (non-SPOT) VM is safer. L4 on-demand is already cheap.

**Cost ballpark:** `g2-standard-8` (1× L4) is ~$0.70–0.85/hr on-demand, less on SPOT. (T4 fallback
on `n1-standard-8` is ~$0.35/hr preemptible, ~$0.60/hr on-demand.) The full 12-run batch is a small
number of GPU-hours either way.
