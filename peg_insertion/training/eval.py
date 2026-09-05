"""
Evaluate a trained diffusion policy on Manipulation_Enviroment by rolling it out.

    python peg_insertion/training/eval.py \
        --checkpoint /path/to/diffusion_policy/outputs/pd_100/checkpoints/latest.ckpt \
        --n-rollouts 50

Why this exists instead of diffusion_policy's built-in RobomimicLowdimRunner:
the built-in runner rebuilds the eval env from the dataset's `env_args` through
robomimic, which calls robosuite.make("Manipulation_Enviroment"). Our env is a custom
subclass robomimic's process has not registered, and the stored `env_args` omit
`controller_configs`. Rather than fight that, training uses a no-op runner and success is
measured here, reusing the exact register_env + suite.make path that collected the demos.

Two properties this eval is built to preserve:
  * UNFILTERED reset -> yaw is uniform over 360 deg, unlike the yaw-biased training set
    (only successful demos were kept). Eval success therefore sits BELOW collection
    success; the per-rollout yaw is recorded so that gap is measurable, not invisible.
  * PAIRED across models -> placement is reseeded per rollout from a fixed seed base, so
    all 12 ablation models are scored on the SAME initial states. Comparing cells is then
    a paired comparison, which is what the ablation actually claims.
"""

import argparse
import json
import os
import sys
from collections import deque

import numpy as np
import torch
import dill
import hydra

# Reuse the env construction that produced the demos, so eval runs on exactly the env the
# policy was trained against (same register_env, same suite.make kwargs, same obs repair).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_collection"))
from collect_demos import make_env, ENV_NAME, spawn_yaw_deg, seed_placement  # noqa: E402

# The diffusion_policy package runs from inside its own repo dir (cwd on sys.path) and its
# editable install is currently broken (pip metadata present, import path missing). Put the
# sibling repo on sys.path so eval works no matter where it is launched from -- same
# extend-sys.path convention the rest of this project uses (no __init__.py anywhere).
_DP_REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "diffusion_policy"))
if os.path.isdir(os.path.join(_DP_REPO, "diffusion_policy")) and _DP_REPO not in sys.path:
    sys.path.insert(0, _DP_REPO)

from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: E402


def load_policy(checkpoint, device):
    """Rebuild the policy from a training checkpoint (mirrors diffusion_policy/eval.py).

    The normalizer is part of the policy's state_dict, so obs/action scaling comes back
    with the weights -- eval must not recompute it from eval data. Returns the EMA policy
    when training used EMA (that is the one that was actually validated).
    """
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=os.path.dirname(checkpoint))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.to(device)
    policy.eval()
    return policy, cfg


def obs_vector(obs, obs_keys):
    """Concatenate the pinned low-dim keys into one vector, in obs_keys order.

    This order MUST match the task config's obs_keys (the same list the dataset and the
    normalizer used); a reorder here feeds the policy silently scrambled input. We read the
    order straight off the checkpoint's cfg to guarantee it matches what the model trained on.
    """
    return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in obs_keys])


@torch.no_grad()
def run_rollout(env, policy, obs_keys, max_steps, episode_seed):
    """One receding-horizon rollout. Returns (success, spawn_yaw_deg, n_steps)."""
    seed_placement(env, episode_seed)   # same initial state across all models for this index
    env.reset()
    # reset() leaves SquareNut_to_robot0_eef_* fabricated until the first step; force an
    # update so the first observation the policy conditions on is real (matches collection).
    obs = env._get_observations(force_update=True)
    policy.reset()

    yaw = spawn_yaw_deg(obs)
    To = policy.n_obs_steps
    # Warm-start the observation history by repeating the first real observation To times.
    hist = deque([obs_vector(obs, obs_keys)] * To, maxlen=To)

    success = False
    steps = 0
    while steps < max_steps and not success:
        obs_seq = np.stack(hist, axis=0)[None]           # (1, To, Do)
        obs_t = torch.from_numpy(obs_seq).to(policy.device, dtype=policy.dtype)
        action = policy.predict_action({"obs": obs_t})["action"][0].cpu().numpy()  # (Ta, 7)

        for a in action:                                  # execute the plan open-loop
            obs, _, done, _ = env.step(a)
            steps += 1
            hist.append(obs_vector(obs, obs_keys))
            if env._check_success():
                success = True
            if success or done:
                break

    return success, yaw, steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to a .ckpt from a training run")
    p.add_argument("--n-rollouts", type=int, default=50)
    p.add_argument("--horizon", type=int, default=600,
                   help="max env steps per rollout; also the env horizon")
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    p.add_argument("--seed-base", type=int, default=100_000,
                   help="placement seed for rollout i is seed_base+i; keep fixed so every "
                        "model is scored on the same initial states (paired eval)")
    p.add_argument("--out", default=None,
                   help="JSON output path (default: eval_results/<dataset stem>.json)")
    args = p.parse_args()

    device = torch.device(args.device)
    policy, cfg = load_policy(args.checkpoint, device)
    obs_keys = list(cfg.task.obs_keys)
    # Label this run by the dataset it trained on, e.g. "pd_100" -> analysis reads this.
    dataset_path = cfg.task.dataset_path
    stem = os.path.splitext(os.path.basename(dataset_path))[0]
    out = args.out or os.path.join("eval_results", f"{stem}.json")

    print(f"model      : {args.checkpoint}")
    print(f"trained on : {dataset_path}  ({stem})")
    print(f"obs_keys   : {obs_keys}")
    print(f"To/Ta      : {policy.n_obs_steps}/{policy.n_action_steps}   env: {ENV_NAME}\n")

    env = make_env(args.horizon)
    per_rollout = []
    n_success = 0
    for i in range(args.n_rollouts):
        success, yaw, steps = run_rollout(
            env, policy, obs_keys, args.horizon, args.seed_base + i)
        n_success += int(success)
        per_rollout.append({"seed": args.seed_base + i, "spawn_yaw": yaw,
                            "success": bool(success), "steps": steps})
        print(f"  rollout {i:3d}  yaw={yaw:+7.1f}  steps={steps:4d}  "
              f"{'SUCCESS' if success else 'fail'}   running={n_success}/{i + 1}")
    env.close()

    rate = n_success / args.n_rollouts
    se = (rate * (1 - rate) / args.n_rollouts) ** 0.5     # binomial standard error
    result = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "dataset": stem,
        "controller": stem.rsplit("_", 1)[0],
        "n_demos": int(stem.rsplit("_", 1)[1]) if stem.rsplit("_", 1)[-1].isdigit() else None,
        "n_rollouts": args.n_rollouts,
        "success_rate": rate,
        "success_se": se,
        "per_rollout": per_rollout,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nsuccess rate: {rate:.1%} +/- {se:.1%}  ({n_success}/{args.n_rollouts})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
