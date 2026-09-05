"""
Watch ONE trained diffusion-policy rollout in the on-screen MuJoCo viewer, for a given seed.

macOS: launch with mjpython (the on-screen viewer needs it), inside the diffpol env:

    mjpython peg_insertion/Tests/visualize_policy.py \
        --checkpoint ../diffusion_policy/outputs/pd_100/checkpoints/latest.ckpt --seed 42

The policy-side twin of visualize_rollout.py: instead of a classical controller it loads a
trained checkpoint and rolls it out with has_renderer=True. Uses eval.py's load_policy /
obs_vector and the SAME seed_placement, so --seed picks the identical scene the classical
viewer (and eval) would see -- run visualize_rollout.py --controller pd --seed 42 and this
with the matching pd checkpoint + --seed 42 to compare demonstrator vs learned policy.

NOTE: on MPS the diffusion sampler is slow (~seconds per action chunk at 100 steps), so the
viewer pauses briefly between chunks while it plans, then plays the chunk. That's expected;
for smoother playback drop the policy's num_inference_steps, or run on --device cuda/cpu.
"""
import argparse
import os
import sys
from collections import deque

import numpy as np
import torch

# eval.py (load_policy, obs_vector) lives in ../training; collect_demos (ENV_NAME,
# seed_placement, spawn_yaw_deg) in ../data_collection. eval.py's own module-level path
# setup (data_collection + the diffusion_policy repo) keys off its __file__, so importing
# it here resolves those regardless of where this file sits.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
sys.path.insert(0, os.path.join(HERE, "..", "data_collection"))

import robosuite as suite  # noqa: E402
from collect_demos import ENV_NAME, seed_placement, spawn_yaw_deg  # noqa: E402
from eval import load_policy, obs_vector  # noqa: E402


def make_render_env(horizon):
    """Same kwargs as collect_demos.make_env, but on-screen instead of headless."""
    return suite.make(
        ENV_NAME,
        robots="Panda",
        has_renderer=True,          # <- show a window
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=horizon,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="path to a .ckpt from a training run, e.g. "
                        "../diffusion_policy/outputs/pd_100/checkpoints/latest.ckpt")
    p.add_argument("--seed", type=int, default=None,
                   help="placement seed; same seed -> same scene as the classical viewer "
                        "and as eval. Omit for a random scene.")
    p.add_argument("--horizon", type=int, default=600, help="max steps")
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    args = p.parse_args()

    device = torch.device(args.device)
    policy, cfg = load_policy(args.checkpoint, device)
    obs_keys = list(cfg.task.obs_keys)
    stem = os.path.splitext(os.path.basename(cfg.task.dataset_path))[0]

    seed = args.seed if args.seed is not None else int(np.random.randint(1_000_000))
    print(f"model={stem}  seed={seed}  device={args.device}  horizon={args.horizon}")

    env = make_render_env(args.horizon)

    # Mirror eval.run_rollout: reseed placement -> reset -> force a real first obs -> warm
    # the To-step history by repeating that first observation.
    seed_placement(env, seed)
    env.reset()
    obs = env._get_observations(force_update=True)
    policy.reset()
    yaw = spawn_yaw_deg(obs)
    print(f"spawn yaw = {yaw:+.1f} deg\n(close the viewer window to exit)")

    To = policy.n_obs_steps
    hist = deque([obs_vector(obs, obs_keys)] * To, maxlen=To)

    success = False
    steps = 0
    with torch.no_grad():
        while steps < args.horizon and not success:
            obs_seq = np.stack(hist, axis=0)[None]           # (1, To, Do)
            obs_t = torch.from_numpy(obs_seq).to(policy.device, dtype=policy.dtype)
            action = policy.predict_action({"obs": obs_t})["action"][0].cpu().numpy()
            for a in action:                                  # execute the plan open-loop
                obs, _, done, _ = env.step(a)
                env.render()
                steps += 1
                hist.append(obs_vector(obs, obs_keys))
                if env._check_success():
                    success = True
                if success or done:
                    break

    print(f"\n{'SUCCESS' if success else 'FAIL'} after {steps} steps  (yaw {yaw:+.1f} deg)")
    env.close()


if __name__ == "__main__":
    main()
