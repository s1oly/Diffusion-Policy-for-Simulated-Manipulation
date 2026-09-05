"""
Watch ONE classical-controller rollout in the on-screen MuJoCo viewer, for a given seed.

macOS: launch with mjpython (the on-screen viewer needs it), inside the diffpol env:

    mjpython peg_insertion/Tests/visualize_rollout.py --controller pd --seed 42
    mjpython peg_insertion/Tests/visualize_rollout.py --controller hybrid            # random scene

Same env + controller + seeding as collect_demos, but with has_renderer=True so you can
watch instead of record. seed_placement() makes the initial nut pose a pure function of
--seed, so the SAME --seed gives the SAME scene across controllers (pd vs noisy vs hybrid) --
run the three in turn with one seed to compare how each handles an identical setup.
"""
import argparse
import os
import sys

import numpy as np

# collect_demos wires ../envs and ../controllers onto sys.path and registers the custom
# Manipulation_Enviroment on import, so importing it makes suite.make(ENV_NAME) work.
# This file lives in peg_insertion/Tests/, so data_collection is one level up.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "data_collection"))

import robosuite as suite  # noqa: E402
from collect_demos import (  # noqa: E402
    ENV_NAME, build_controller, seed_placement, spawn_yaw_deg,
)


def make_render_env(horizon):
    """Same kwargs as collect_demos.make_env, but on-screen instead of headless."""
    return suite.make(
        ENV_NAME,
        robots="Panda",
        has_renderer=True,          # <- the one change from make_env: show a window
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=horizon,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controller", default="pd",
                   choices=["random", "pd", "hybrid", "pure_noise"])
    p.add_argument("--seed", type=int, default=None,
                   help="placement seed; same seed -> same scene across controllers. "
                        "Omit for a random scene.")
    p.add_argument("--horizon", type=int, default=600,
                   help="max steps (an FSM controller usually finishes well before this)")
    args = p.parse_args()

    seed = args.seed if args.seed is not None else int(np.random.randint(1_000_000))
    print(f"controller={args.controller}  seed={seed}  horizon={args.horizon}")

    env = make_render_env(args.horizon)
    controller = build_controller(args.controller, env)

    # Reseed the placement samplers BEFORE reset so this scene depends only on `seed`
    # (mirrors run_episode in collect_demos), then force a real first observation.
    seed_placement(env, seed)
    env.reset()
    obs = env._get_observations(force_update=True)
    controller.reset()
    print(f"spawn yaw = {spawn_yaw_deg(obs):+.1f} deg\n(close the viewer window to exit)")

    success = False
    done = False
    steps = 0
    while not done:
        action = np.asarray(controller.get_action(obs), dtype=np.float64)
        obs, _, done, _ = env.step(action)
        env.render()
        steps += 1
        if env._check_success():
            success = True
        # End when the controller's FSM reports done (same as collect_demos), not at horizon.
        if getattr(controller, "finished", False):
            done = True
        if steps >= args.horizon:
            done = True

    print(f"\n{'SUCCESS' if success else 'FAIL'} after {steps} steps  (yaw {spawn_yaw_deg(obs):+.1f} deg)")
    env.close()


if __name__ == "__main__":
    main()
