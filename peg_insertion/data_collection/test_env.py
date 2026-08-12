"""
Sanity checks for Manipulation_Enviroment (Phase 2 gate).

Run:  python peg_insertion/data_collection/test_env.py
      python peg_insertion/data_collection/test_env.py --cameras   (also checks offscreen rendering)
      mjpython peg_insertion/data_collection/test_env.py --render  (opens the viewer; macOS needs mjpython)

--render exists to eyeball what the numeric checks cannot: place_nut() sets qpos and calls
sim.forward(), which recomputes kinematics but does not integrate, so it will happily leave
the nut interpenetrating the peg. Some swept poses are geometrically unreachable. The
reward only reads body positions so the monotonicity results still hold -- but they
describe the reward function over a pose grid, not over physically reachable states.

Checks, in order:
  1. env registers and suite.make() finds it
  2. reset() returns the expected observation keys
  3. staged_rewards() returns 6 stages, each within its cap
  4. insert/seat increase monotonically as the nut descends the peg
  5. insert/seat collapse when the nut is off-axis in XY
  6. reward() is nonzero while shaping, and exactly 1.0 at success
  7. a manually stepped episode produces nonzero shaped rewards
"""

import argparse
import os
import sys
import time

import numpy as np

# envs/ has no __init__.py, so put it on the path directly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs"))

import robosuite as suite
from robosuite.environments.base import register_env

from Manipulation_Enviroment import Manipulation_Enviroment

register_env(Manipulation_Enviroment)

ENV_NAME = Manipulation_Enviroment.__name__  # register_env keys off the class name
STAGE_NAMES = ["reach", "grasp", "lift", "hover", "insert", "seat"]
STAGE_CAPS = np.array([0.1, 0.35, 0.5, 0.7,
                       Manipulation_Enviroment.INSERT_MULT,
                       Manipulation_Enviroment.SEAT_MULT])

# Nut body origin sits at mid-thickness (geom half-height 0.01), so a nut resting
# flat on the table has z = table_offset[2] + 0.01.
SEATED_Z = 0.83

_failures = []
_render = False


def check(label, passed, detail=""):
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not passed:
        _failures.append(label)


def draw(env, dwell=0.0):
    """Hold the viewer on the current sim state for `dwell` seconds. No-op without --render.

    Two robosuite quirks make this less obvious than env.render():

    1. MjviewerRenderer.render() is literally `pass` (renderers/viewer/mjviewer_renderer.py:20).
       The method that launches and syncs the passive viewer is update(), which the env only
       calls from step(). Checks that teleport the nut never step, so they have to drive
       update() directly or nothing is ever drawn.
    2. launch_passive does not run its own event loop, so sleeping through a dwell leaves the
       window unresponsive. Syncing at ~60Hz for the duration keeps it live and draggable.
    """
    if not _render:
        return
    deadline = time.time() + dwell
    while True:
        if env.viewer is not None:
            env.viewer.update()
        else:
            env.render()
        if time.time() >= deadline:
            return
        time.sleep(1.0 / 60.0)


def make_env(use_cameras=False, render=False):
    return suite.make(
        ENV_NAME,
        robots="Panda",
        has_renderer=render,
        has_offscreen_renderer=use_cameras,
        use_camera_obs=use_cameras,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        render_camera="frontview",
        control_freq=20,
        horizon=200,
    )


def place_nut(env, dxy, z):
    """Teleport the square nut to (peg_xy + dxy, z) and refresh derived sim state.

    sim.forward() recomputes kinematics but does not integrate, so the nut is placed
    exactly where asked even if that intersects the peg. See the module docstring.
    """
    peg = np.array(env.sim.data.body_xpos[env.peg1_body_id])
    qpos = np.concatenate([[peg[0] + dxy, peg[1], z], [1.0, 0.0, 0.0, 0.0]])
    env.sim.data.set_joint_qpos(env.nuts[0].joints[0], qpos)
    env.sim.forward()


def test_make_and_reset(env):
    print("\n1-2. registration + reset")
    obs = env.reset()
    draw(env, 1.0)
    check("suite.make + reset", True, f"({ENV_NAME})")
    check("reward_shaping defaults True", env.reward_shaping is True)
    check("object observables present",
          any("SquareNut" in k for k in obs), f"{len([k for k in obs if 'SquareNut' in k])} nut keys")
    if env.use_camera_obs:
        img = obs["agentview_image"]
        check("agentview_image is (84, 84, 3)", img.shape == (84, 84, 3), str(img.shape))


def test_stage_structure(env):
    print("\n3. stage structure")
    env.reset()
    stages = env.staged_rewards()
    check("staged_rewards returns 6 values", len(stages) == 6, str(len(stages)))
    check("all stages finite", bool(np.all(np.isfinite(stages))))

    # Sweep a wide range of nut poses and confirm nothing ever exceeds its cap.
    # Under --render this is an unwatchable blur, so cut it right down.
    n_poses = 50 if _render else 2000
    rng = np.random.default_rng(0)
    worst = np.zeros(6)
    for _ in range(n_poses):
        place_nut(env, rng.uniform(-0.30, 0.30), rng.uniform(0.80, 1.05))
        worst = np.maximum(worst, env.staged_rewards())
        draw(env, 0.02)
    over = [n for n, w, c in zip(STAGE_NAMES, worst, STAGE_CAPS) if w > c + 1e-9]
    check(f"no stage exceeds its cap ({n_poses} poses)", not over, f"max={np.round(worst, 3)}")


def test_descent_monotonic(env):
    print("\n4. insert/seat rise as the nut descends the peg")
    env.reset()
    heights = [0.95, 0.92, 0.89, 0.86, SEATED_Z]
    inserts, seats = [], []
    print(f"       {'z':>6}{'hover':>9}{'insert':>9}{'seat':>9}")
    for z in heights:
        place_nut(env, 0.0, z)
        draw(env, 1.5)   # slow: this is the pose the numeric checks cannot validate
        s = env.staged_rewards()
        inserts.append(s[4])
        seats.append(s[5])
        print(f"       {z:6.2f}{s[3]:9.3f}{s[4]:9.3f}{s[5]:9.3f}")
    check("insert is monotonically increasing", bool(np.all(np.diff(inserts) > 0)))
    check("seat is monotonically increasing", bool(np.all(np.diff(seats) > 0)))
    check("seat >= insert at every height", bool(np.all(np.array(seats) >= np.array(inserts))))


def test_xy_selectivity(env):
    print("\n5. insert/seat collapse when off-axis in XY")
    env.reset()
    print(f"       {'dxy':>6}{'insert':>9}{'seat':>9}")
    vals = []
    for dxy in [0.0, 0.02, 0.05, 0.10]:
        place_nut(env, dxy, SEATED_Z)
        draw(env, 1.5)
        s = env.staged_rewards()
        vals.append((s[3], s[4], s[5]))
        print(f"       {dxy:6.2f}{s[4]:9.3f}{s[5]:9.3f}")
    inserts = [v[1] for v in vals]
    check("insert decreases with XY error", bool(np.all(np.diff(inserts) < 0)))
    hover_far, insert_far, _ = vals[-1]
    check("at 10cm off, insert adds ~nothing over hover",
          insert_far - hover_far < 0.01, f"delta={insert_far - hover_far:.4f}")


def test_reward_scalar(env):
    print("\n6. reward() scalar behaviour")
    env.reset()
    r_reset = env.reward()
    check("reward at reset is nonzero", r_reset > 0.0, f"{r_reset:.4f}")

    # Best reachable shaped (unsolved) state: nut seated, but _check_success not yet run.
    place_nut(env, 0.0, SEATED_Z)
    draw(env, 1.5)
    best_staged = max(env.staged_rewards())
    check("best staged <= SEAT_MULT", best_staged <= Manipulation_Enviroment.SEAT_MULT + 1e-9,
          f"{best_staged:.4f}")

    r_success = env.reward()
    check("reward at success is exactly 1.0", abs(r_success - 1.0) < 1e-9, f"{r_success:.4f}")
    check("_check_success is True", bool(env._check_success()))
    check("staged collapse once on peg (sparse/staged are mutually exclusive)",
          max(env.staged_rewards()) < 1e-6)


def test_stepped_episode(env):
    print("\n7. manually stepped episode")
    env.reset()
    rewards = []
    for _ in range(40):
        action = np.zeros(env.action_dim)
        action[2] = -0.3          # drift the arm down so the reward actually moves
        _, reward, _, _ = env.step(action)
        draw(env, 0.05)   # real physics here; slowed just enough to follow
        rewards.append(reward)
    rewards = np.array(rewards)
    check("all rewards finite", bool(np.all(np.isfinite(rewards))))
    check("shaped rewards are nonzero", bool(np.any(rewards > 0.0)),
          f"min={rewards.min():.4f} max={rewards.max():.4f}")
    check("rewards stay within [0, 1]", bool(np.all((rewards >= 0.0) & (rewards <= 1.0))))


def main():
    global _render

    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", action="store_true",
                        help="also verify offscreen rendering (slower)")
    parser.add_argument("--render", action="store_true",
                        help="open the interactive viewer; macOS requires mjpython")
    args = parser.parse_args()
    _render = args.render

    if _render and not os.path.basename(sys.executable).startswith("mjpython"):
        print("NOTE: --render on macOS needs mjpython, not python.\n"
              f"      current interpreter: {sys.executable}\n"
              "      try: mjpython peg_insertion/data_collection/test_env.py --render\n")

    env = make_env(use_cameras=args.cameras, render=args.render)
    try:
        test_make_and_reset(env)
        test_stage_structure(env)
        test_descent_monotonic(env)
        test_xy_selectivity(env)
        test_reward_scalar(env)
        test_stepped_episode(env)
        if _render:
            print("\nviewer: holding 30s so the final state stays up (Ctrl-C to exit early)")
            try:
                draw(env, 30.0)
            except KeyboardInterrupt:
                print("  closed early")
    finally:
        env.close()

    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED:")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed - Phase 2 complete.")


if __name__ == "__main__":
    main()
