"""
Watch the controller run, or replay a recorded demo, in the live MuJoCo viewer.

macOS needs `mjpython` for an on-screen window, and mjpython only exists inside
the conda env -- activate `diffpol` first.

    mamba activate diffpol

    # drive the env live with the PD controller, printing phase transitions
    mjpython peg_insertion/data_collection/view_demo.py live --episodes 3

    # replay an episode out of a collected HDF5, exactly as recorded
    mjpython peg_insertion/data_collection/view_demo.py replay data/pd_10.hdf5 --demo 0

Replay is exact rather than approximate: `collect_demos.py` saves the flattened
MuJoCo state each step, so this restores physics state directly instead of
re-simulating the actions and accumulating divergence.

Headless variants (plain `python`, no window) for when you only want the numbers:

    python peg_insertion/data_collection/view_demo.py stats data/pd_10.hdf5
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "envs"))
sys.path.insert(0, os.path.join(HERE, "..", "controllers"))

import robosuite as suite
from robosuite.environments.base import register_env

from Manipulation_Enviroment import Manipulation_Enviroment

register_env(Manipulation_Enviroment)
ENV_NAME = Manipulation_Enviroment.__name__


def make_env(render, horizon):
    return suite.make(
        ENV_NAME,
        robots="Panda",
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=horizon,
        seed=0,
    )


def cmd_live(args):
    from pd_controller import PDController

    env = make_env(render=True, horizon=args.horizon)
    ctrl = PDController(env)

    for ep in range(args.episodes):
        env.reset()
        obs = env._get_observations(force_update=True)
        ctrl.reset()

        last, t = None, 0
        while not ctrl.finished and t < args.horizon:
            action = ctrl.get_action(obs)
            if ctrl.phase != last:
                print(f"  t={t:4d}  {ctrl.phase.value}")
                last = ctrl.phase
            obs, _, _, _ = env.step(action)
            env.render()
            t += 1

        print(f"episode {ep}: {ctrl.phase.value} after {t} steps, "
              f"success={bool(env._check_success())}, reward={env.reward():.3f}\n")
    env.close()


def cmd_replay(args):
    import h5py

    with h5py.File(args.path, "r") as f:
        key = f"demo_{args.demo}"
        if key not in f["data"]:
            raise SystemExit(f"{key} not in {args.path}; "
                             f"has {list(f['data'].keys())[:8]} ...")
        g = f["data"][key]
        states = g["states"][:]
        rewards = g["rewards"][:]
        ok = bool(g.attrs.get("successful", False))
        env_args = json.loads(f["data"].attrs["env_args"])

    horizon = env_args["env_kwargs"].get("horizon", len(states) + 10)
    env = make_env(render=True, horizon=horizon)
    env.reset()

    print(f"{key}: {len(states)} steps, successful={ok}")
    for t, s in enumerate(states):
        env.sim.set_state_from_flattened(s)
        env.sim.forward()
        env.render()
        if args.trace and t % 20 == 0:
            print(f"  t={t:4d}  reward={rewards[t]:.3f}")
    print(f"final reward {rewards[-1]:.3f}")
    env.close()


BINS = np.arange(-180.0, 181.0, 45.0)


def yaw_report(data, names):
    """Spawn-yaw coverage of the kept demos, and the per-bin success rate.

    Attempts are uniform in yaw by construction, so any non-uniformity in the
    kept set is the success filter talking. This matters because Phase 4
    evaluates from an unfiltered env.reset() that IS uniform -- so wherever a bin
    is underfilled here, the policy meets orientations at eval that it barely saw
    in training. Report this next to the ablation table; the gap it creates is
    not a property of diffusion policy.
    """
    kept = np.array([data[n].attrs["spawn_yaw"] for n in names
                     if "spawn_yaw" in data[n].attrs], dtype=float)
    if kept.size == 0:
        print("\n  (no spawn_yaw recorded -- collected before yaw logging was added)")
        return

    a_yaw = np.asarray(data.attrs.get("attempt_yaws", []), dtype=float)
    a_ok = np.asarray(data.attrs.get("attempt_success", []), dtype=int)

    kept_h, _ = np.histogram(kept, bins=BINS)
    peak = max(kept_h.max(), 1)

    print(f"\n  spawn-yaw coverage of kept demos "
          f"(uniform would be ~{kept.size / (len(BINS) - 1):.1f} per bin)")
    for i in range(len(BINS) - 1):
        bar = "#" * int(round(20 * kept_h[i] / peak))
        line = f"    [{BINS[i]:+5.0f},{BINS[i+1]:+5.0f})  {kept_h[i]:4d}  {bar:<20s}"
        if a_yaw.size:
            m = (a_yaw >= BINS[i]) & (a_yaw < BINS[i + 1])
            line += (f"  success {a_ok[m].sum():3d}/{m.sum():3d}"
                     f" = {a_ok[m].mean():4.0%}" if m.sum() else "  success   -/  -")
        print(line)

    # chi-square-free sanity number: how far from flat the kept set is
    expected = kept.size / (len(BINS) - 1)
    skew = np.abs(kept_h - expected).sum() / (2.0 * kept.size)
    print(f"    total-variation distance from uniform: {skew:.2f}  "
          f"(0 = unbiased, 1 = fully concentrated)")
    if skew > 0.15:
        print("    NOTE: the training distribution is materially skewed vs the "
              "uniform distribution\n          evaluation samples from. Report this.")


def cmd_stats(args):
    import h5py

    with h5py.File(args.path, "r") as f:
        data = f["data"]
        names = sorted(data.keys(), key=lambda n: int(n.split("_")[1]))
        lens = np.array([data[n].attrs["num_samples"] for n in names])
        ok = np.array([bool(data[n].attrs.get("successful", False)) for n in names])

        print(f"{args.path}")
        print(f"  controller : {data.attrs.get('controller', '?')}")
        print(f"  demos      : {len(names)}  ({ok.sum()} successful)")
        print(f"  transitions: {int(data.attrs['total'])}")
        print(f"  length     : min {lens.min()}  median {int(np.median(lens))}  max {lens.max()}")

        g = data[names[0]]
        print(f"  actions    : {g['actions'].shape}")
        print(f"  obs keys   :")
        for k in sorted(g["obs"].keys()):
            print(f"      {k:34s} {g['obs'][k].shape}")

        yaw_report(data, names)

        # a demo that ran to the horizon was cut off, not finished
        horizon = json.loads(data.attrs["env_args"])["env_kwargs"].get("horizon")
        if horizon and (lens >= horizon).any():
            n = int((lens >= horizon).sum())
            print(f"\n  WARNING: {n} demo(s) hit the {horizon}-step horizon. Those were "
                  f"truncated mid-task rather than ending at retreat -- raise --horizon.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("live", help="run the PD controller in the viewer")
    a.add_argument("--episodes", type=int, default=1)
    a.add_argument("--horizon", type=int, default=600)
    a.set_defaults(func=cmd_live)

    b = sub.add_parser("replay", help="replay a recorded demo in the viewer")
    b.add_argument("path")
    b.add_argument("--demo", type=int, default=0)
    b.add_argument("--trace", action="store_true", help="print the reward trace")
    b.set_defaults(func=cmd_replay)

    c = sub.add_parser("stats", help="summarise an HDF5 (no window needed)")
    c.add_argument("path")
    c.set_defaults(func=cmd_stats)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
