"""
Controller regression harness: success rate, failure breakdown, phase tracing.

Run this after ANY change to a controller. Success rate alone is not enough to
debug with -- the useful signal is *which phase* the failures die in, because
each phase fails for a different physical reason.

    mamba activate diffpol

    # the regression check: success rate + per-episode table + failure histogram
    python peg_insertion/tools/rate.py --n 20

    # per-step trace of one phase in one episode, to see WHY it fails
    python peg_insertion/tools/rate.py --episode 2 --trace insert

    # why transport would not hand off: which gate component is blocking
    python peg_insertion/tools/rate.py --n 20 --gate

Headless, so plain `python`. For a window use data_collection/view_demo.py.
"""

import argparse
import collections
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "envs"))
sys.path.insert(0, os.path.join(HERE, "..", "controllers"))

import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.environments.base import register_env

from Manipulation_Enviroment import Manipulation_Enviroment

register_env(Manipulation_Enviroment)
ENV_NAME = Manipulation_Enviroment.__name__


def build(name, env):
    if name == "pd":
        from pd_controller import PDController
        return PDController(env)
    if name == "hybrid":
        from hybrid_controller import HybridController
        return HybridController(env)
    if name == "pure_noise":
        from pure_noise_controller import PureNoiseController
        return PureNoiseController(env)
    raise SystemExit(f"unknown controller: {name}")


def trace_line(ctrl, obs, action, t):
    """One line of per-step telemetry -- the quantities that actually diagnose."""
    nut = obs["SquareNut_pos"]
    eef = obs["robot0_eef_pos"]
    f = ctrl._world_force(obs)
    dxy = np.linalg.norm(nut[:2] - ctrl.peg_xy) * 1000
    tilt = np.degrees(ctrl._nut_tilt(obs))
    return (f"  t={t:4d} {ctrl.phase.value:9s} nutz={nut[2]:.4f} eefz={eef[2]:.4f} "
            f"|dxy|={dxy:6.2f}mm tilt={tilt:5.2f}deg "
            f"f=[{f[0]:6.1f} {f[1]:6.1f} {f[2]:6.1f}] "
            f"a=[{action[0]:6.3f} {action[1]:6.3f} {action[2]:6.3f} {action[5]:6.3f}]")


def gate_line(ctrl, obs):
    """The four transport->insert gate components, so you can see which blocks."""
    from pd_controller import yaw_of, wrap
    _, th_d = ctrl._eef_target_for_nut(ctrl.peg_xy, ctrl.psi_des)
    return dict(
        xy=np.linalg.norm(obs["SquareNut_pos"][:2] - ctrl.peg_xy) * 1000,
        yaw=abs(np.degrees(wrap(th_d - yaw_of(obs["robot0_eef_quat"]), np.pi))),
        clear=(obs["SquareNut_pos"][2] - 0.01 - ctrl.PEG_TIP_Z) * 1000,
        tilt=np.degrees(ctrl._nut_tilt(obs)),
        steps=ctrl.phase_steps,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controller", default="pd", choices=["pd", "hybrid", "pure_noise"])
    p.add_argument("--n", type=int, default=20, help="episodes to run")
    p.add_argument("--horizon", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episode", type=int, default=None,
                   help="with --trace, which episode to watch")
    p.add_argument("--trace", default=None,
                   help="phase name to trace per-step, e.g. insert / transport / descend")
    p.add_argument("--every", type=int, default=6, help="trace every Nth step")
    p.add_argument("--gate", action="store_true",
                   help="on failure, print the transport gate components")
    args = p.parse_args()

    env = suite.make(ENV_NAME, robots="Panda", has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False,
                     control_freq=20, horizon=args.horizon, seed=args.seed)
    ctrl = build(args.controller, env)

    n = args.n if args.episode is None else args.episode + 1
    fails = collections.Counter()
    lens, ok = [], 0
    yaws_ok, yaws_bad = [], []

    for ep in range(n):
        env.reset()
        obs = env._get_observations(force_update=True)
        ctrl.reset()
        # nut spawn yaw is uniform over 360 deg and is the task's dominant
        # variation, so it is the axis any success/failure bias shows up along
        from pd_controller import yaw_of
        yaw0 = np.degrees(yaw_of(obs["SquareNut_quat"]))

        watch = args.trace is not None and (args.episode is None or ep == args.episode)
        last_live, t, k = ctrl.phase, 0, 0
        gate = {}

        while not ctrl.finished and t < args.horizon:
            action = ctrl.get_action(obs)

            if watch and ctrl.phase.value == args.trace:
                if k % args.every == 0:
                    print(trace_line(ctrl, obs, action, t))
                k += 1
            if ctrl.phase.value == "transport":
                gate = gate_line(ctrl, obs)
            if not ctrl.finished:
                last_live = ctrl.phase

            obs, _, _, _ = env.step(action)
            t += 1

        success = bool(env._check_success())
        ok += success
        lens.append(t)
        (yaws_ok if success else yaws_bad).append(yaw0)
        if not success:
            fails[last_live.value] += 1

        print(f"ep{ep:3d} yaw0={yaw0:7.1f} end={ctrl.phase.value:7s} last={last_live.value:9s} t={t:4d} "
              f"success={success} nut_z={obs['SquareNut_pos'][2]:.4f} "
              f"xy={np.linalg.norm(obs['SquareNut_pos'][:2] - ctrl.peg_xy):.4f}")

        if args.gate and not success and gate:
            print(f"     gate: xy={gate['xy']:6.2f}mm(<{ctrl.TRANSPORT_XY_TOL*1000:.1f}) "
                  f"yaw={gate['yaw']:5.2f}deg(<{np.degrees(ctrl.YAW_TOL):.0f}) "
                  f"clear={gate['clear']:6.2f}mm(>5) "
                  f"tilt={gate['tilt']:5.2f}deg(<{np.degrees(ctrl.TILT_TOL):.0f}) "
                  f"steps={gate['steps']}")

    print(f"\nsuccess {ok}/{n} = {ok / n:.0%}   median len {int(np.median(lens))}")
    print("failures by last live phase:", dict(fails))
    if yaws_ok and yaws_bad:
        print(f"spawn yaw: success mean |yaw| {np.mean(np.abs(yaws_ok)):5.1f} deg, "
              f"failure mean |yaw| {np.mean(np.abs(yaws_bad)):5.1f} deg")
    env.close()


if __name__ == "__main__":
    main()
