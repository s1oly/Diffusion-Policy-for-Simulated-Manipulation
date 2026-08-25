"""
Collect demonstrations from a classical controller into a robomimic-style HDF5.

    python peg_insertion/data_collection/collect_demos.py --controller pd --n 100

Output layout (what diffusion_policy's robomimic lowdim loader expects):

    data/                          attrs: total, env_args
      demo_0/                      attrs: num_samples, successful
        obs/<key>   (T, D)         one dataset per low-dim observation key
        actions     (T, A)
        rewards     (T,)
        dones       (T,)
        states      (T, S)         flattened mujoco state, for exact replay

NOTE: robosuite's DataCollectionWrapper is NOT used here. Despite what you may
read, it does not save observations or rewards and does not write HDF5 -- it
dumps flattened mujoco states + actions into per-episode .npz directories, which
then need robomimic's dataset_states_to_obs.py to be replayed back into
observations. Since we drive the env from a scripted controller we already have
obs in hand each step, so we record it directly and skip that round trip. The
`states` dataset is still saved so episodes remain exactly replayable.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "controllers"))

import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.environments.base import register_env

from Manipulation_Enviroment import Manipulation_Enviroment

register_env(Manipulation_Enviroment)

ENV_NAME = Manipulation_Enviroment.__name__


def spawn_yaw_deg(obs):
    """Nut spawn yaw in degrees, from an (x,y,z,w) obs quaternion.

    Recorded per demo because success is BIASED along this axis and only
    successful episodes are kept -- so the training distribution is not the
    uniform-yaw distribution that evaluation samples from. Keeping the attempted
    and kept yaws makes that bias measurable after the fact instead of invisible.
    """
    R = T.quat2mat(obs["SquareNut_quat"])
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


class RandomController:
    """Placeholder so the recording pipeline is testable before Phase 3 lands.

    Any controller plugged in here needs the same three methods.
    """

    def __init__(self, env):
        self.env = env
        self.rng = np.random.default_rng()

    def reset(self):
        pass

    def get_action(self, obs):
        return self.rng.uniform(-1.0, 1.0, self.env.action_dim)


def build_controller(name, env):
    if name == "random":
        return RandomController(env)
    # Phase 3: each of these exposes __init__(env) / reset() / get_action(obs).
    if name == "pd":
        from pd_controller import PDController
        return PDController(env)
    if name == "noisy":
        from noisy_controller import NoisyController
        return NoisyController(env)
    if name == "hybrid":
        from hybrid_controller import HybridController
        return HybridController(env)
    raise ValueError(f"unknown controller: {name}")


def make_env(horizon):
    return suite.make(
        ENV_NAME,
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=horizon,
    )


def lowdim_keys(obs):
    """Every observation key that is not an image and is numeric."""
    keys = []
    for k, v in obs.items():
        if k.endswith("image") or k.endswith("depth") or k.endswith("segmentation"):
            continue
        if np.asarray(v).dtype.kind in "fiub":
            keys.append(k)
    return sorted(keys)


def run_episode(env, controller, keys):
    """Roll out one episode. Returns a dict of arrays, plus the success flag."""
    env.reset()
    # reset() returns SquareNut_to_robot0_eef_pos = [0,0,0] and _quat = [0,0,0,0];
    # they only populate after the first step. With obs_horizon = 2 that makes the
    # first training window of every demo half fabricated.
    obs = env._get_observations(force_update=True)
    controller.reset()
    yaw0 = spawn_yaw_deg(obs)

    traj = {k: [] for k in keys}
    actions, rewards, states = [], [], []
    success = False

    done = False
    while not done:
        state = env.sim.get_state().flatten()
        action = np.asarray(controller.get_action(obs), dtype=np.float64)

        for k in keys:
            traj[k].append(np.asarray(obs[k], dtype=np.float64).ravel())
        states.append(state)
        actions.append(action)

        obs, reward, done, _ = env.step(action)

        rewards.append(reward)
        # env._check_success() was already refreshed by env.reward() inside step()
        if env._check_success():
            success = True

        # End the rollout when the FSM finishes, not at the horizon. Diffusion
        # policy samples fixed-length windows uniformly across the buffer, so
        # trailing idle steps would teach the policy to output zeros. The demo
        # ends at RELEASE + RETREAT, which is past the first _check_success.
        if getattr(controller, "finished", False):
            done = True

    # robomimic expects `dones` one-hot at the terminal step, not sticky-after-
    # success. The success flag lives in the demo's `successful` attr instead.
    dones = np.zeros(len(actions), dtype=np.int64)
    if len(dones):
        dones[-1] = 1

    episode = {k: np.array(v) for k, v in traj.items()}
    episode["actions"] = np.array(actions)
    episode["rewards"] = np.array(rewards)
    episode["dones"] = dones
    episode["states"] = np.array(states)
    episode["successful"] = success
    episode["spawn_yaw"] = yaw0
    return episode, success


def write_hdf5(path, episodes, env, keys, controller_name,
               attempt_yaws=None, attempt_success=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "w") as f:
        grp = f.create_group("data")
        total = 0
        for i, ep in enumerate(episodes):
            g = grp.create_group(f"demo_{i}")
            og = g.create_group("obs")
            for k in keys:
                og.create_dataset(k, data=ep[k], compression="gzip")
            for k in ("actions", "rewards", "dones", "states"):
                g.create_dataset(k, data=ep[k], compression="gzip")
            g.attrs["num_samples"] = len(ep["actions"])
            g.attrs["successful"] = bool(ep["successful"])
            g.attrs["spawn_yaw"] = float(ep["spawn_yaw"])
            total += len(ep["actions"])

        grp.attrs["total"] = total
        grp.attrs["env_args"] = json.dumps({
            "env_name": ENV_NAME,
            "type": 1,
            "env_kwargs": {
                "robots": "Panda",
                "control_freq": env.control_freq,
                "horizon": env.horizon,
                "reward_shaping": env.reward_shaping,
                "use_camera_obs": False,
            },
        })
        grp.attrs["controller"] = controller_name
        grp.attrs["obs_keys"] = json.dumps(keys)
        # Every attempt, kept or discarded. Attempts are uniform in yaw by
        # construction, so pairing these two arrays gives the per-yaw success
        # rate -- i.e. exactly how the filter skewed the training set.
        if attempt_yaws is not None:
            grp.attrs["attempt_yaws"] = np.asarray(attempt_yaws, dtype=np.float64)
            grp.attrs["attempt_success"] = np.asarray(attempt_success, dtype=np.int64)
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controller", default="random", choices=["random", "pd", "noisy", "hybrid"])
    p.add_argument("--n", type=int, default=10, help="number of SUCCESSFUL episodes to keep")
    p.add_argument("--max-attempts", type=int, default=None,
                   help="give up after this many rollouts (default: 10x --n)")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-failures", action="store_true",
                   help="save every rollout instead of only successful ones")
    args = p.parse_args()

    out = args.out or f"data/{args.controller}_{args.n}.hdf5"
    max_attempts = args.max_attempts or args.n * 10

    np.random.seed(args.seed)
    env = make_env(args.horizon)
    controller = build_controller(args.controller, env)

    keys = lowdim_keys(env.reset())
    print(f"recording {len(keys)} low-dim obs keys: {keys}\n")

    episodes = []
    attempt_yaws, attempt_success = [], []
    attempts = 0
    while len(episodes) < args.n and attempts < max_attempts:
        attempts += 1
        ep, success = run_episode(env, controller, keys)
        attempt_yaws.append(ep["spawn_yaw"])
        attempt_success.append(int(success))
        if success or args.keep_failures:
            episodes.append(ep)
        print(f"  attempt {attempts:4d}  T={len(ep['actions']):4d}  "
              f"final_reward={ep['rewards'][-1]:.3f}  "
              f"{'SUCCESS' if success else 'fail'}  kept={len(episodes)}")

    env.close()

    if not episodes:
        print(f"\nNo episodes collected in {attempts} attempts - nothing written.")
        sys.exit(1)

    total = write_hdf5(out, episodes, env, keys, args.controller,
                       attempt_yaws, attempt_success)
    rate = sum(1 for e in episodes if e["successful"]) / attempts
    print(f"\nwrote {len(episodes)} episodes ({total} transitions) -> {out}")
    print(f"success rate: {rate:.1%} over {attempts} attempts")


if __name__ == "__main__":
    main()
