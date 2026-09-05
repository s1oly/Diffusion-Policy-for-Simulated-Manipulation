"""
Pure-noise controller -- ablation arm 3 of 3, the worst-case demonstrator.
(Formerly HybridController / hybrid_controller.py; that name was a historical
misnomer -- this arm is not a blend -- so it was renamed to what it is.)

Where HybridController is PD + noise, this is naive-heuristic + LARGE noise: the
measured PD control laws are swapped for a crude heuristic, and the command is
then dominated by Gaussian noise. It keeps the base FSM's phase sequencing,
gripper handling, and terminal phases -- so a rollout is still a well-formed,
terminating episode that collect_demos can record and occasionally keep -- but it
throws away everything that made PD accurate:

  * no force feedback / jam detection (insert just pushes down),
  * no nut-droop levelling (_level_nut is disabled),
  * no fine servo -- a single weak proportional pull toward the phase target,
    swamped by noise of comparable-or-larger magnitude.

The result is "worst-case demonstrations that occasionally succeed by chance,"
which is exactly the low-quality arm the ablation needs. The single thing that
differs from PDController is the control LAW (naive+noise vs measured PD); the
FSM scaffolding is shared so the three arms stay comparable.

TUNING NOTE -- this is the knob to expect trouble on. The base phase gates
(e.g. GRASP needs XY settled inside XY_FINE = 3.5 mm for DWELL steps) are hard to
satisfy under heavy noise, so at large NOISE_SIGMA the arm may almost never grasp
and collect_demos will burn through --max-attempts keeping few episodes. NAIVE_GAIN
and NOISE_SIGMA set where this sits between "never succeeds" (can't collect) and
"succeeds too often" (indistinguishable from noisy). Sweep them paired, fixed
--seed, and read rate.py's success rate + failure-by-phase histogram -- the target
is a low but nonzero keep rate. Neither value below is tuned.
"""

import numpy as np

from pd_controller import PDController, yaw_of, wrap


class PureNoiseController(PDController):
    """Naive heuristic drowned in Gaussian noise; reuses the base FSM only."""

    # Set from a paired gain x sigma sweep (N=15, same task instances per cell):
    #
    #   gain \ sigma   0.30   0.20   0.10
    #     8.0            0%     0%    13%
    #    12.0           0%     7%     7%
    #    15.0           0%     0%    40%
    #
    # Two things the sweep settled. (1) SIGMA is the lever, not gain: sigma >= 0.20
    # is ~0% at every gain -- noise pins the nut short of the peg. (2) The base
    # FSM's settle gates (XY_FINE = 3.5 mm, DWELL) force accurate tracking through
    # approach/grasp/transport, so the ONLY collectable configs track at near-PD
    # gain with small noise. Hence gain = KP_POS, sigma = 0.10 -> ~40%, right at the
    # top of the collectable band. Consequence to keep in mind: this makes the arm
    # resemble HybridController, differing mainly in the naive insert law (no force,
    # no tilt) and no fine servo -- a cruder demonstrator needs the gates relaxed.
    NAIVE_GAIN = 15.0       # = PDController.KP_POS; weaker loses the coarse gates
    NAIVE_DESCENT = 0.3     # constant downward push during insert (no force loop)
    NOISE_SIGMA = 0.10      # std on each pose channel, in action (+-1) units

    def __init__(self, env, gain=None, sigma=None, seed=None):
        super().__init__(env)
        self.gain = self.NAIVE_GAIN if gain is None else gain
        self.sigma = self.NOISE_SIGMA if sigma is None else sigma
        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1))
        self.rng = np.random.default_rng(seed)

    # -- naive replacements for the two PD control laws ----------------------
    #
    # The phase handlers (_do_approach ... _do_retreat) are inherited unchanged,
    # so they still compute sensible targets and advance the FSM. We only replace
    # what those handlers call to turn a target into an action, which is where all
    # of PD's accuracy lived. Keeping the smart targets but a crude, noisy law is
    # the cleanest single-point swap; if you want a *cruder* heuristic (eef
    # straight at peg XY, ignoring the 5.4 cm handle offset), override the target
    # here rather than reaching back into the handlers.

    def _servo_to(self, target_pos, target_yaw, obs):
        """Naive position/yaw push + large noise. No KD, no force, no tilt."""
        pos_error = np.asarray(target_pos) - obs["robot0_eef_pos"]
        yaw_error = wrap(target_yaw - yaw_of(obs["robot0_eef_quat"]), np.pi)

        action = np.zeros(7)
        action[0:3] = self.gain * pos_error
        action[5] = self.KP_ROT * yaw_error
        action[0:6] += self.rng.normal(0.0, self.sigma, size=6)
        action[6] = self.gripper_state       # gripper still held, not noised
        return np.clip(action, -1.0, 1.0)

    def _servo_insert(self, obs):
        """Naive insert: constant down-push + weak XY toward peg + noise.

        Strips the force correction, unwedging, and compliance of the base law --
        it just shoves the nut down and hopes. This is why pure noise rarely seats
        cleanly: with no jam detection it drives straight into a wedge.
        """
        nut_err = self.peg_xy - obs["SquareNut_pos"][:2]

        action = np.zeros(7)
        action[0:2] = self.gain * nut_err
        action[2] = -self.NAIVE_DESCENT
        action[0:6] += self.rng.normal(0.0, self.sigma, size=6)
        action[6] = self.gripper_state
        return np.clip(action, -1.0, 1.0)

    def _level_nut(self, obs):
        """Disabled: the naive arm does not correct nut droop."""
        return np.zeros(2)
