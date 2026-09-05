"""
Hybrid controller (PD + noise) -- ablation arm 2 of 3. (Formerly NoisyController /
noisy_controller.py; renamed so the arm names match what they are.)

This is `PDController` with exactly ONE thing changed: Gaussian noise is added to
the pose channels of the action right before it is returned. Everything else --
the phase FSM, the targets, the force feedback, the gripper sequencing -- is
inherited unchanged. The ablation (clean / hybrid / pure-noise) is only
interpretable if the three controllers differ in one thing, so this file must
stay a thin override; do NOT re-implement any sequencing here.

Interpretation: the loop still SENSES cleanly (the FSM reads the true obs) and
still CORRECTS via force feedback -- only the actuation is corrupted. Because the
goal is recomputed from the achieved pose every step, the controller closes the
loop around its own noise, which is what makes these "realistic degraded"
demonstrations rather than open-loop garbage: the policy sees a competent
operator with a shaky hand, not a broken one.

Two deliberate choices, both flagged for tuning:

  * Noise is added to the 6 POSE channels only, not the gripper (action[6]). The
    gripper is a held +-1 command; Gaussian noise on it would randomly half-open
    the fingers and drop the nut, so at any appreciable sigma the arm would never
    complete a grasp and the "noise level" knob would collapse to "does it grasp
    at all." Keeping the gripper clean makes NOISE_SIGMA a monotone quality dial,
    which is what the ablation needs. Set NOISE_GRIPPER = True to include it.
  * NOISE_SIGMA is in ACTION units (the +-1 command space), so 0.1 is ~10% of full
    scale on each channel per step -- comparable to the 0.02-0.3 commands the PD
    loop issues mid-phase. It is the sweep knob for this arm and is NOT tuned;
    pick the value(s) that give the degraded-but-functional success rate you want
    (sweep it paired, fixed --seed, exactly like KP_INSERT_XY).
"""

import numpy as np

from pd_controller import PDController


class HybridController(PDController):
    """PDController whose returned action is corrupted by additive Gaussian noise
    (the "hybrid" arm: clean PD sensing/feedback blended with noisy actuation)."""

    NOISE_SIGMA = 0.1       # std added to each pose channel, in action (+-1) units
    NOISE_GRIPPER = False   # keep the held gripper command clean (see module docstring)

    def __init__(self, env, sigma=None, seed=None):
        super().__init__(env)
        self.sigma = self.NOISE_SIGMA if sigma is None else sigma
        # Seed from the legacy global so a `np.random.seed(args.seed)` in
        # collect_demos makes the noise realization reproducible for the whole
        # run, while still differing episode-to-episode within it.
        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1))
        self.rng = np.random.default_rng(seed)

    def get_action(self, obs):
        # The base call runs the full clean FSM against the true obs and returns
        # an already-clipped 7-dim action. We corrupt and re-clip.
        action = np.asarray(super().get_action(obs), dtype=np.float64).copy()

        n = 7 if self.NOISE_GRIPPER else 6
        action[:n] += self.rng.normal(0.0, self.sigma, size=n)
        return np.clip(action, -1.0, 1.0)
