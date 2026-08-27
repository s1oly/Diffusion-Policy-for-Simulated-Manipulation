"""
Clean PD controller for the square-nut peg insertion task.

Two layers matter here. robosuite already runs an OSC impedance controller
(kp=150) underneath; this class does not compute torques, it chooses a SETPOINT
each control step. action[0:3] = 1.0 means "put the goal 5 cm ahead of where I am
now", and the goal is recomputed from the achieved pose every step.

Action layout (7-dim, all clipped to [-1, 1]):

    [0:3]  position delta   +-1 -> +-0.05 m goal offset, base frame
    [3:6]  axis-angle delta +-1 -> +-0.50 rad, base frame (origin_ori = I, so
                                 index 5 rotates about WORLD z)
    [6]    gripper          +1 closes, -1 opens; a HELD command, not an event

Structure: `get_action` owns the state machine -- it decides what the target is
and when to advance. `_servo_to` is phase-agnostic and answers only "given a
target, what action gets me there". `_servo_insert` is the one phase with a
different control law, because XY there regulates contact FORCE rather than
position.

Gains were measured, not guessed (see controller_practice.py):
  * kp = 15 on position error in metres converges in ~12 steps with no overshoot
    at both 150 mm and 7 mm travel; kd = 0.
  * There is a ~0.1 mm floor from the OSC controller's own steady-state
    deflection that no proportional gain closes. Task clearance is 6.75 mm, so
    P-only is correct here and an integrator would only add windup to manage.
  * kp_rot = kp_pos * (0.05 / 0.5) = kp_pos / 10, so both channels request the
    same fraction of their error per step despite the 10x scale difference.
"""

import numpy as np
from enum import Enum

import robosuite.utils.transform_utils as T


def yaw_of(quat):
    """World yaw (rad) from an (x, y, z, w) quaternion.

    robosuite's transform_utils is xyzw; MuJoCo's sim.data.body_xquat is wxyz.
    Mixing them silently produces garbage, so only obs quats come through here.
    """
    R = T.quat2mat(quat)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def wrap(angle, period=2.0 * np.pi):
    """Shortest equivalent rotation: wraps into (-period/2, +period/2]."""
    return float(angle - period * np.round(angle / period))


def rot2d(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


class Phase(Enum):
    APPROACH = "approach"      # over the handle, at safe height
    DESCEND = "descend"        # down onto the handle
    GRASP = "grasp"            # hold still, close fingers
    LIFT = "lift"              # straight up, clear of the table
    TRANSPORT = "transport"    # nut over the peg, yaw aligned (absorbs "align")
    INSERT = "insert"          # descend onto peg; XY on force, Z on position
    RELEASE = "release"        # hold pose, open fingers
    RETREAT = "retreat"        # straight up, clear of the nut
    DONE = "done"              # terminal -- collect_demos stops the episode here
    FAILED = "failed"          # terminal, unsuccessfully


TERMINAL = (Phase.DONE, Phase.FAILED)


class PDController:
    """Phase-indexed state machine driving the env from `obs`.

    NoisyController and HybridController subclass this so the ablation differs in
    exactly one thing. Do not re-implement the sequencing downstream.
    """

    # ---- geometry, metres (measured; see roadmap "Task geometry") ----
    TABLE_Z = 0.82          # table top surface
    PEG_TIP_Z = 0.95        # top of the exposed peg
    # Hover height. The nut hangs ~1 cm below the eef and is 1 cm thick, so its
    # underside sits at eef_z - 0.02; clearing the 0.95 peg tip needs eef_z >
    # 0.97. It must ALSO be reachable: the peg is ~0.79 m from the Panda base, and
    # at 1.02 the arm cannot make it -- measured, transport stalled with 7-18 mm
    # of residual error in BOTH z and xy, the signature of a workspace limit
    # rather than a control problem.
    SAFE_Z = 1.01
    TRANSPORT_CAP = 0.30    # a saturated 35 cm traverse swings the nut on its
                            # 5.4 cm arm; measured tilt reached 30 deg
    GRASP_DZ = 0.0          # eef z relative to nut body origin when on the handle
    HANDLE_R = 0.054        # handle site -> nut body origin, rotates with nut yaw
    SEATED_Z = 0.830        # nut body z once resting seated on the peg
    # `on_peg` accepts z < table_offset + 0.05 = 0.87, and a nut held at 0.85 is
    # already inserted -- it drops the last 2 cm on release. Insisting on 0.83
    # while still gripping asks the arm to push through a hold it cannot break.
    SEATED_TOL = 0.040
    # Reach ceiling from the Panda base, metres. The nut hangs 5.4 cm off the
    # handle, so the orientation chosen at grasp decides whether the eef ends up
    # on the near or far side of the peg -- and the far side can exceed the arm's
    # reach. Measured with tools/rate.py: successful transports sit at <= 0.806 m
    # of base->eef reach, failures pile up at ~0.85. `_capture_grasp` uses this to
    # turn to the closer 90-deg-equivalent orientation only when the default one
    # is over the wall (paired seeds 0-4: 67% -> 79%, no seed regressing).
    REACH_WALL = 0.81

    # ---- gains ----
    KP_POS = 15.0
    KD_POS = 0.0
    KP_ROT = 1.5            # = KP_POS * (0.05 / 0.5)
    # Insert XY is deliberately SOFT. At KP_POS the servo wins every argument
    # with the peg: once the nut touches down it cannot move, the eef keeps
    # driving toward the target anyway, and the grip twists the nut into a
    # tighter wedge. Measured, that grew a 0.8 mm error to 8.5 mm and locked.
    #
    # TODO(tuning): 5.0 is NOT converged -- it is one end of a bracket. Dropped
    # from 15 to stop the wedging, and at 5 the command is only 0.02-0.07 against
    # a 10-12 mm post-contact error, too soft to overcome friction (measured via
    # `tools/rate.py --trace insert`). The right value is somewhere in 5-15 and
    # is the clearest remaining lever on the 55% success rate. Sweep it with a
    # FIXED --seed so the comparison is paired.
    KP_INSERT_XY = 5.0
    DESCENT_RATE = 0.22     # constant command, not proportional: a steady gentle push
    # Cap on the descend-to-grasp z command. A saturated 13 cm drop drags XY
    # along with it -- measured cross-axis coupling is ~5.6%, which turned a
    # 1.6 mm handle error into 9.6 mm by touchdown and made the fingers hit the
    # nut instead of straddling the handle. Slower z lets XY keep up.
    APPROACH_DESCENT_CAP = 0.25
    # The nut hangs off the handle 5.4 cm out and droops under gravity. Left
    # uncorrected the droop RUNS AWAY on contact -- measured 4.4 deg growing to
    # 16.5 deg -- and the tilted plate wedges across the peg instead of sliding
    # down it. Nothing else in the controller commands wrist roll/pitch.
    KP_TILT = 4.0
    MAX_TILT_CMD = 0.35
    KP_LATERAL = 0.004      # command per newton
    KD_LATERAL = 0.0004

    # ---- force setpoints, newtons, AFTER subtracting the per-reset bias ----
    LATERAL_DEADBAND = 5.0      # below this it is sensor noise, not binding
    SEAT_FORCE_THRESH = 35.0    # vertical spike meaning the nut has bottomed out
    MAX_LATERAL_CMD = 0.15      # cap the force term's authority over XY
    JAM_FORCE = 12.0            # lateral force meaning the nut is binding
    JAM_PATIENCE = 10           # binding steps tolerated before backing off
    BACKOFF_STEPS = 8           # how long to lift when unwedging
    BACKOFF_RATE = 0.18

    # ---- tolerances ----
    XY_COARSE = 0.010
    XY_FINE = 0.0035        # well inside the 6.75 mm clearance
    # The hanging nut cannot be servoed tighter than ~5 mm: it swings on a 5.4 cm
    # arm and the levelling command keeps shifting it. Demanding 3.5 mm here just
    # times transport out. Insert is compliant and force-corrected, so it closes
    # the rest -- measured, the residual sits at 4.8-6 mm on the near misses.
    TRANSPORT_XY_TOL = 0.0055
    Z_TOL = 0.008
    YAW_TOL = np.deg2rad(5)
    TILT_TOL = np.deg2rad(3)
    DWELL = 3               # consecutive steps a predicate must hold before advancing

    # ---- budgets, in control steps ----
    PHASE_TIMEOUT = 180     # a stuck phase fails the episode rather than hanging
    INSERT_TIMEOUT = 200    # the descent is deliberately slow, so it needs more
    TRANSPORT_TIMEOUT = 260 # ~35 cm traversed at a capped rate, then fine align
    GRIPPER_SETTLE = 18     # measured: ~25 steps for a full open/close travel
    RETREAT_STEPS = 15
    GRASP_LOSS_STEPS = 4    # _check_grasp flickers; require consecutive misses

    def __init__(self, env):
        self.env = env

        # set on reset, never hardcoded
        self.peg_xy = None
        self.force_bias = None
        self.phase = None
        self.gripper_state = None
        self.control_steps = 0
        self.phase_steps = 0
        self.dwell = 0
        self.grasp_misses = 0
        self.jam_steps = 0
        self.backoff_steps = 0

        # captured at grasp: the nut's pose relative to the eef, which is fixed
        # in the EEF frame because the nut is rigidly held
        self.grasp_offset = None    # (2,) nut_xy - eef_xy, expressed in eef frame
        self.grasp_dyaw = None      # nut_yaw - eef_yaw
        self.psi_des = None         # chosen once: reach-gated 90-deg-equivalent yaw
        self.grasp_theta = None     # wrist yaw at grasp; held through LIFT

        self.prev_pos_error = np.zeros(3)
        self.prev_lateral_error = np.zeros(2)

        self.dt = 1.0 / env.control_freq
        self.nut = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        self.peg_xy = self.env.sim.data.body_xpos[self.env.peg1_body_id][:2].copy()
        self.force_bias = self.env.robots[0].ee_force["right"].copy()
        self.nut = self._square_nut()

        self.phase = Phase.APPROACH
        self.gripper_state = -1.0          # open
        self.control_steps = 0
        self.phase_steps = 0
        self.dwell = 0
        self.grasp_misses = 0
        self.jam_steps = 0
        self.backoff_steps = 0

        self.grasp_offset = None
        self.grasp_dyaw = None
        self.psi_des = None
        self.grasp_theta = None
        self.prev_pos_error = np.zeros(3)
        self.prev_lateral_error = np.zeros(2)

    @property
    def finished(self):
        """True once the FSM is done. collect_demos ends the episode on this."""
        return self.phase in TERMINAL

    def _square_nut(self):
        for nut in self.env.nuts:
            if "square" in nut.name.lower():
                return nut
        return self.env.nuts[0]

    # ------------------------------------------------------------------
    # state machine
    # ------------------------------------------------------------------

    def get_action(self, obs):
        """Advance the FSM one step and return a 7-dim action."""
        self.control_steps += 1
        self.phase_steps += 1

        if self.phase in TERMINAL:
            return self._hold(obs)

        budget = {Phase.INSERT: self.INSERT_TIMEOUT,
                  Phase.TRANSPORT: self.TRANSPORT_TIMEOUT}.get(self.phase, self.PHASE_TIMEOUT)
        if self.phase_steps > budget:
            self.phase = Phase.FAILED
            return self._hold(obs)

        handler = getattr(self, "_do_" + self.phase.value)
        return handler(obs)

    def _advance(self, nxt):
        self.phase = nxt
        self.phase_steps = 0
        self.dwell = 0

    def _settled(self, predicate):
        """True only after `predicate` has held for DWELL consecutive steps.

        Steady-state error means a bare `error < tol` test chatters; this is the
        dwell counter, in CONTROL STEPS. Wall-clock would be nondeterministic
        because the sim does not run in real time.
        """
        self.dwell = self.dwell + 1 if predicate else 0
        return self.dwell >= self.DWELL

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------

    def _handle_xy(self, obs):
        """Grasp target. The handle is 5.4 cm from the nut origin and its
        direction equals the nut yaw exactly, so no site lookup is needed."""
        psi = yaw_of(obs["SquareNut_quat"])
        return obs["SquareNut_pos"][:2] + self.HANDLE_R * np.array([np.cos(psi), np.sin(psi)])

    def _capture_grasp(self, obs):
        """Record the nut's pose relative to the eef, in the EEF frame.

        Expressed there it is constant, because the nut is rigidly held and the
        offset rotates with the wrist. Measuring it beats assuming a perfect
        grasp on the handle -- the real one is a few mm off every episode.
        """
        theta = yaw_of(obs["robot0_eef_quat"])
        psi = yaw_of(obs["SquareNut_quat"])
        d_world = obs["SquareNut_pos"][:2] - obs["robot0_eef_pos"][:2]
        self.grasp_offset = rot2d(-theta) @ d_world
        self.grasp_dyaw = wrap(psi - theta)
        self.grasp_theta = theta    # hold this through LIFT; rotate in TRANSPORT
        # Choose psi_des ONCE (recomputing each step would chatter near a 45-deg
        # boundary). The square is 4-fold symmetric, but the gripper is symmetric
        # under 180 deg, so only TWO orientations are physically distinct at the
        # wrist: the nearest 90-deg-equivalent and a 90-deg turn from it. They put
        # the eef on opposite sides of the peg; when the nearest one lands the eef
        # past REACH_WALL and the turned one is closer, turn to bring it in reach.
        psi0 = psi - wrap(psi, np.pi / 2.0)
        base_xy = self.env.sim.data.get_body_xpos(
            self.env.robots[0].robot_model.root_body)[:2]
        reach = {}
        for k in (0, 1):
            th = wrap(psi0 + k * np.pi / 2.0 - self.grasp_dyaw)
            reach[k] = np.linalg.norm(self.peg_xy - rot2d(th) @ self.grasp_offset - base_xy)
        turn = reach[0] > self.REACH_WALL and reach[1] < reach[0]
        self.psi_des = wrap(psi0 + (np.pi / 2.0 if turn else 0.0))

    def _eef_target_for_nut(self, nut_xy, psi_des):
        """Invert the grasp offset: where must the eef be to put the nut there?

        Because the position target is derived FROM the desired yaw, position and
        yaw converge as one motion instead of fighting each other -- rotating the
        wrist swings the nut on a 5.4 cm radius, so servoing them independently
        means each undoes the other.
        """
        theta_des = wrap(psi_des - self.grasp_dyaw)
        eef_xy = np.asarray(nut_xy) - rot2d(theta_des) @ self.grasp_offset
        return eef_xy, theta_des

    def _nut_xy_now(self, obs):
        theta = yaw_of(obs["robot0_eef_quat"])
        return obs["robot0_eef_pos"][:2] + rot2d(theta) @ self.grasp_offset

    def _grasped(self):
        return self.env._check_grasp(
            gripper=self.env.robots[0].gripper["right"],
            object_geoms=self.nut.contact_geoms,
        )

    def _lost_grasp(self):
        """_check_grasp flickers on a firm hold, so a single miss is not a drop."""
        self.grasp_misses = 0 if self._grasped() else self.grasp_misses + 1
        return self.grasp_misses >= self.GRASP_LOSS_STEPS

    def _nut_target_eef_xy(self, obs, nut_goal_xy):
        """Closed-loop correction: shift the eef by the nut's own XY error.

        Driving this off the live SquareNut_pos rather than the offset captured at
        grasp matters -- the nut shifts a few mm in the fingers during transport,
        and an open-loop offset lets that error through to the gate unseen.
        """
        return obs["robot0_eef_pos"][:2] + (np.asarray(nut_goal_xy) - obs["SquareNut_pos"][:2])

    def _nut_tilt(self, obs):
        """Angle (rad) between the nut's plane normal and vertical."""
        nz = T.quat2mat(obs["SquareNut_quat"])[:, 2]
        return float(np.arccos(np.clip(abs(nz[2]), -1.0, 1.0)))

    def _level_nut(self, obs):
        """Roll/pitch command (action[3:5]) that brings the nut back to flat.

        The nut's own z axis in world is R[:, 2]; the rotation that takes it to
        vertical is about `nz x z_world`, with magnitude sin(tilt). The plate is
        symmetric, so flip to the upward normal first.
        """
        nz = T.quat2mat(obs["SquareNut_quat"])[:, 2]
        if nz[2] < 0.0:
            nz = -nz
        axis = np.cross(nz, np.array([0.0, 0.0, 1.0]))
        return np.clip(self.KP_TILT * axis[:2], -self.MAX_TILT_CMD, self.MAX_TILT_CMD)

    def _world_force(self, obs):
        """Contact force in WORLD axes, bias-subtracted.

        ee_force is in the EEF SITE frame and carries a standing ~4 N bias with
        no contact at all, so both corrections are needed before thresholding.
        """
        f = self.env.robots[0].ee_force["right"] - self.force_bias
        return T.quat2mat(obs["robot0_eef_quat"]) @ f

    # ------------------------------------------------------------------
    # control laws
    # ------------------------------------------------------------------

    def _servo_to(self, target_pos, target_yaw, obs):
        """Phase-agnostic position + yaw servo. Knows nothing about phases."""
        pos_error = np.asarray(target_pos) - obs["robot0_eef_pos"]
        theta = yaw_of(obs["robot0_eef_quat"])
        # the gripper is symmetric under 180 degrees, so never rotate more than 90
        yaw_error = wrap(target_yaw - theta, np.pi)

        cmd = self.KP_POS * pos_error
        if self.KD_POS:
            cmd = cmd + self.KD_POS * (pos_error - self.prev_pos_error) / self.dt
        self.prev_pos_error = pos_error

        action = np.zeros(7)
        action[0:3] = cmd
        action[5] = self.KP_ROT * yaw_error       # index 5 is about WORLD z
        action[6] = self.gripper_state
        return np.clip(action, -1.0, 1.0)

    def _servo_insert(self, obs):
        """Insert-phase law: compliant XY, constant-rate Z, force-driven unwedging.

        This is the one phase that is not `_servo_to`. Three differences, each
        forced by a measurement:
          * XY runs soft (KP_INSERT_XY), because a stiff servo wedges the nut.
          * Z is a constant rate rather than proportional, so the downward push
            does not grow as the remaining distance does.
          * Lateral force both corrects XY and, when it persists, suspends the
            descent and lifts -- you cannot push a wedged part free.
        """
        _, theta_des = self._eef_target_for_nut(self.peg_xy, self.psi_des)
        nut_err = self.peg_xy - obs["SquareNut_pos"][:2]
        yaw_error = wrap(theta_des - yaw_of(obs["robot0_eef_quat"]), np.pi)

        action = np.zeros(7)
        action[0:2] = self.KP_INSERT_XY * nut_err
        action[3:5] = self._level_nut(obs)
        action[5] = self.KP_ROT * yaw_error
        action[6] = self.gripper_state

        f = self._world_force(obs)
        lateral = f[:2]
        mag = float(np.linalg.norm(lateral))

        # XY force correction, outside a deadband set by the sensor noise floor
        if mag > self.LATERAL_DEADBAND:
            err = lateral * (1.0 - self.LATERAL_DEADBAND / mag)
            corr = self.KP_LATERAL * err
            corr = corr + self.KD_LATERAL * (err - self.prev_lateral_error) / self.dt
            self.prev_lateral_error = err
            action[0:2] -= np.clip(corr, -self.MAX_LATERAL_CMD, self.MAX_LATERAL_CMD)
        else:
            self.prev_lateral_error = np.zeros(2)

        # Z: descend, pause while binding, lift if the binding persists
        if self.backoff_steps > 0:
            self.backoff_steps -= 1
            action[2] = self.BACKOFF_RATE
            action[0:2] *= 0.3          # relax laterally too, so it can re-centre
        elif mag > self.JAM_FORCE:
            self.jam_steps += 1
            action[2] = 0.0
            if self.jam_steps > self.JAM_PATIENCE:
                self.jam_steps = 0
                self.backoff_steps = self.BACKOFF_STEPS
        else:
            self.jam_steps = max(0, self.jam_steps - 1)
            action[2] = -self.DESCENT_RATE

        return np.clip(action, -1.0, 1.0)

    def _hold(self, obs):
        """Stay put. Used during grasp/release, where drifting drags the nut."""
        action = np.zeros(7)
        action[6] = self.gripper_state
        return action

    # ------------------------------------------------------------------
    # phase handlers -- each returns exactly one action, always
    # ------------------------------------------------------------------

    def _do_approach(self, obs):
        handle = self._handle_xy(obs)
        psi = yaw_of(obs["SquareNut_quat"])
        target = np.array([handle[0], handle[1], self.SAFE_Z])

        xy_err = np.linalg.norm(obs["robot0_eef_pos"][:2] - handle)
        z_err = abs(obs["robot0_eef_pos"][2] - self.SAFE_Z)
        yaw_err = abs(wrap(psi - yaw_of(obs["robot0_eef_quat"]), np.pi))

        if self._settled(xy_err < self.XY_FINE and z_err < self.Z_TOL
                         and yaw_err < self.YAW_TOL):
            self._advance(Phase.DESCEND)
        return self._servo_to(target, psi, obs)

    def _do_descend(self, obs):
        handle = self._handle_xy(obs)
        psi = yaw_of(obs["SquareNut_quat"])
        grasp_z = obs["SquareNut_pos"][2] + self.GRASP_DZ
        target = np.array([handle[0], handle[1], grasp_z])

        xy_err = np.linalg.norm(obs["robot0_eef_pos"][:2] - handle)
        if self._settled(abs(obs["robot0_eef_pos"][2] - grasp_z) < self.Z_TOL
                         and xy_err < self.XY_FINE):
            self.gripper_state = 1.0       # close, and HOLD it from here on
            self._advance(Phase.GRASP)

        action = self._servo_to(target, psi, obs)
        action[2] = np.clip(action[2], -self.APPROACH_DESCENT_CAP, self.APPROACH_DESCENT_CAP)
        return action

    def _do_grasp(self, obs):
        # hold the pose while the fingers travel; servoing here would drag the nut
        if self.phase_steps >= self.GRIPPER_SETTLE:
            if self._grasped():
                self._capture_grasp(obs)
                self._advance(Phase.LIFT)
            else:
                self._advance(Phase.FAILED)
        return self._hold(obs)

    def _do_lift(self, obs):
        if self._lost_grasp():
            self._advance(Phase.FAILED)
            return self._hold(obs)

        eef = obs["robot0_eef_pos"]
        target = np.array([eef[0], eef[1], self.SAFE_Z])

        if self._settled(abs(eef[2] - self.SAFE_Z) < self.Z_TOL):
            self._advance(Phase.TRANSPORT)
        # Hold the grasp orientation and climb straight up; do NOT pre-rotate to
        # theta_des here. A ~90-deg wrist turn while still low flings the nut on
        # its 5.4 cm arm (tilt spikes to 10-15 deg) and retracts the arm into a
        # pose it cannot re-extend from -- measured, the nut ended 350-380 mm off
        # the peg. TRANSPORT does the rotation at SAFE_Z where the nut has room.
        return self._servo_to(target, self.grasp_theta, obs)

    def _do_transport(self, obs):
        """Absorbs the old separate 'align' phase.

        With the nut being servoed (not the eef), transport and align compute the
        same target and differ only in tolerance -- so they are one phase whose
        exit gate is the tight one.
        """
        if self._lost_grasp():
            self._advance(Phase.FAILED)
            return self._hold(obs)

        _, theta_des = self._eef_target_for_nut(self.peg_xy, self.psi_des)
        eef_xy = self._nut_target_eef_xy(obs, self.peg_xy)
        target = np.array([eef_xy[0], eef_xy[1], self.SAFE_Z])

        nut_err = np.linalg.norm(obs["SquareNut_pos"][:2] - self.peg_xy)
        yaw_err = abs(wrap(theta_des - yaw_of(obs["robot0_eef_quat"]), np.pi))
        tilt = self._nut_tilt(obs)
        # gate on CLEARANCE, not on arriving at an exact height -- SAFE_Z is an
        # aim point and the arm need not reach it, only get the nut over the peg
        clear = obs["SquareNut_pos"][2] - 0.01 > self.PEG_TIP_Z + 0.005

        # descending misaligned OR tilted wedges the nut instead of sliding it on
        if self._settled(nut_err < self.TRANSPORT_XY_TOL and yaw_err < self.YAW_TOL
                         and clear and tilt < self.TILT_TOL):
            self._advance(Phase.INSERT)

        action = self._servo_to(target, theta_des, obs)
        action[0:3] = np.clip(action[0:3], -self.TRANSPORT_CAP, self.TRANSPORT_CAP)
        action[3:5] = self._level_nut(obs)
        return np.clip(action, -1.0, 1.0)

    def _do_insert(self, obs):
        if self._lost_grasp():
            self._advance(Phase.FAILED)
            return self._hold(obs)

        # Height is the primary seating test -- SquareNut_pos is an obs key, so
        # using it keeps the controller obs-only. Force alone is not enough: a nut
        # binding on the peg rim spikes the sensor while still 6 cm too high, and
        # releasing there drops the nut rather than seating it.
        nut_z = obs["SquareNut_pos"][2]
        nut_xy_err = np.linalg.norm(obs["SquareNut_pos"][:2] - self.peg_xy)
        f = self._world_force(obs)
        seated = nut_z < self.SEATED_Z + self.SEATED_TOL and nut_xy_err < self.XY_COARSE
        jammed_home = abs(f[2]) > self.SEAT_FORCE_THRESH and nut_z < self.SEATED_Z + 0.03

        if self._settled(seated or jammed_home):
            self.gripper_state = -1.0      # open
            self._advance(Phase.RELEASE)
            return self._hold(obs)
        return self._servo_insert(obs)

    def _do_release(self, obs):
        # hold the pose: drifting while the fingers open drags the nut off the peg
        if self.phase_steps >= self.GRIPPER_SETTLE:
            self._advance(Phase.RETREAT)
        return self._hold(obs)

    def _do_retreat(self, obs):
        # straight up first -- moving laterally runs the open fingers through the nut
        eef = obs["robot0_eef_pos"]
        target = np.array([eef[0], eef[1], self.SAFE_Z])
        theta = yaw_of(obs["robot0_eef_quat"])

        if self.phase_steps >= self.RETREAT_STEPS or eef[2] > self.SAFE_Z - self.Z_TOL:
            self._advance(Phase.DONE)
        return self._servo_to(target, theta, obs)

    def _do_done(self, obs):
        return self._hold(obs)

    def _do_failed(self, obs):
        return self._hold(obs)
