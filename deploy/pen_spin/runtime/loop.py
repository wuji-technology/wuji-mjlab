# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""The control loop — the only code that runs every tick.

The control itself is three pure pieces, checkable without a hand, a viewer, or
a camera stream:

    LowPass       finite-difference velocity smoothing
    ActionMap     raw policy output -> joint target (residual, clip, EMA)
    tick()        one full observe->infer->command step

Three states: (1) waiting — hand frozen until the operator commits,
(2) hold — the policy runs on the clip's first frame, clip clock stopped,
(3) policy — the policy runs and the clip plays.
Starting inference resets the EMA and velocity filter; hold -> policy keeps
both, because that hand-over is part of what the policy was trained on.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from lib.config_loader import get_timing
from lib.math_utils import quat_apply

from runtime import obs as obs_mod
from runtime.hand import ACTION_DIM, clamp_to_limits

_log = logging.getLogger("run")


def _event(msg: str) -> None:
    """Console keeps its spacing; the run log gets the line with a time on it."""
    print(msg, flush=True)
    _log.info(" ".join(msg.split()))

# Two absolute timing values come from config/control.yaml's timing block:
#   jv_lowpass_cutoff_hz  The hand exposes no velocity API, so velocity is
#       finite-differenced from position and smoothed; training saw exact qvel.
#       The corner sits above the loop's Nyquist frequency, so the filter is
#       near-passthrough (measured alphas in config/control.yaml).
#   idle_viewer_hz  Viewer refresh while paused; not a control rate.

#   ENTER  ──▶ one step towards POLICY:  FROZEN ─▶ HOLD ─▶ POLICY
#   r      ──▶ reset, hand opens
#
# FROZEN holds the hand at the start pose with the viewer live while the operator
# places the object; nothing is commanded and the clip clock does not advance.
# HOLD is the warmup the policy was trained with (first_frame_hold_* in the
# training command).
FROZEN, HOLD, POLICY = "frozen", "hold", "policy"

# The placement gate on FROZEN -> HOLD. Training ends an episode once the pen is
# this far from the reference (object_position_error / object_orientation_error
# in tasks/pen_spin/config/wuji_hand2/env_cfgs.py); the shaft bound also catches
# a pen placed end for end. The age bound is the longest camera dropout the
# policy was trained through (ObjectPoseCameraModel in mdp/observations.py), so
# a pose cached from before the pen was placed cannot pass the gate.
START_MAX_POS_ERR_M = 0.05
START_MAX_SHAFT_ERR_DEG = 60.0
START_MAX_POSE_AGE_S = 0.8
_SHAFT_AXIS = np.array([1.0, 0.0, 0.0])


def placement_error(live_pose: np.ndarray, ref_pose: np.ndarray) -> tuple[float, float]:
    """(position error in m, shaft direction error in deg) of two palm-local pen poses."""
    live_pose, ref_pose = (np.asarray(pose, dtype=np.float64) for pose in (live_pose, ref_pose))
    live_shaft = quat_apply(live_pose[3:], _SHAFT_AXIS)
    ref_shaft = quat_apply(ref_pose[3:], _SHAFT_AXIS)
    cosine = np.clip(live_shaft @ ref_shaft / (np.linalg.norm(live_shaft) * np.linalg.norm(ref_shaft)), -1.0, 1.0)
    return float(np.linalg.norm(live_pose[:3] - ref_pose[:3])), float(np.degrees(np.arccos(cosine)))


class LowPass:
    """Per-channel 1st-order IIR: y = a*x + (1-a)*y, a = dt/(dt + 1/(2*pi*fc))."""

    def __init__(self, dim: int, cutoff_hz: float, dt: float):
        if cutoff_hz <= 0.0 or dt <= 0.0:
            raise ValueError(f"cutoff and dt must be positive, got {cutoff_hz}, {dt}")
        rc = 1.0 / (2.0 * np.pi * cutoff_hz)
        self._alpha = dt / (dt + rc)
        self._dim = dim
        self._y: np.ndarray | None = None

    def reset(self, value: np.ndarray | None = None) -> None:
        self._y = (np.zeros(self._dim, dtype=np.float32) if value is None
                   else value.astype(np.float32).copy())

    def update(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self._y is None:
            self._y = x.copy()
        else:
            self._y = self._alpha * x + (1.0 - self._alpha) * self._y
        return self._y.copy()


@dataclass
class ActionMap:
    """Raw policy output -> joint target, as a residual on the reference.

    target = reference_qpos + EMA(clip(raw * scale)). The EMA state is the only
    thing carried between ticks, and it is reset on a mode switch so a jump in
    the policy's output does not become a jump at the fingers.
    """

    scale: float
    clip_lo: float
    clip_hi: float
    alpha: float
    prev_residual: np.ndarray = field(
        default_factory=lambda: np.zeros(ACTION_DIM, dtype=np.float32))

    def reset(self) -> None:
        self.prev_residual = np.zeros(ACTION_DIM, dtype=np.float32)

    def step(self, raw: np.ndarray, ref_qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        residual = np.clip(raw * self.scale, self.clip_lo, self.clip_hi).astype(np.float32)
        processed = self.alpha * residual + (1.0 - self.alpha) * self.prev_residual
        self.prev_residual = processed
        return ref_qpos + processed, residual


def pace_to_deadline(deadline: float) -> None:
    """Absolute-time pacing. sleep(dt - elapsed) drifts; a deadline does not."""
    while True:
        slack = deadline - time.perf_counter()
        if slack <= 0:
            return
        if slack > 0.001:
            time.sleep(slack - 0.0005)
        # else spin the last <1 ms for low jitter


@dataclass
class TickResult:
    frame: int
    obs: np.ndarray
    raw: np.ndarray
    residual: np.ndarray
    actual: np.ndarray
    # 7-D wrist-local object pose observed live this tick; None when the object
    # came from the clip (--obj-from-ref).
    obj_local: np.ndarray | None = None
    # The reference the residual was added to.
    reference_qpos: np.ndarray | None = None


class ControlLoop:
    """Observe -> infer -> command, at the checkpoint's own control rate."""

    def __init__(self, session):
        s = session
        self.session = s
        self.loaded = s.loaded
        self.motion = s.motion
        self.hand = s.hand
        self.objpose = s.objpose
        self.viewer = s.viewer
        self.open_loop = s.mode.open_loop
        self.obj_from_ref = not s.mode.needs_camera

        loaded = s.loaded
        timing = get_timing()
        self.ctrl_dt = s.ctrl_dt
        self.idle_dt = 1.0 / timing["idle_viewer_hz"]
        self.action = ActionMap(loaded.action_scale, loaded.clip_lo,
                                loaded.clip_hi, loaded.alpha)
        self.jv = LowPass(ACTION_DIM, timing["jv_lowpass_cutoff_hz"], self.ctrl_dt)
        # Age of the pen sample the last tick used; telemetry for the status line.
        self.obj_age_s: float | None = None

        self.lower, self.upper = (np.asarray(self.hand.read_joint_lower_limit()),
                                  np.asarray(self.hand.read_joint_upper_limit()))
        self.state = FROZEN
        self.last_raw = np.zeros(ACTION_DIM, dtype=np.float32)
        self.prev_qpos = np.zeros(ACTION_DIM, dtype=np.float32)
        self.step_count = 0
        self.prev_time: float | None = None
        # Which reference frame the idle viewer shows. 0 before the run starts;
        # the last executed frame if we ever pause mid-clip.
        self.idle_frame = 0

    # ----- state ------------------------------------------------------------

    def enter_state(self, state: str, qpos: np.ndarray) -> None:
        """Go to ``state``. Idempotent — re-pressing a key changes nothing.

        Clears the EMA and the velocity filter on every real change, so entering
        or leaving inference cannot inject a step into either. HOLD -> POLICY is
        the exception: inference is already running, and only the clip starts.
        """
        if state not in (FROZEN, HOLD, POLICY):
            raise ValueError(f"unsupported control state: {state!r}")
        if state == self.state:
            return
        started_clip = self.state == HOLD and state == POLICY
        self.state = state
        if started_clip:
            return
        self.action.reset()
        self.jv.reset(np.zeros(ACTION_DIM, dtype=np.float32))
        self.prev_qpos = np.asarray(qpos, dtype=np.float32).copy()
        self.last_raw = np.zeros(ACTION_DIM, dtype=np.float32)
        self.prev_time = None

    # ----- one tick ---------------------------------------------------------

    def tick(self, ctrl, now: float) -> TickResult:
        """One control step. Always steps the policy and always commands."""
        if self.state not in (HOLD, POLICY):
            raise RuntimeError("policy control requires HOLD or POLICY state")
        # 1. object pose — from the clip, or the latest live sample
        if self.obj_from_ref:
            obj_pos = obj_quat = None
            self.obj_age_s = None
        else:
            snap = self.objpose.snapshot()
            self.obj_age_s = snap.get("pen_age_s")
            if not self.objpose.has_pose(snap):
                # Nothing has ever arrived; the receiver's identity default
                # would put a fabricated pen in the observation.
                raise RuntimeError(
                    "no pen pose has ever arrived on the observer stream — "
                    "start scripts/pen_world_observer.py, or run --obj-from-ref")
            obj_pos, obj_quat = self.objpose.local(snap)

        # 2. joint state from the SDK's cached upstream — never a bus round-trip
        #    in the hot path (15-30 ms blocking vs sub-ms cache hit).
        qpos = np.asarray(ctrl.get_joint_actual_position()).flatten().astype(np.float32)
        dt = self.ctrl_dt if self.prev_time is None else max(now - self.prev_time, 1e-6)
        jv = self.jv.update((qpos - self.prev_qpos) / dt)
        self.prev_time = now
        self.prev_qpos = qpos

        # 3. advance the clip clock (HOLD stays on the first frame) and assemble
        #    the observation
        k = self.idle_frame if self.state == HOLD else self.motion.step(now=now)
        self.idle_frame = k
        observation = obs_mod.build_observation(
            self.motion, k, qpos, jv, obj_pos, obj_quat,
            last_raw_action=self.last_raw,
            open_loop=self.open_loop,
            obj_from_ref=self.obj_from_ref,
        )
        ref_qpos = self.motion.reference_qpos[k]

        # 4. Every active step uses the policy; reference joints are its baseline.
        raw = self.loaded.infer(observation)
        self.last_raw = raw  # the obs term is the RAW output, not the residual
        target, residual = self.action.step(raw, ref_qpos)

        if not np.isfinite(target).all():
            raise FloatingPointError("non-finite policy joint target")
        sent = clamp_to_limits(target, self.lower, self.upper)
        ctrl.set_joint_target_position(sent)
        self.step_count += 1
        live = (None if obj_pos is None
                else np.concatenate([np.asarray(obj_pos, dtype=np.float64),
                                     np.asarray(obj_quat, dtype=np.float64)]))
        return TickResult(k, observation, raw, residual, qpos,
                          obj_local=live, reference_qpos=np.asarray(ref_qpos))

    def placement(self) -> tuple[str, str | None] | None:
        """(status text, reason the start is refused or None) for the placed pen.

        Compares the live pen with the reference frame the hold will run on.
        None when the object is not sensed (--obj-from-ref): nothing to place.
        """
        if self.obj_from_ref or self.objpose is None:
            return None
        snap = self.objpose.snapshot()
        if not self.objpose.has_pose(snap):
            return "pen: not yet detected by camera", "the camera has not detected the pen yet"
        pos, quat = self.objpose.local(snap)
        pos_err, shaft_err = placement_error(np.concatenate([pos, quat]),
                                             self.motion.object_pose_local[self.idle_frame])
        age = snap.get("pen_age_s")
        text = f"pen: position error {pos_err * 1000:.0f} mm  shaft angle error {shaft_err:.0f} deg"
        if age is not None and age > START_MAX_POSE_AGE_S:
            return f"{text}  (pose has not updated for {age:.1f} s)", "the camera cannot see the pen; the values above are stale"
        if shaft_err > 180.0 - START_MAX_SHAFT_ERR_DEG:
            return text, "the pen is reversed; turn it around and place it again"
        if shaft_err > START_MAX_SHAFT_ERR_DEG:
            return text, f"shaft differs from the reference by {shaft_err:.0f} deg, exceeding the trained range of {START_MAX_SHAFT_ERR_DEG:.0f} deg"
        if pos_err > START_MAX_POS_ERR_M:
            return text, (f"pen is {pos_err * 1000:.0f} mm from the reference position, "
                          f"exceeding the trained range of {START_MAX_POS_ERR_M * 1000:.0f} mm")
        return text, None

    def tick_idle(self, ctrl) -> None:
        """One pre-start / paused iteration. Commands nothing, advances nothing.

        The operator is placing the object into the frozen hand, so the viewer
        shows live joint feedback and the live object pose.
        """
        dt = self.idle_dt
        if self.viewer is None or not self.viewer.is_running():
            time.sleep(dt)
            return
        qpos = np.asarray(ctrl.get_joint_actual_position()).flatten().astype(np.float32)
        live = None
        if self.objpose is not None:
            snap = self.objpose.snapshot()
            # Placing the object occludes its markers; the receiver holds the
            # last valid sample, so the pen lags rather than disappears.
            if self.objpose.has_pose(snap):
                pos, quat = self.objpose.local(snap)
                live = np.concatenate([np.asarray(pos, dtype=np.float64),
                                       np.asarray(quat, dtype=np.float64)])
        self.viewer.update(TickResult(
            frame=self.idle_frame,
            obs=np.zeros(1, dtype=np.float32),
            raw=np.zeros(ACTION_DIM, dtype=np.float32),
            residual=np.zeros(ACTION_DIM, dtype=np.float32),
            actual=qpos,
            obj_local=live,
        ), self.motion)
        time.sleep(dt)

    # ----- driver -----------------------------------------------------------

    def run(self, ctrl, *, keys=None) -> bool:
        """Run until the clip ends or a reset key. True = reset asked.

        ``keys`` is any object with ``.get() -> str | None`` (runtime.safety.Keys
        merges the terminal and the viewer window). The loop begins in FROZEN and
        waits for Enter: the first starts the policy on the held first frame
        (HOLD), the second starts the clip.
        """
        self.prev_qpos = np.asarray(
            ctrl.get_joint_actual_position()).flatten().astype(np.float32)
        self.jv.reset(np.zeros(ACTION_DIM, dtype=np.float32))
        self.motion.start(now=time.perf_counter())
        # A headless run (piped stdin, no window) has nobody to press a key, and
        # holding in FROZEN there would hang.
        watched = keys is not None and (keys.interactive or self.viewer is not None)
        # The clip clock is wall-clock anchored; shift_clock takes the time spent
        # in FROZEN and HOLD back off it.
        clip_stopped_since = time.perf_counter()
        if not watched:
            self.enter_state(POLICY, self.prev_qpos)
            clip_stopped_since = None
        last_print = time.perf_counter()

        while not self.motion.done:
            t0 = time.perf_counter()

            if keys is not None:
                key = keys.get()
                if key in ("r", "R"):
                    return True
                if key in ("\n", "\r"):
                    want = HOLD if self.state == FROZEN else POLICY
                else:
                    want = None
                if want == HOLD:
                    placed = self.placement()
                    if placed is not None and placed[1] is not None:
                        _event(f"\n  [Enter] policy not started: {placed[1]}   ({placed[0]})")
                        want = None
                if want is not None and want != self.state:
                    if clip_stopped_since is not None and want != HOLD:   # the clip starts
                        self.motion.shift_clock(time.perf_counter() - clip_stopped_since)
                        clip_stopped_since = None
                    self.enter_state(
                        want,
                        np.asarray(ctrl.get_joint_actual_position()).flatten())
                    _event("\n  [Enter]"
                           f" -> {self.state}"
                           + ("   (policy is holding the first frame; press Enter again to start motion)" if want == HOLD else ""))

            if self.state == FROZEN:
                self.tick_idle(ctrl)
                if t0 - last_print > 1.0:
                    placed = self.placement()
                    if placed is not None:
                        print(f"  [frozen] {placed[0]}"
                              + (f"   <- {placed[1]}" if placed[1] else "   press Enter when ready"), flush=True)
                    last_print = t0
                continue

            res = self.tick(ctrl, t0)

            if self.viewer is not None:
                self.viewer.update(res, self.motion)

            now = time.perf_counter()
            if now - last_print > 1.0:
                age = ("" if self.obj_age_s is None
                       else f" obj_age={self.obj_age_s * 1000:.0f}ms")
                _event(f"  [{self.state}] step={self.step_count:5d} "
                       f"k={res.frame:4d}/{self.motion.num_frames}  "
                       f"raw_max={np.abs(res.raw).max():.2f} "
                       f"residual_max={np.abs(res.residual).max():.2f}{age}")
                last_print = now

            pace_to_deadline(t0 + self.ctrl_dt)
        return False
