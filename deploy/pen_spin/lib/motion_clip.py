# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Load reference clips and assemble wrist-local policy inputs without the training stack."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib.math_utils import (
    normalize_quat,
    quat_apply,
    quat_inv,
    quat_mul,
)

# Joint and link order must match the public clip schema and the training policy.
RIGHT_FINGER_JOINT_NAMES: tuple[str, ...] = tuple(
    f"right_finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5)
)
RIGHT_HAND_LINK_NAMES: tuple[str, ...] = ("right_palm_link",) + tuple(
    f"right_finger{i}_link{j}" for i in range(1, 6) for j in range(1, 5)
)
NUM_JOINTS = len(RIGHT_FINGER_JOINT_NAMES)  # 20
NUM_LINKS = len(RIGHT_HAND_LINK_NAMES)  # 21

# Match SingleHandTrackingCommandCfg.axis_point_radius_m on the training side.
DEFAULT_AXIS_POINT_RADIUS = 0.10

REQUIRED_KEYS = (
    "mediapipe_landmarks_w",
    "reference_qpos",
    "wrist_pose_w",
    "object_pose_w",
    "link_pos_w",
)

DEFAULT_OBJECT_POSITION_TANH_SCALE_M = 0.15


def _world_pose_to_wrist_local(
    pos_w: np.ndarray,
    quat_w: np.ndarray,
    wrist_pos: np.ndarray,
    wrist_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    wrist_quat = normalize_quat(wrist_quat)
    quat_w = normalize_quat(quat_w)
    inv = quat_inv(wrist_quat)
    local_pos = quat_apply(inv, pos_w - wrist_pos)
    local_quat = normalize_quat(quat_mul(inv, quat_w))
    return local_pos, local_quat


def _world_points_to_wrist_local(
    points_w: np.ndarray,  # (T, P, 3) or (P, 3)
    wrist_pos: np.ndarray,  # (T, 3) or (3,)
    wrist_quat: np.ndarray,  # (T, 4) or (4,)
) -> np.ndarray:
    wrist_quat = normalize_quat(wrist_quat)
    inv = quat_inv(wrist_quat)
    if points_w.ndim == 3:
        inv_broadcast = inv[:, None, :]
        return quat_apply(inv_broadcast, points_w - wrist_pos[:, None, :])
    return quat_apply(inv, points_w - wrist_pos)


def make_axis_points(radius: float = DEFAULT_AXIS_POINT_RADIUS) -> np.ndarray:
    """2 ±X shaft end points used to represent pen pose as 6-D vector."""
    return np.array(
        [
            [+radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


def object_pose_to_points_wrist_local(
    object_pose: np.ndarray,  # (..., 7) pos+quat in wrist-local frame
    axis_points: np.ndarray,  # (P, 3) template
) -> np.ndarray:
    """Lift the (P, 3) axis template onto the cube at object_pose, in wrist local.

    Output shape: (..., P, 3). Use .reshape(..., -1) -> 6-D for obs.
    """
    pos = object_pose[..., :3]
    quat = object_pose[..., 3:7]
    num_points = axis_points.shape[0]
    expanded_quat = np.broadcast_to(
        quat[..., None, :], pos.shape[:-1] + (num_points, 4)
    )
    expanded_points = np.broadcast_to(axis_points, pos.shape[:-1] + (num_points, 3))
    rotated = quat_apply(expanded_quat, expanded_points)
    return pos[..., None, :] + rotated


def bounded_object_pose_to_points_wrist_local(
    object_pose: np.ndarray,
    axis_points: np.ndarray,
    *,
    position_tanh_scale_m: float = DEFAULT_OBJECT_POSITION_TANH_SCALE_M,
) -> np.ndarray:
    """Training-exact bounded axis-point representation used by pen-spin."""
    if position_tanh_scale_m <= 0.0:
        raise ValueError("position_tanh_scale_m must be positive")
    points = object_pose_to_points_wrist_local(object_pose, axis_points)
    return position_tanh_scale_m * np.tanh(points / position_tanh_scale_m)


# ---------------------------------------------------------------------------
# Clip loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MotionClipData:
    """Pre-baked tensors for a single motion clip."""

    reference_qpos: np.ndarray  # (T, 20) float32
    reference_qvel: np.ndarray  # (T, 20) float32 — finite-diff
    object_pose_local: np.ndarray  # (T, 7)  pos + quat (wxyz) in wrist local
    axis_points: np.ndarray  # (2, 3)  template
    bounded_object_axis_points_local: np.ndarray  # (T, 6) reference command
    link_pos_local: np.ndarray  # (T, 21, 3)
    num_frames: int
    fps: float
    frame_dt: float
    source_path: Path


def _validate_required_keys(data) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in data.files]
    if missing:
        raise KeyError(f"motion clip is missing keys: {missing}")


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    vel = np.zeros_like(values)
    if values.shape[0] <= 1:
        return vel
    step = (values[1:] - values[:-1]) / dt
    vel[:-1] = step
    vel[-1] = step[-1]
    return vel


def load_motion_clip(
    npz_path: str | Path,
    *,
    fps: float = 50.0,
    axis_point_radius: float = DEFAULT_AXIS_POINT_RADIUS,
) -> MotionClipData:
    """Load NPZ clip and pre-compute wrist-local representations."""
    path = Path(npz_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"motion clip not found: {path}")
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    frame_dt = 1.0 / fps

    with np.load(path, allow_pickle=False) as data:
        _validate_required_keys(data)
        reference_qpos = np.asarray(data["reference_qpos"], dtype=np.float32)
        wrist_pose_w = np.asarray(data["wrist_pose_w"], dtype=np.float32)
        object_pose_w = np.asarray(data["object_pose_w"], dtype=np.float32)
        link_pos_w = np.asarray(data["link_pos_w"], dtype=np.float32)

    T, J = reference_qpos.shape
    if J != NUM_JOINTS:
        raise ValueError(f"reference_qpos has {J} joints, expected {NUM_JOINTS}")
    if T <= 0:
        raise ValueError("clip has zero frames")

    if wrist_pose_w.shape != (T, 7):
        raise ValueError(f"wrist_pose_w shape {wrist_pose_w.shape}, expected ({T}, 7)")
    if object_pose_w.shape != (T, 7):
        raise ValueError(
            f"object_pose_w shape {object_pose_w.shape}, expected ({T}, 7)"
        )
    if link_pos_w.shape != (T, NUM_LINKS, 3):
        raise ValueError(
            f"link_pos_w shape {link_pos_w.shape}, expected ({T}, {NUM_LINKS}, 3)"
        )

    # Promote to float64 for stable transforms; cast back to float32 at obs time.
    wrist_pos_w = wrist_pose_w[:, :3].astype(np.float64)
    wrist_quat_w = normalize_quat(wrist_pose_w[:, 3:7].astype(np.float64))
    object_pos_w = object_pose_w[:, :3].astype(np.float64)
    object_quat_w = normalize_quat(object_pose_w[:, 3:7].astype(np.float64))

    # Per-frame world -> wrist-local for object pose.
    object_local_pos, object_local_quat = _world_pose_to_wrist_local(
        object_pos_w, object_quat_w, wrist_pos=wrist_pos_w, wrist_quat=wrist_quat_w
    )
    object_pose_local = np.concatenate(
        [object_local_pos, object_local_quat], axis=-1
    ).astype(np.float32)

    # 21 link positions in wrist-local frame.
    link_pos_local = _world_points_to_wrist_local(
        link_pos_w.astype(np.float64),
        wrist_pos=wrist_pos_w,
        wrist_quat=wrist_quat_w,
    ).astype(np.float32)

    # Axis point representation of cube pose in wrist-local: (T, 2, 3) -> (T, 6).
    axis_points = make_axis_points(axis_point_radius)
    bounded_object_axis_points_local = (
        bounded_object_pose_to_points_wrist_local(object_pose_local, axis_points)
        .reshape(T, -1)
        .astype(np.float32)
    )

    # qvel via simple finite-diff of reference_qpos.
    reference_qvel = _finite_difference(reference_qpos.astype(np.float32), frame_dt)

    return MotionClipData(
        reference_qpos=reference_qpos,
        reference_qvel=reference_qvel,
        object_pose_local=object_pose_local,
        axis_points=axis_points.astype(np.float32),
        bounded_object_axis_points_local=bounded_object_axis_points_local,
        link_pos_local=link_pos_local,
        num_frames=int(T),
        fps=float(fps),
        frame_dt=float(frame_dt),
        source_path=path,
    )


# ---------------------------------------------------------------------------
# MotionClip — runtime wrapper with wall-clock-anchored frame indexing
# ---------------------------------------------------------------------------


class MotionClip:
    """Runtime motion clip player for the deploy control loop.

    Indexing is anchored to wall-clock at start_now() so 50 Hz tick drift
    cannot accumulate over a 60 s clip. step(now) -> int frame_idx in
    [0, num_frames - 1], clamped at the end, with done=True after the last
    frame is read once. With hold_last_frame, it keeps returning that frame
    with zero reference velocity and done=False.

    Wall-clock semantics stay inside the clip player rather than the policy
    bookkeeping.
    """

    def __init__(
        self,
        npz_path: str | Path,
        *,
        fps: float = 50.0,
        axis_point_radius: float = DEFAULT_AXIS_POINT_RADIUS,
        hold_last_frame: bool = False,
    ):
        self._data = load_motion_clip(
            npz_path, fps=fps, axis_point_radius=axis_point_radius
        )
        self.hold_last_frame = bool(hold_last_frame)
        if self.hold_last_frame:
            # The infinite tail is a stationary reference. Do not repeat the
            # final moving-frame derivative.
            self._data.reference_qvel[-1] = 0.0
        self._t0: float | None = None
        self._done: bool = False

    # ----- properties ---------------------------------------------------------

    @property
    def num_frames(self) -> int:
        return self._data.num_frames

    @property
    def done(self) -> bool:
        return self._done

    @property
    def reference_qpos(self) -> np.ndarray:
        return self._data.reference_qpos

    @property
    def reference_qvel(self) -> np.ndarray:
        return self._data.reference_qvel

    @property
    def object_pose_local(self) -> np.ndarray:
        return self._data.object_pose_local

    @property
    def axis_points(self) -> np.ndarray:
        return self._data.axis_points

    # ----- runtime API --------------------------------------------------------

    def start(self, now: float | None = None) -> None:
        """Anchor the wall-clock origin; call right before the first step."""
        self._t0 = now if now is not None else time.perf_counter()
        self._done = False

    def shift_clock(self, delta_s: float) -> None:
        """Advance the internal wall-clock anchor by ``delta_s`` seconds.

        Use this to absorb a pause / stall in the consumer loop without
        fast-forwarding the clip: shift t0 forward by the pause duration so
        the next ``step(now)`` returns the same frame the pause started on.
        """
        if self._t0 is None:
            raise RuntimeError("MotionClip.shift_clock() before start()")
        self._t0 += float(delta_s)

    def step(self, now: float | None = None) -> int:
        """Return the integer frame index at wall-clock `now`.

        Wall-clock anchored: ``k = round((now - t0) / frame_dt)``. Without this
        anchoring, summing ctrl_dt per tick drifts (50 Hz tick at 19.8 ms over
        60 s = 1.2 s offset; motion clip would be 60 frames behind).
        """
        if self._t0 is None:
            raise RuntimeError("MotionClip.start() must be called before step()")
        if now is None:
            now = time.perf_counter()
        raw_k = int(round((now - self._t0) / self._data.frame_dt))
        if raw_k >= self._data.num_frames:
            self._done = not self.hold_last_frame
            return self._data.num_frames - 1
        return max(0, raw_k)

    # ----- obs assembly -------------------------------------------------------

    def get_current_command_term(self, k: int) -> np.ndarray:
        """89-D command used by the fixed-wrist pen-spin policy.

        Matches ``SingleHandTrackingCommand.current_command``: current
        reference qpos (20), bounded reference object axis points (6), and
        current wrist-local link positions (63). There is no lookahead.
        """
        k = max(0, min(int(k), self._data.num_frames - 1))
        return np.concatenate(
            [
                self._data.reference_qpos[k],
                self._data.bounded_object_axis_points_local[k],
                self._data.link_pos_local[k].reshape(-1),
            ]
        ).astype(np.float32)
