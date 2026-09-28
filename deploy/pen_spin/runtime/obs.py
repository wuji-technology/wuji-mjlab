# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Exact observation assembly for the fixed-wrist pen-spin policy.

There is one supported layout:

    current motion command                  89
    joint_pos_rel (training default = 0)    20
    joint_vel * 0.25                        20
    bounded live-object axis points          6
    previous raw action                     20
                                            ---
                                            155
"""

from __future__ import annotations

import numpy as np

from lib.motion_clip import bounded_object_pose_to_points_wrist_local
from lib.onnx_policy import OBS_DIM


def build_observation(
    motion,
    frame_idx: int,
    finger_qpos: np.ndarray,
    finger_qvel: np.ndarray,
    object_local_pos: np.ndarray | None,
    object_local_quat_wxyz: np.ndarray | None,
    *,
    last_raw_action: np.ndarray,
    open_loop: bool = False,
    obj_from_ref: bool = False,
) -> np.ndarray:
    """Build the single 155-D observation used by pen-spin training."""
    k = max(0, min(int(frame_idx), motion.num_frames - 1))
    motion_term = motion.get_current_command_term(k)

    if open_loop:
        joint_pos = np.asarray(motion.reference_qpos[k], dtype=np.float32)
        joint_vel = np.asarray(motion.reference_qvel[k], dtype=np.float32)
    else:
        joint_pos = np.asarray(finger_qpos, dtype=np.float32)
        joint_vel = np.asarray(finger_qvel, dtype=np.float32)

    if obj_from_ref:
        object_pose = np.asarray(motion.object_pose_local[k], dtype=np.float64)
    else:
        if object_local_pos is None or object_local_quat_wxyz is None:
            raise ValueError("live pen pose is required unless obj_from_ref=True")
        object_pose = np.concatenate(
            [
                np.asarray(object_local_pos, dtype=np.float64),
                np.asarray(object_local_quat_wxyz, dtype=np.float64),
            ]
        )

    object_points = (
        bounded_object_pose_to_points_wrist_local(object_pose, motion.axis_points)
        .reshape(-1)
        .astype(np.float32)
    )
    obs = np.concatenate(
        [
            motion_term,
            joint_pos,  # joint_pos_rel: fixed task's default joint position is zero
            joint_vel * 0.25,
            object_points,
            np.asarray(last_raw_action, dtype=np.float32),
        ]
    ).astype(np.float32)
    if obs.shape != (OBS_DIM,) or not np.isfinite(obs).all():
        raise RuntimeError(f"pen-spin observation must be finite and {OBS_DIM}D, got {obs.shape}")
    return obs
