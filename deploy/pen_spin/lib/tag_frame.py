# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Transform wrist-tag poses into the physical Wuji Hand 2 wrist frame.

Positions use metres and quaternions use (w, x, y, z)."""

from __future__ import annotations

import numpy as np

from lib.math_utils import normalize_quat, quat_apply, quat_mul

# Calibrated wrist-tag pose in the physical wrist frame, metres and wxyz.
TAG_IN_PALM_POS = np.array([-0.00300004, 0.037157842, 0.0632369])
TAG_IN_PALM_QUAT = np.array([0.5, -0.5, -0.5, -0.5])

# Physical wrist marker; shared by the camera preview and deployment scene.
WORLD_TAG_ID = 0
WORLD_TAG_SIZE_M = 0.0504  # black-square edge

# OpenCV marker axes need a half-turn about Z to match the calibrated wrist-tag frame.
WRIST_AXES_IN_OPENCV_TAG = np.diag([-1.0, -1.0, 1.0])

# Hand root in the viewer world (palm-up mount). Viewer-only: policy
# observations remain local to the physical hand2 wrist.
ROOT_POS = np.array([0.0, 0.0, 0.5])
ROOT_QUAT = np.array([0.70710678, 0.0, -0.70710678, 0.0])


def pen_in_palm(
  pen_pos_tag: np.ndarray, pen_quat_tag: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Lift a tag-frame pen pose into the hand2 wrist used by the policy.

  Args:
    pen_pos_tag: (3,) pen position in the wrist-tag frame, metres.
    pen_quat_tag: (4,) pen orientation in the wrist-tag frame, (w,x,y,z).

  Returns:
    (pos(3,), quat(4,)) in the hand2 wrist frame — exactly what runtime/obs.py
    consumes as object_local_pos / object_local_quat_wxyz.
  """
  pos = TAG_IN_PALM_POS + quat_apply(
    TAG_IN_PALM_QUAT, np.asarray(pen_pos_tag, dtype=np.float64)
  )
  quat = quat_mul(TAG_IN_PALM_QUAT, np.asarray(pen_quat_tag, dtype=np.float64))
  n = np.linalg.norm(quat)
  if n < 1e-9:
    raise ValueError(f"degenerate pen quaternion: {pen_quat_tag}")
  return pos, quat / n


# ───────── scene-frame helpers ───────────────────────────────────────────
# The functions below are pure SE(3) geometry used by runtime/viz.py to lift a
# palm-local pose into the rendered scene.


def compose(
  a_pos: np.ndarray,
  a_quat: np.ndarray,
  b_pos: np.ndarray,
  b_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """SE(3) compose A∘B (poses as local→world transforms, wxyz quat).

  world = A applied to B:  pos = a_pos + R(a_quat)·b_pos,
                           quat = a_quat ⊗ b_quat.
  """
  a_pos = np.asarray(a_pos, dtype=np.float64)
  b_pos = np.asarray(b_pos, dtype=np.float64)
  a_quat = normalize_quat(a_quat)
  b_quat = normalize_quat(b_quat)
  out_pos = a_pos + quat_apply(a_quat, b_pos)
  out_quat = normalize_quat(quat_mul(a_quat, b_quat))
  return out_pos, out_quat


def lift_wrist_local_to_scene(
  local_pos: np.ndarray,
  local_quat: np.ndarray,
  scene_palm_pos: np.ndarray,
  scene_palm_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Lift a wrist-local pose using the palm pose queried from the viewer model."""
  return compose(scene_palm_pos, scene_palm_quat, local_pos, local_quat)
