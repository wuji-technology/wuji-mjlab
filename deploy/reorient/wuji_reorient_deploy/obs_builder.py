# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""207-dim observation assembly + action post-processing, in plain numpy.

  policy group (term-major, history flattened oldest→newest, H=3):
    noisy_joint_angles (20) : clamp((q-center)/(half+1e-6), -1, 1)   [noise off in deploy]
    qpos_error         (20) : norm(q) - norm(target)
    cube_pos_in_tag    (3)  : cube position in tag frame            [from observer]
    cube_ori_error     (6)  : rot6d(cube_quat_tag ⊗ goal_quat_tag⁻¹) [from observer + goal]
    action_history     (20) : previous raw policy action
  → 3 × (20+20+3+6+20) = 207
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from .constants import Constants
from .math_utils import quat_inv, quat_mul, rot6d_from_quat


def _normalize_joints(q: np.ndarray, soft_center: np.ndarray,
                       soft_half_range: np.ndarray) -> np.ndarray:
  return np.clip((q - soft_center) / (soft_half_range + 1e-6), -1.0, 1.0)


def warmup_steps(warmup_time_s: float, ctrl_dt: float) -> int:
  """Control steps to hold at the default pose, as an exact integer count."""
  if ctrl_dt <= 0.0:
    raise ValueError(f"ctrl_dt must be positive, got {ctrl_dt}")
  return max(0, math.ceil(warmup_time_s / ctrl_dt - 1e-9))


class ActionPostprocessor:
  """Convert policy actions into smoothed, bounded joint targets."""

  def __init__(self, constants: Constants, action_scale: float, ema_alpha: float,
               warmup_time_s: float, ctrl_dt: float):
    """Initialize action post-processing state."""
    self.constants = constants
    self.action_scale = float(action_scale)
    self.ema_alpha = float(ema_alpha)
    self.warmup_time_s = float(warmup_time_s)
    self.ctrl_dt = float(ctrl_dt)
    self.warmup_steps = warmup_steps(warmup_time_s, ctrl_dt)
    self._prev_target = constants.default_joint_pos.copy()
    self._step = 0

  @property
  def default_target(self) -> np.ndarray:
    """Return a copy of the hand's default joint target."""
    return self.constants.default_joint_pos.copy()

  def process(self, raw_action: np.ndarray) -> np.ndarray:
    """Convert one raw policy action into a commanded joint target."""
    c = self.constants
    clamped = np.clip(raw_action, -1.0, 1.0)
    raw_target = np.clip(
      c.default_joint_pos + clamped * self.action_scale, c.soft_lower, c.soft_upper
    )
    smoothed = self.ema_alpha * raw_target + (1.0 - self.ema_alpha) * self._prev_target
    in_warmup = self._step < self.warmup_steps
    target = c.default_joint_pos.copy() if in_warmup else smoothed
    self._prev_target = target.copy()
    self._step += 1
    return target


class ObsAssembler:
  """Maintain per-term history and emit the flattened policy observation."""

  def __init__(self, constants: Constants):
    """Initialize empty observation histories."""
    self.constants = constants
    self._hist = {name: deque(maxlen=constants.history_length)
                  for name, _ in constants.obs_terms}

  def _push_term(self, name: str, vec: np.ndarray) -> None:
    dq = self._hist[name]
    if not dq:  # backfill entire history with the first frame (matches CircularBuffer)
      for _ in range(self.constants.history_length):
        dq.append(vec.copy())
    else:
      dq.append(vec.copy())

  def build(
    self,
    joint_pos: np.ndarray,
    target: np.ndarray,        # (20,) commanded target just applied (for qpos_error)
    cube_pos_tag: np.ndarray,
    cube_quat_tag: np.ndarray, # (4,)  cube orientation in tag frame (w,x,y,z)
    goal_quat_tag: np.ndarray, # (4,)  goal orientation in tag frame (w,x,y,z)
    prev_raw_action: np.ndarray,
  ) -> np.ndarray:
    """Assemble one flattened observation from the current control state."""
    c = self.constants
    q_err = quat_mul(cube_quat_tag, quat_inv(goal_quat_tag))
    terms = {
      "noisy_joint_angles": _normalize_joints(joint_pos, c.soft_center, c.soft_half_range),
      "qpos_error": (_normalize_joints(joint_pos, c.soft_center, c.soft_half_range)
                     - _normalize_joints(target, c.soft_center, c.soft_half_range)),
      "cube_pos_in_tag": np.asarray(cube_pos_tag, dtype=np.float64),
      "cube_ori_error": rot6d_from_quat(q_err),
      "action_history": np.asarray(prev_raw_action, dtype=np.float64),
    }
    for name, _ in c.obs_terms:
      self._push_term(name, terms[name])
    parts = [np.concatenate(list(self._hist[name])) for name, _ in c.obs_terms]
    obs = np.concatenate(parts).astype(np.float32)
    assert obs.shape == (c.obs_dim,), obs.shape
    return obs
