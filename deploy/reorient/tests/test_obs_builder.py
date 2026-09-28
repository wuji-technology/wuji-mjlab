# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import numpy as np
from wuji_reorient_deploy.constants import load_constants
from wuji_reorient_deploy.math_utils import quat_inv, quat_mul, rot6d_from_quat
from wuji_reorient_deploy.obs_builder import ActionPostprocessor, ObsAssembler


def _term_slices(c):
  out, off = {}, 0
  for name, dim in c.obs_terms:
    width = dim * c.history_length
    out[name] = (off, off + width)
    off += width
  return out


def test_obs_assembler_layout_backfill_and_ori_error():
  c = load_constants(2, "right")
  asm = ObsAssembler(c)
  n = c.num_joints

  joint_pos = c.default_joint_pos.copy()
  cube_pos_tag = np.array([0.01, -0.02, 0.03])
  cube_quat_tag = np.array([0.9238795, 0.3826834, 0.0, 0.0])
  goal_quat_tag = np.array([1.0, 0.0, 0.0, 0.0])
  prev_raw_action = np.linspace(-1.0, 1.0, n)

  obs = asm.build(
    joint_pos=joint_pos,
    target=joint_pos,
    cube_pos_tag=cube_pos_tag,
    cube_quat_tag=cube_quat_tag,
    goal_quat_tag=goal_quat_tag,
    prev_raw_action=prev_raw_action,
  )
  assert obs.shape == (c.obs_dim,)

  sl = _term_slices(c)
  H = c.history_length

  lo, hi = sl["qpos_error"]
  np.testing.assert_allclose(obs[lo:hi], 0.0, atol=1e-6)

  expected_ori = rot6d_from_quat(quat_mul(cube_quat_tag, quat_inv(goal_quat_tag)))
  lo, hi = sl["cube_ori_error"]
  block = obs[lo:hi].reshape(H, -1)
  for frame in block:
    np.testing.assert_allclose(frame, expected_ori, atol=1e-5)

  lo, hi = sl["action_history"]
  block = obs[lo:hi].reshape(H, n)
  for frame in block:
    np.testing.assert_allclose(frame, prev_raw_action, atol=1e-6)

  lo, hi = sl["noisy_joint_angles"]
  ja = obs[lo:hi].reshape(H, n)
  for frame in ja[1:]:
    np.testing.assert_array_equal(frame, ja[0])


def test_action_postprocessor_warmup_hold_then_ema():
  c = load_constants(2, "right")
  scale, ema, warmup, dt = 0.5, 0.5, 0.1, 0.05
  post = ActionPostprocessor(
    c, action_scale=scale, ema_alpha=ema, warmup_time_s=warmup, ctrl_dt=dt
  )
  a = np.ones(c.num_joints)

  for _ in range(2):
    np.testing.assert_array_equal(post.process(a), c.default_joint_pos)

  raw_target = np.clip(
    c.default_joint_pos + np.clip(a, -1, 1) * scale, c.soft_lower, c.soft_upper
  )
  expected = ema * raw_target + (1.0 - ema) * c.default_joint_pos
  np.testing.assert_allclose(post.process(a), expected, atol=1e-9)
