# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import mujoco
import numpy as np
from wuji_mjlab.tasks.reorient.config.wuji_hand.env_cfgs import (
  get_wuji_hand_rig_cfg,
  wuji_hand_reorient_env_cfg,
)
from wuji_mjlab.tasks.reorient.config.wuji_hand2.env_cfgs import (
  wuji_hand2_reorient_env_cfg,
)
from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
  REORIENT_HAND2_PALM_NORMAL_AXIS,
)
from wuji_mjlab.tasks.reorient.reorient_constants import (
  REORIENT_PALM_NORMAL_AXIS,
  REORIENT_WRIST_TAG_POS,
  REORIENT_WRIST_TAG_ROT,
)
from wuji_reorient_deploy.constants import load_constants


def test_hand1_wrist_tag_site_matches_reorient_rig_pose():
  model = get_wuji_hand_rig_cfg("right").spec_fn().compile()
  tag = model.site("right_wrist_tag")
  assert model.site_bodyid[tag.id] == model.body("right_palm_link").id
  np.testing.assert_allclose(tag.pos, REORIENT_WRIST_TAG_POS, atol=1e-6)
  expected_quat = np.asarray(REORIENT_WRIST_TAG_ROT)
  assert np.allclose(tag.quat, expected_quat, atol=1e-6) or np.allclose(
    tag.quat, -expected_quat, atol=1e-6
  )


def test_deploy_hand1_tag_pose_matches_training_scene():
  deploy = load_constants(1, "right")
  np.testing.assert_allclose(deploy.tag_in_palm_pos, REORIENT_WRIST_TAG_POS, atol=1e-6)
  expected_quat = np.asarray(REORIENT_WRIST_TAG_ROT)
  assert np.allclose(deploy.tag_in_palm_quat, expected_quat, atol=1e-6) or np.allclose(
    deploy.tag_in_palm_quat, -expected_quat, atol=1e-6
  )


def _hand2_model() -> mujoco.MjModel:
  cfg = wuji_hand2_reorient_env_cfg(num_envs=1)
  return cfg.scene.entities["robot"].spec_fn().compile()


def _tag_z_in_palm(model: mujoco.MjModel) -> np.ndarray:
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  site_id = model.site("right_wrist_tag").id
  palm_id = model.body("right_palm_link").id
  palm_rotation = data.xmat[palm_id].reshape(3, 3)
  tag_rotation = data.site_xmat[site_id].reshape(3, 3)
  return palm_rotation.T @ tag_rotation[:, 2]


def test_tag_z_identifies_each_hand_palm_normal_axis():
  hand1_model = get_wuji_hand_rig_cfg("right").spec_fn().compile()
  hand2_model = _hand2_model()

  hand1_axis = int(np.argmax(np.abs(_tag_z_in_palm(hand1_model))))
  hand2_axis = int(np.argmax(np.abs(_tag_z_in_palm(hand2_model))))

  assert hand1_axis == REORIENT_PALM_NORMAL_AXIS
  assert hand2_axis == REORIENT_HAND2_PALM_NORMAL_AXIS


def test_palm_normal_axis_drives_cage_and_height_metric():
  for cfg, expected_axis in (
    (wuji_hand_reorient_env_cfg(num_envs=1), REORIENT_PALM_NORMAL_AXIS),
    (wuji_hand2_reorient_env_cfg(num_envs=1), REORIENT_HAND2_PALM_NORMAL_AXIS),
  ):
    assert cfg.rewards["cage_escape"].params["up_axis"] == expected_axis
    assert (
      cfg.metrics["cube_height_above_palm"].params["palm_normal_axis"] == expected_axis
    )


def test_hand2_right_has_wrist_tag_site():
  m = _hand2_model()
  assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_wrist_tag") >= 0


def _hand2_tag_pose_in_palm() -> tuple[np.ndarray, np.ndarray]:
  m = _hand2_model()
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_wrist_tag")
  pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_palm_link")
  palm_pos = d.xpos[pid].copy()
  palm_mat = d.xmat[pid].reshape(3, 3).copy()
  rel_pos = palm_mat.T @ (d.site_xpos[sid] - palm_pos)
  rel_mat = palm_mat.T @ d.site_xmat[sid].reshape(3, 3)
  rel_quat = np.empty(4)
  mujoco.mju_mat2Quat(rel_quat, rel_mat.flatten())
  return rel_pos, rel_quat


def test_hand2_right_tag_pose_matches_measured():
  rel_pos, rel_quat = _hand2_tag_pose_in_palm()
  assert np.allclose(rel_pos, (-0.00299939065, 0.0371578892, 0.063237), atol=1e-5)
  expected_q = np.array((0.4999959, -0.5000041, -0.4999959, -0.5000041))
  assert np.allclose(rel_quat, expected_q, atol=1e-5) or np.allclose(
    rel_quat, -expected_q, atol=1e-5
  )


def test_deploy_hand2_tag_pose_matches_training_scene():
  rel_pos, rel_quat = _hand2_tag_pose_in_palm()
  deploy = load_constants(2, "right")
  assert np.allclose(deploy.tag_in_palm_pos, rel_pos, atol=1e-6)
  assert np.allclose(deploy.tag_in_palm_quat, rel_quat, atol=1e-6) or np.allclose(
    deploy.tag_in_palm_quat, -rel_quat, atol=1e-6
  )
