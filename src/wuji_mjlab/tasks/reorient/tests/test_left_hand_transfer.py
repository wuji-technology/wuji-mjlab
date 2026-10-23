# SPDX-License-Identifier: Apache-2.0
"""Coordinate parity and isolation of experimental left-hand playback."""

import dataclasses
import types

import mjlab.utils.lab_api.math as math_utils
import mujoco
import numpy as np
import pytest
import torch
import wuji_mjlab.assets.robots.wuji_hand2.wuji_hand2_cfg as hand
import wuji_mjlab.tasks.reorient.mdp.cage as cage
import wuji_mjlab.utils.hand_mirroring as mirroring
import wuji_mjlab.utils.left_hand_playback as transfer
import wuji_mjlab.utils.task_cfg_utils as task_cfg_utils


def test_quaternion_reflection_preserves_rotation_action():
  generator = torch.Generator().manual_seed(123)
  quat = torch.randn(16, 4, generator=generator, dtype=torch.float64)
  quat /= quat.norm(dim=-1, keepdim=True)
  points = torch.randn(16, 3, generator=generator, dtype=torch.float64)
  reflected = mirroring.quat_y(quat)
  torch.testing.assert_close(
    math_utils.quat_apply(reflected, mirroring.points_y(points)),
    mirroring.points_y(math_utils.quat_apply(quat, points)),
  )
  torch.testing.assert_close(mirroring.quat_y(reflected), quat)
  matrix = math_utils.matrix_from_quat(quat)
  expected = math_utils.matrix_from_quat(reflected).reshape(16, 9)[:, 3:]
  torch.testing.assert_close(
    mirroring.rotation_error_y(matrix.reshape(16, 9)[:, 3:]), expected
  )


def test_cage_margin_follows_the_left_palm_normal():
  generator = torch.Generator().manual_seed(7)
  links = torch.randn(12, 21, 3, generator=generator) * 0.05
  positions = torch.randn(12, 3, generator=generator) * 0.08
  right = cage._compute_cage_aabb_escape_in_palm(positions, links, 0.01, up_axis=1)
  left = cage._compute_cage_aabb_escape_in_palm(
    mirroring.points_y(positions),
    mirroring.points_y(links),
    0.01,
    up_axis=1,
    up_sign=-1,
  )
  torch.testing.assert_close(left, right)


def test_official_left_geometry_preserves_joint_correspondence():
  right = hand.get_wuji_hand2_cfg("right").spec_fn().compile()
  left = hand.get_wuji_hand2_cfg("left").spec_fn().compile()
  assert left.nq == right.nq == 20
  assert [left.joint(i).name for i in range(20)] == [
    right.joint(i).name for i in range(20)
  ]
  np.testing.assert_array_equal(left.jnt_range, right.jnt_range)
  # Hinge axes are axial vectors: reflection applies det(S) S, not S.
  np.testing.assert_allclose(left.jnt_axis, right.jnt_axis * (-1, 1, -1))
  assert hand.wuji_hand2_mesh_dir("left").name == "left"
  assert not np.allclose(left.body_pos, right.body_pos)
  assert not np.array_equal(left.mesh_vert, right.mesh_vert)


@pytest.mark.parametrize("task_id", transfer.SUPPORTED_TASKS)
def test_left_config_keeps_the_right_config_and_actions_unchanged(task_id):
  right, _ = task_cfg_utils.prepare_task_cfgs(task_id, [], play=True)
  right_model_before = right.scene.entities["robot"].spec_fn().compile()
  root_before = dataclasses.asdict(right.scene.entities["robot"].init_state)
  left = transfer.configure_left_hand_playback(
    right, task_id, mirror_policy_observations=True
  )
  native = transfer.configure_left_hand_playback(
    right, task_id, mirror_policy_observations=False
  )
  assert dataclasses.asdict(right.scene.entities["robot"].init_state) == root_before
  assert dataclasses.asdict(left.actions["joint_pos"]) == dataclasses.asdict(
    right.actions["joint_pos"]
  )
  lm = left.scene.entities["robot"].spec_fn().compile()
  nm = native.scene.entities["robot"].spec_fn().compile()
  rm = right.scene.entities["robot"].spec_fn().compile()
  for field in ("body_pos", "body_quat", "geom_pos", "geom_quat", "jnt_axis"):
    np.testing.assert_array_equal(getattr(lm, field), getattr(nm, field))
    np.testing.assert_array_equal(
      getattr(rm, field), getattr(right_model_before, field)
    )
  if task_id == "WujiHand2_PenSpin":
    assert left.commands["motion"].mirror_reference_y
    assert not right.commands["motion"].mirror_reference_y
  else:
    # Compare tag poses in the palm-rooted model, including actual mount offsets.
    rd, ld = mujoco.MjData(rm), mujoco.MjData(lm)
    mujoco.mj_forward(rm, rd)
    mujoco.mj_forward(lm, ld)
    np.testing.assert_allclose(
      ld.site("right_wrist_tag").xpos,
      rd.site("right_wrist_tag").xpos * (1, -1, 1),
      atol=1e-12,
    )
    reflection = np.diag((1, -1, 1))
    np.testing.assert_allclose(
      ld.site("right_wrist_tag").xmat.reshape(3, 3),
      reflection @ rd.site("right_wrist_tag").xmat.reshape(3, 3) @ reflection,
      atol=1e-12,
    )
    assert left.rewards["cage_escape"].params["up_sign"] == -1
    assert "up_sign" not in right.rewards["cage_escape"].params


def test_pen_command_adapter_keeps_joint_targets_and_input_buffer():
  joints = torch.arange(20).reshape(1, 20).float()
  points = torch.arange(69).reshape(1, 23, 3).float()
  original = torch.cat((joints, points.flatten(1)), dim=-1)
  before = original.clone()
  motion = types.SimpleNamespace(joint_names=tuple(range(20)))
  env = types.SimpleNamespace(
    command_manager=types.SimpleNamespace(get_term=lambda _: motion)
  )
  actual = transfer._policy_observation(lambda env: original, "motion", env)
  torch.testing.assert_close(actual[:, :20], joints)
  torch.testing.assert_close(
    actual[:, 20:].reshape(1, 23, 3), points * torch.tensor((1, -1, 1))
  )
  torch.testing.assert_close(original, before)


def test_rejects_unsupported_task_and_training_config():
  right, _ = task_cfg_utils.prepare_task_cfgs("WujiHand2_Reorient_50Hz", [], play=True)
  with pytest.raises(ValueError, match="supports only"):
    transfer.configure_left_hand_playback(
      right, "WujiHand1_Reorient", mirror_policy_observations=True
    )
  right.observations["policy"].enable_corruption = True
  with pytest.raises(ValueError, match="play configuration"):
    transfer.configure_left_hand_playback(
      right, "WujiHand2_Reorient_50Hz", mirror_policy_observations=True
    )
