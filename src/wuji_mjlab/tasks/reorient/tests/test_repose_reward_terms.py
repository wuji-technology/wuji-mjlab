# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

from types import SimpleNamespace

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from wuji_mjlab.tasks.reorient.mdp.rewards import (
  repose_action_l2_penalty,
  repose_inverse_orientation_reward,
  repose_position_distance,
)


class _FakeRobot:
  def __init__(self, palm_pose_w: torch.Tensor) -> None:
    self.data = SimpleNamespace(body_link_pose_w=palm_pose_w)

  def find_bodies(self, body_name: str) -> tuple[list[int], list[str]]:
    if body_name == "palm":
      return [0], ["palm"]
    return [], []


def _make_env(
  *,
  object_pos_w: torch.Tensor,
  object_quat_w: torch.Tensor | None = None,
  goal_quat_w: torch.Tensor | None = None,
  action: torch.Tensor | None = None,
) -> SimpleNamespace:
  device = object_pos_w.device
  num_envs = object_pos_w.shape[0]
  palm_pose_w = torch.zeros((num_envs, 1, 7), device=device)
  palm_pose_w[:, 0, 3] = 1.0
  robot = _FakeRobot(palm_pose_w)

  if object_quat_w is None:
    object_quat_w = torch.zeros((num_envs, 4), device=device)
    object_quat_w[:, 0] = 1.0
  if goal_quat_w is None:
    goal_quat_w = object_quat_w.clone()
  if action is None:
    action = torch.zeros((num_envs, 3), device=device)

  obj = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=object_pos_w,
      root_link_quat_w=object_quat_w,
    )
  )
  command = SimpleNamespace(goal_quat=goal_quat_w)
  return SimpleNamespace(
    num_envs=num_envs,
    device=device,
    scene={"robot": robot, "object": obj},
    command_manager=SimpleNamespace(get_term=lambda _name: command),
    action_manager=SimpleNamespace(action=action),
  )


def test_repose_position_distance_is_zero_at_palm_local_target():
  offset = (0.055, 0.005, 0.075)
  object_pos = torch.tensor([offset, offset], dtype=torch.float32)
  env = _make_env(object_pos_w=object_pos)

  result = repose_position_distance(
    env,
    cube_offset_in_palm=offset,
    robot_cfg=SceneEntityCfg("robot", body_names=("palm",)),
  )

  assert torch.allclose(result, torch.zeros(2))


def test_repose_position_distance_matches_known_offset():
  offset = (0.055, 0.005, 0.075)
  object_pos = torch.tensor([[0.085, -0.035, 0.075]], dtype=torch.float32)
  env = _make_env(object_pos_w=object_pos)

  result = repose_position_distance(
    env,
    cube_offset_in_palm=offset,
    robot_cfg=SceneEntityCfg("robot", body_names=("palm",)),
  )

  assert torch.allclose(result, torch.tensor([0.05]))


def test_repose_inverse_orientation_reward_is_ten_at_goal():
  quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
  env = _make_env(object_pos_w=torch.zeros((1, 3)), object_quat_w=quat)

  result = repose_inverse_orientation_reward(env)

  assert torch.allclose(result, torch.tensor([10.0]))


def test_repose_action_l2_penalty_sums_squared_actions():
  action = torch.tensor([[1.0, -2.0, 3.0], [0.5, 0.5, -0.5]])
  env = _make_env(object_pos_w=torch.zeros((2, 3)), action=action)

  result = repose_action_l2_penalty(env)

  assert torch.allclose(result, torch.tensor([14.0, 0.75]))
