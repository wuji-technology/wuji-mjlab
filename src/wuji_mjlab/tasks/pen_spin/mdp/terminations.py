# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
import math

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.nan_guard import NanGuard

from .observations import object_position_error, object_shaft_direction_error


def object_position_error_exceeded(
  env,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
  command_name: str = "motion",
  threshold: float = 0.05,
) -> torch.Tensor:
  error = object_position_error(
    env,
    robot_cfg,
    object_cfg,
    command_name,
  )
  return torch.linalg.norm(error, dim=-1) > threshold


def object_shaft_direction_error_exceeded(
  env,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
  command_name: str = "motion",
  threshold_deg: float = 60.0,
) -> torch.Tensor:
  error = object_shaft_direction_error(env, robot_cfg, object_cfg, command_name)
  return error.squeeze(-1) >= math.radians(threshold_deg)


def numerical_instability(
  env,
  object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
  max_pos: float = 3.0,
  max_vel: float = 80.0,
) -> torch.Tensor:
  """Terminate invalid physics or explosive motion before another control step."""
  robot = env.scene["robot"]
  object_entity = env.scene[object_cfg.name]
  pos = object_entity.data.root_link_pos_w
  local_pos = pos - env.scene.env_origins
  lin_vel = object_entity.data.root_link_lin_vel_w
  joint_vel = robot.data.joint_vel
  return (
    NanGuard.detect_nans(env.sim.data)
    | (local_pos.abs() > max_pos).any(-1)
    | (lin_vel.abs() > max_vel).any(-1)
    | (joint_vel.abs() > max_vel).any(-1)
  )


def reference_end(env, command_name="motion"):
  motion = env.command_manager.get_term(command_name)
  return motion.time_steps + 1 >= motion.trajectory_lengths
