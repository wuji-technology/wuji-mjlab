# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Tracking-error and smoothness metrics."""

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .observations import object_position_error, object_shaft_direction_error

_ROBOT = SceneEntityCfg("robot")


def object_pos_tracking_error_cm(
  env,
  command_name: str = "motion",
  robot_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Object position tracking error (cm) vs the motion reference."""
  error = object_position_error(
    env,
    robot_cfg=robot_cfg,
    command_name=command_name,
  )
  return torch.linalg.norm(error, dim=-1) * 100.0


def object_ori_tracking_error_deg(
  env,
  command_name: str = "motion",
  robot_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Pen shaft direction error (deg) vs the motion reference."""
  error = object_shaft_direction_error(
    env,
    robot_cfg=robot_cfg,
    command_name=command_name,
  ).squeeze(-1)
  return torch.rad2deg(error)


def joint_vel_rms(env, joint_cfg: SceneEntityCfg) -> torch.Tensor:
  """RMS joint velocity over the selected joints (rad/s).

  RMS rather than the sum of squares so the number reads in physical units and
  does not scale with how many joints the selector happens to cover.
  """
  vel = env.scene[joint_cfg.name].data.joint_vel[:, joint_cfg.joint_ids]
  return vel.square().mean(dim=-1).sqrt()


def joint_acc_rms(env, joint_cfg: SceneEntityCfg) -> torch.Tensor:
  """RMS joint acceleration over the selected joints (rad/s^2), read from qacc."""
  acc = env.scene[joint_cfg.name].data.joint_acc[:, joint_cfg.joint_ids]
  return acc.square().mean(dim=-1).sqrt()


def actuator_saturation_share(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Share of finger actuators at their torque limit this step.

  The simulated counterpart of the firmware's ``current_limit_active`` flag that
  deploy/pen_spin records per tick, so the two can be read against each other.
  """
  asset = env.scene[asset_cfg.name]
  ids = asset.indexing.ctrl_ids[asset_cfg.actuator_ids]
  limits = env.sim.model.actuator_forcerange[:, ids, 1]
  return (
    (asset.data.actuator_force[:, asset_cfg.actuator_ids].abs() >= 0.98 * limits)
    .float()
    .mean(dim=-1)
  )
