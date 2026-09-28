# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from wuji_mjlab.tasks.reorient.mdp.runtime_state import get_fingertip_contact_binding
from wuji_mjlab.tasks.reorient.reorient_constants import REORIENT_PALM_NORMAL_AXIS

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_JERK_PREV_DELTA_ATTR = "_metrics_prev_action_delta"


def _get_action_delta(env: ManagerBasedRlEnv) -> torch.Tensor:
  action = env.action_manager.action
  prev_action = env.action_manager.prev_action
  return action - prev_action


def action_delta_rms(env: ManagerBasedRlEnv) -> torch.Tensor:
  delta = _get_action_delta(env)
  return torch.sqrt(torch.mean(delta * delta, dim=-1))


def action_jerk_rms(env: ManagerBasedRlEnv) -> torch.Tensor:
  delta = _get_action_delta(env)

  cached_delta: torch.Tensor | None = getattr(env, _JERK_PREV_DELTA_ATTR, None)
  if cached_delta is None or cached_delta.shape != delta.shape:
    prev_delta: torch.Tensor = torch.zeros_like(delta)
  else:
    prev_delta = cached_delta

  if hasattr(env, "reset_buf"):
    reset_mask = env.reset_buf.bool()
    if reset_mask.any():
      prev_delta = prev_delta.clone()
      prev_delta[reset_mask] = 0.0

  jerk = delta - prev_delta
  setattr(env, _JERK_PREV_DELTA_ATTR, delta.clone())

  return torch.sqrt(torch.mean(jerk * jerk, dim=-1))


def cage_escape_frequency(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """1.0 while the shared cage penalty counter is above zero, else 0.0.

  Averaged over episode steps it is a "recently outside" fraction,
  not a strict fraction of steps spent outside.
  """
  counter = getattr(env, "_cage_penalty_counter", None)
  if counter is None:
    return torch.zeros(env.num_envs, device=env.device)
  return (counter > 0).float()


class FingertipContactCount(ManagerTermBase):
  def __init__(self, cfg, env):
    super().__init__(env)
    robot_cfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
    sensor_cfg = cfg.params.get("sensor_cfg", SceneEntityCfg("tip_object_contact"))
    self._sensor = env.scene[sensor_cfg.name]
    self._binding = get_fingertip_contact_binding(env, robot_cfg, sensor_cfg)

  def __call__(self, env, **_params) -> torch.Tensor:
    found = self._sensor.data.found
    if found is None:
      return torch.zeros(env.num_envs, device=env.device)
    return self._binding.contact_mask(found > 0).sum(dim=-1).float()


_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


def torque_saturation_ratio(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  threshold_frac: float = 0.9,
) -> torch.Tensor:
  robot = env.scene[asset_cfg.name]
  torques = robot.data.actuator_force[:, asset_cfg.actuator_ids]
  ctrl_ids = robot.indexing.ctrl_ids[asset_cfg.actuator_ids]
  limits = env.sim.model.actuator_forcerange[:, ctrl_ids, :].abs().amax(dim=-1)
  saturated = (torques.abs() > threshold_frac * limits).float()
  return saturated.mean(dim=-1)


def joint_acceleration_rms(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  robot = env.scene[asset_cfg.name]
  acc = robot.data.joint_acc[:, asset_cfg.joint_ids]
  return torch.sqrt(torch.mean(acc * acc, dim=-1))


def joint_vel_rms(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  robot = env.scene[asset_cfg.name]
  vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.sqrt(torch.mean(vel * vel, dim=-1))


_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")


def cube_height_above_palm(
  env: "ManagerBasedRlEnv",
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  palm_normal_axis: int = REORIENT_PALM_NORMAL_AXIS,
) -> torch.Tensor:
  """Cube displacement along the palm's surface normal, gravity-invariant."""
  obj = env.scene[object_cfg.name]
  robot = env.scene[robot_cfg.name]
  cube_pos_w = obj.data.root_link_pos_w
  palm_pos_w = robot.data.body_link_pose_w[:, robot_cfg.body_ids[0], :3]
  palm_quat_w = robot.data.body_link_pose_w[:, robot_cfg.body_ids[0], 3:7]
  cube_in_palm = quat_apply_inverse(palm_quat_w, cube_pos_w - palm_pos_w)
  return cube_in_palm[:, palm_normal_axis]


def finger_collision_frequency(
  env: "ManagerBasedRlEnv",
  sensor_cfg: SceneEntityCfg = SceneEntityCfg("finger_collision"),
) -> torch.Tensor:
  sensor = env.scene[sensor_cfg.name]
  if sensor.data.found is None:
    return torch.zeros(env.num_envs, device=env.device)
  return (torch.sum((sensor.data.found > 0).float(), dim=-1) > 0).float()


def palm_detach_frequency(
  env: "ManagerBasedRlEnv",
  sensor_cfg: SceneEntityCfg = SceneEntityCfg("palm_object_found"),
  distal_sensor_cfg: SceneEntityCfg = SceneEntityCfg("distal_finger_object_found"),
) -> torch.Tensor:
  palm_sensor = env.scene[sensor_cfg.name]
  distal_sensor = env.scene[distal_sensor_cfg.name]
  if palm_sensor.data.found is None or distal_sensor.data.found is None:
    return torch.zeros(env.num_envs, device=env.device)
  palm_found = (palm_sensor.data.found > 0).any(dim=-1)
  distal_found = (distal_sensor.data.found > 0).any(dim=-1)
  return (~palm_found & distal_found).float()


def cube_survival_steps(env: "ManagerBasedRlEnv") -> torch.Tensor:
  return env.episode_length_buf.float()


def success_interval(
  env: "ManagerBasedRlEnv",
  command_name: str = "reorient_command",
) -> torch.Tensor:
  """Steps since last goal switch (sawtooth proxy for success interval).

  Interpret as rough trend indicator, not exact interval.
  """
  command = env.command_manager.get_term(command_name)
  return command.goal_timer.float()


def curriculum_progress(
  env: "ManagerBasedRlEnv",
  command_name: str = "reorient_command",
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  return command.goal_reach_count.float()
