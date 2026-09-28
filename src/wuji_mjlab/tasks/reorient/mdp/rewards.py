# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import math

import torch
from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from wuji_mjlab.tasks.reorient.mdp.runtime_state import (
  get_fingertip_contact_binding,
  get_reorient_runtime_state,
)
from wuji_mjlab.utils.reward_decorators import curriculum_scaled


def finger_self_collision_penalty(
  env,
  sensor_cfg: SceneEntityCfg = SceneEntityCfg("finger_collision"),
) -> torch.Tensor:
  sensor = env.scene[sensor_cfg.name]
  return torch.sum((sensor.data.found > 0).float(), dim=-1)


def tolerance(
  value: torch.Tensor,
  bounds: tuple[float, float],
  margin: float,
  value_at_margin: float = 0.1,
) -> torch.Tensor:
  lower, upper = bounds
  in_bounds = (value >= lower) & (value <= upper)
  if margin <= 0.0:
    return in_bounds.float()

  below = torch.clamp(lower - value, min=0.0)
  above = torch.clamp(value - upper, min=0.0)
  d = torch.maximum(below, above)

  sigma = margin / math.sqrt(-2.0 * math.log(value_at_margin))
  return torch.where(in_bounds, torch.ones_like(d), torch.exp(-0.5 * (d / sigma) ** 2))


def tolerance_linear(
  value: torch.Tensor,
  bounds: tuple[float, float],
  margin: float,
) -> torch.Tensor:
  lower, upper = bounds
  in_bounds = (value >= lower) & (value <= upper)
  if margin <= 0.0:
    return in_bounds.float()

  below = torch.clamp(lower - value, min=0.0)
  above = torch.clamp(value - upper, min=0.0)
  d = torch.maximum(below, above)

  return torch.where(
    in_bounds, torch.ones_like(d), torch.clamp(1.0 - d / margin, min=0.0)
  )


_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")


@curriculum_scaled
def orientation_alignment(
  env,
  command_name: str = "reorient_command",
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  margin: float = 3.14159,
  bound: float = 0.2,
  curriculum_term: str = "",
  curriculum_min: float = 0.0,
) -> torch.Tensor:
  del curriculum_term, curriculum_min
  obj: Entity = env.scene[object_cfg.name]
  goal_quat = env.command_manager.get_term(command_name).goal_quat
  ori_err = quat_error_magnitude(obj.data.root_link_quat_w, goal_quat)
  return tolerance_linear(ori_err, bounds=(0.0, bound), margin=margin)


@curriculum_scaled
def hand_pose_penalty(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  curriculum_term: str = "",
  curriculum_min: float = 0.0,
) -> torch.Tensor:
  del curriculum_term, curriculum_min
  robot: Entity = env.scene[asset_cfg.name]
  joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
  default_pos = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(joint_pos - default_pos), dim=-1)


@curriculum_scaled
def action_rate_combined(
  env,
  curriculum_term: str = "",
  curriculum_min: float = 0.0,
) -> torch.Tensor:
  del curriculum_term, curriculum_min
  action = env.action_manager.action
  prev = env.action_manager.prev_action
  prev_prev = env.action_manager.prev_prev_action

  first_order = torch.sum(torch.square(action - prev), dim=-1)
  second_order = torch.sum(torch.square(action - 2.0 * prev + prev_prev), dim=-1)

  return first_order + second_order


@curriculum_scaled
def joint_vel_penalty(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  max_velocity: float = 5.0,
  vel_tolerance: float = 1.0,
  curriculum_term: str = "",
  curriculum_min: float = 0.0,
) -> torch.Tensor:
  del curriculum_term, curriculum_min
  robot: Entity = env.scene[asset_cfg.name]
  denom = max(max_velocity - vel_tolerance, 1e-6)
  vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(vel / denom), dim=-1)


@curriculum_scaled
def energy_penalty(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  curriculum_term: str = "",
  curriculum_min: float = 0.0,
) -> torch.Tensor:
  del curriculum_term, curriculum_min
  robot: Entity = env.scene[asset_cfg.name]
  torques = robot.data.actuator_force[:, asset_cfg.actuator_ids]
  velocities = robot.data.joint_vel[:, asset_cfg.joint_ids]
  n = min(torques.shape[1], velocities.shape[1])
  return torch.sum(torch.abs(velocities[:, :n]) * torch.abs(torques[:, :n]), dim=-1)


@curriculum_scaled
def torque_penalty(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  curriculum_term: str = "",
  curriculum_min: float = 0.0,
) -> torch.Tensor:
  del curriculum_term, curriculum_min
  robot: Entity = env.scene[asset_cfg.name]
  torques = robot.data.actuator_force[:, asset_cfg.actuator_ids]
  torque_limits = getattr(robot.data, "actuator_effort_limit", None)
  if torque_limits is not None:
    limits = torque_limits[:, asset_cfg.actuator_ids].clamp_min(1e-3)
    normed = torques / limits
  else:
    normed = torques
  return torch.sum(torch.square(normed), dim=-1)


@curriculum_scaled
def joint_acc_penalty(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  denom: float = 4.0,
  curriculum_term: str = "",
  curriculum_min: float = 0.0,
) -> torch.Tensor:
  """Requires reset_joint_acc_cache to clear previous joint-velocity state on episode reset."""
  del curriculum_term, curriculum_min
  robot: Entity = env.scene[asset_cfg.name]
  vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  state = get_reorient_runtime_state(env)

  if state.prev_joint_vel is None:
    state.prev_joint_vel = vel.clone()
    return torch.zeros(env.num_envs, device=env.device)

  acc = (vel - state.prev_joint_vel) / max(denom, 1e-6)
  state.prev_joint_vel = vel.clone()
  return torch.sum(torch.square(acc), dim=-1)


def palm_detach_reward(
  env,
  sensor_cfg: SceneEntityCfg = SceneEntityCfg("palm_object_found"),
  distal_sensor_cfg: SceneEntityCfg = SceneEntityCfg("distal_finger_object_found"),
) -> torch.Tensor:
  palm_sensor: ContactSensor = env.scene[sensor_cfg.name]
  distal_sensor: ContactSensor = env.scene[distal_sensor_cfg.name]
  if palm_sensor.data.found is None or distal_sensor.data.found is None:
    return torch.zeros(env.num_envs, device=env.device)
  palm_found = (palm_sensor.data.found > 0).any(dim=-1)
  distal_found = (distal_sensor.data.found > 0).any(dim=-1)
  return (~palm_found & distal_found).float()


class TipSlidePenalty(ManagerTermBase):
  def __init__(self, cfg, env):
    super().__init__(env)
    robot_cfg = cfg.params.get("robot_cfg", _DEFAULT_ROBOT_CFG)
    object_cfg = cfg.params.get("object_cfg", _DEFAULT_OBJECT_CFG)
    sensor_cfg = cfg.params.get("sensor_cfg", SceneEntityCfg("tip_object_contact"))
    self._robot = env.scene[robot_cfg.name]
    self._object = env.scene[object_cfg.name]
    self._sensor = env.scene[sensor_cfg.name]
    self._threshold = cfg.params.get("contact_threshold", 0.0)
    self._binding = get_fingertip_contact_binding(env, robot_cfg, sensor_cfg)

  def __call__(self, env, **_params) -> torch.Tensor:
    tip_lin_vel = self._robot.data.site_lin_vel_w[:, self._binding.site_ids, :]
    obj_lin_vel = self._object.data.root_link_lin_vel_w.unsqueeze(1)
    rel_vel = torch.norm(tip_lin_vel - obj_lin_vel, dim=-1)
    data = self._sensor.data
    if data.found is not None:
      in_contact = self._binding.contact_mask(data.found > self._threshold)
    elif data.force is not None:
      in_contact = self._binding.contact_mask(
        torch.norm(data.force, dim=-1) > self._threshold
      )
    else:
      return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(rel_vel * in_contact, dim=-1)


def hold_escalation_value(
  in_window: torch.Tensor,
  within_threshold: torch.Tensor,
  timer: torch.Tensor,
) -> torch.Tensor:
  return (in_window & within_threshold).float() * timer.float()


def hold_escalation(
  env,
  command_name: str = "reorient_command",
) -> torch.Tensor:
  """Time-escalating dense reward during SUCCESS_WINDOW.

  Uses ``reward_hold_counter_snapshot`` (captured before the goal-switch reset)
  so the final step before goal switch is not lost.
  """
  command = env.command_manager.get_term(command_name)
  return hold_escalation_value(
    command.in_success_window,
    command.within_threshold,
    command.reward_hold_counter_snapshot,
  )


def drop_penalty_sparse(
  env,
  term_name: str = "cube_drop",
) -> torch.Tensor:
  return env.termination_manager.get_term(term_name).float()


class ActionHighFreqPenalty(ManagerTermBase):
  def __init__(self, cfg, env):
    super().__init__(env)
    self._c1: torch.Tensor | None = None
    self._c2: torch.Tensor | None = None
    self._age: torch.Tensor | None = None

  def reset(self, env_ids=None) -> None:
    if self._c1 is None:
      return
    if env_ids is None:
      self._c1.zero_()
      self._c2.zero_()
      self._age.zero_()
    else:
      self._c1[env_ids] = 0.0
      self._c2[env_ids] = 0.0
      self._age[env_ids] = 0

  def __call__(self, env) -> torch.Tensor:
    c = env.action_manager.get_term("joint_pos").processed_action
    if self._c1 is None:
      self._c1 = torch.zeros_like(c)
      self._c2 = torch.zeros_like(c)
      self._age = torch.zeros(c.shape[0], dtype=torch.long, device=c.device)
    jerk = c - 2.0 * self._c1 + self._c2
    penalty = torch.sum(torch.square(jerk), dim=-1)
    penalty = torch.where(self._age >= 2, penalty, torch.zeros_like(penalty))
    self._c2 = self._c1
    self._c1 = c.clone()
    self._age = self._age + 1
    return penalty
