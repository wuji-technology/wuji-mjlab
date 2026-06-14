# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import math

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_error_magnitude

from wuji_mjlab.tasks.reorient.mdp.event_impl.state import get_reorient_event_state
from wuji_mjlab.utils.reward_decorators import curriculum_scaled

# ---------------------------------------------------------------------------
# Finger self-collision
# ---------------------------------------------------------------------------


def finger_self_collision_penalty(
  env,
  sensor_cfg: SceneEntityCfg = SceneEntityCfg("finger_collision"),
) -> torch.Tensor:
  """Count active finger-finger contacts. Pure binary — no force scaling."""
  sensor = env.scene[sensor_cfg.name]
  return torch.sum((sensor.data.found > 0).float(), dim=-1)


# ---------------------------------------------------------------------------
# Tolerance kernels (reorient-specific)
# ---------------------------------------------------------------------------


def tolerance(
  value: torch.Tensor,
  bounds: tuple[float, float],
  margin: float,
  value_at_margin: float = 0.1,
) -> torch.Tensor:
  """Gaussian tolerance kernel.

  Returns 1.0 inside *bounds*, decays as a Gaussian outside.
  *sigma* is derived so that the return value equals *value_at_margin*
  at exactly *margin* distance from the nearest bound.
  """
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
  """Linear tolerance kernel (MJX-aligned).

  Returns 1.0 inside *bounds*, linearly decays to 0 at *margin* distance.
  """
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


# --- Dense rewards (× step_dt) ---


def _resolve_first_body_id(asset: Entity, cfg: SceneEntityCfg) -> int:
  body_ids = cfg.body_ids
  if not isinstance(body_ids, slice):
    if isinstance(body_ids, torch.Tensor):
      return int(body_ids[0].item())
    return int(body_ids[0])

  body_names = cfg.body_names
  if body_names is None:
    raise ValueError(
      f"SceneEntityCfg for entity '{cfg.name}' must specify body_names."
    )
  patterns = (body_names,) if isinstance(body_names, str) else tuple(body_names)
  for pattern in patterns:
    resolved, _ = asset.find_bodies(pattern)
    if resolved:
      return int(resolved[0])
  raise ValueError(f"No body matched pattern {body_names} on entity '{cfg.name}'.")


def repose_position_distance(
  env,
  cube_offset_in_palm: tuple[float, float, float],
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Distance from cube to the RevoLab repose in-hand position target."""
  obj: Entity = env.scene[object_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  palm_id = _resolve_first_body_id(robot, robot_cfg)
  palm_pose_w = robot.data.body_link_pose_w[:, palm_id, :]
  palm_pos_w = palm_pose_w[:, :3]
  palm_quat_w = palm_pose_w[:, 3:7]
  offset = torch.tensor(
    cube_offset_in_palm, device=env.device, dtype=palm_pos_w.dtype
  ).expand(env.num_envs, 3)
  target_pos_w = palm_pos_w + quat_apply(palm_quat_w, offset)
  return torch.norm(obj.data.root_link_pos_w - target_pos_w, p=2, dim=-1)


def repose_inverse_orientation_reward(
  env,
  command_name: str = "reorient_command",
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  rot_eps: float = 0.1,
) -> torch.Tensor:
  """RevoLab repose orientation reward: ``1 / (rot_error + rot_eps)``."""
  obj: Entity = env.scene[object_cfg.name]
  goal_quat = env.command_manager.get_term(command_name).goal_quat
  rot_dist = quat_error_magnitude(obj.data.root_link_quat_w, goal_quat)
  return 1.0 / (torch.abs(rot_dist) + rot_eps)


def repose_action_l2_penalty(env) -> torch.Tensor:
  """RevoLab repose action regularization: ``sum(actions ** 2)``."""
  return torch.sum(torch.square(env.action_manager.action), dim=-1)


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
  """MJX-aligned orientation reward using a linear tolerance kernel."""
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
  """L2 deviation from default joint positions."""
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
  """First-order + second-order penalty on raw policy actions."""
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
  """Normalized squared joint velocity penalty.

  Matches MJX: sum((vel / (max_velocity - tolerance))^2).
  """
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
  """Energy: sum(|vel| * |torque|)."""
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
  """Normalized squared torque: sum((torque / torque_max)^2)."""
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
  """Joint acceleration penalty: Σ((vel_t - vel_{t-1}) / denom)².

  Penalizes joint jerk/vibration. Requires reset_joint_acc_cache event
  to clear shared previous-joint-velocity state on episode reset.
  """
  del curriculum_term, curriculum_min
  robot: Entity = env.scene[asset_cfg.name]
  vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  state = get_reorient_event_state(env)

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
  """Dense reward for the cube being held by the distal fingers only.

  Returns 1.0 per env iff BOTH:
    - palm_object_found.found == 0 (cube is not touching palm or proximal
      finger links link1/link2), AND
    - distal_finger_object_found.found > 0 (cube is in contact with the
      distal phalanges link3/link4 of at least one finger).

  Returns 0.0 if either condition fails — i.e., the cube is rolling on the
  palm/proximal links, or the cube is floating away with no fingertip
  contact at all. Encourages the policy to lift the cube fully into the
  fingertips.

  Uses the binary ``found`` field rather than a force threshold so it
  remains correct for very light cubes (the 54 mm cube weighs only
  ~1.18 N, below typical force thresholds).
  """
  palm_sensor: ContactSensor = env.scene[sensor_cfg.name]
  distal_sensor: ContactSensor = env.scene[distal_sensor_cfg.name]
  if palm_sensor.data.found is None or distal_sensor.data.found is None:
    return torch.zeros(env.num_envs, device=env.device)
  palm_found = palm_sensor.data.found[:, 0]
  distal_found = distal_sensor.data.found[:, 0]
  return ((palm_found == 0) & (distal_found > 0)).float()


def tip_slide_penalty(
  env,
  robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  sensor_cfg: SceneEntityCfg = SceneEntityCfg("tip_object_contact"),
  contact_threshold: float = 0.0,
) -> torch.Tensor:
  """MJX-aligned tip slide penalty without curriculum scaling.

  Prioritizes binary contact detection via sensor ``found`` when available,
  matching the MJX task's gating behavior more closely.
  """
  robot: Entity = env.scene[robot_cfg.name]
  obj: Entity = env.scene[object_cfg.name]
  sensor: ContactSensor = env.scene[sensor_cfg.name]

  tip_lin_vel = robot.data.site_lin_vel_w[:, robot_cfg.site_ids, :]
  obj_lin_vel = obj.data.root_link_lin_vel_w.unsqueeze(1)
  rel_vel = torch.norm(tip_lin_vel - obj_lin_vel, dim=-1)

  if sensor.data.found is not None:
    in_contact = (sensor.data.found > contact_threshold).float()
  elif sensor.data.force is not None:
    in_contact = (torch.norm(sensor.data.force, dim=-1) > contact_threshold).float()
  else:
    return torch.zeros(env.num_envs, device=env.device)

  if in_contact.shape[1] != rel_vel.shape[1]:
    n = min(in_contact.shape[1], rel_vel.shape[1])
    in_contact = in_contact[:, :n]
    rel_vel = rel_vel[:, :n]

  return torch.sum(rel_vel * in_contact, dim=-1)


def hold_escalation_value(
  in_window: torch.Tensor,
  within_threshold: torch.Tensor,
  timer: torch.Tensor,
) -> torch.Tensor:
  """Pure function: timer-scaled reward, active only in window & threshold."""
  return (in_window & within_threshold).float() * timer.float()


def hold_escalation(
  env,
  command_name: str = "reorient_command",
) -> torch.Tensor:
  """Time-escalating dense reward during SUCCESS_WINDOW.

  Replaces sparse success (+100) and flat hold_bonus (+1.0/step).
  Returns timer value (1..20) when in SUCCESS_WINDOW and within
  threshold, zero otherwise. With weight=11.4 and dt=0.05,
  total per cycle ≈ 120 (matching old sparse + hold_bonus).

  Uses reward_window_timer (pre-reset snapshot) so the final step
  before goal switch is not lost.
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
  """Penalty when the referenced termination term triggers."""
  return env.termination_manager.get_term(term_name).float()
