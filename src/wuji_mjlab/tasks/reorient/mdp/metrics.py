# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Reorient task metrics — action smoothness + cage/contact diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JERK_PREV_DELTA_ATTR = "_metrics_prev_action_delta"


def _get_action_delta(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return action_t - action_{t-1}, shape (num_envs, action_dim)."""
  action = env.action_manager.action
  prev_action = env.action_manager.prev_action
  return action - prev_action


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------


def action_delta_rms(env: ManagerBasedRlEnv) -> torch.Tensor:
  """RMS of first-order action difference per environment.

  Returns (num_envs,) tensor of ``sqrt(mean((a_t - a_{t-1})^2))``.
  """
  delta = _get_action_delta(env)
  # mean over action dims, then sqrt -> RMS per env
  return torch.sqrt(torch.mean(delta * delta, dim=-1))


def action_jerk_rms(env: ManagerBasedRlEnv) -> torch.Tensor:
  """RMS of second-order action difference (jerk proxy) per environment.

  Computes ``sqrt(mean((delta_t - delta_{t-1})^2))`` where
  ``delta_t = action_t - action_{t-1}``.

  Uses ``env._metrics_prev_action_delta`` to cache the previous delta;
  initialised to zeros on first call or after reset.
  """
  delta = _get_action_delta(env)

  # Retrieve or initialise previous delta cache
  cached_delta: torch.Tensor | None = getattr(env, _JERK_PREV_DELTA_ATTR, None)
  if cached_delta is None or cached_delta.shape != delta.shape:
    prev_delta: torch.Tensor = torch.zeros_like(delta)
  else:
    prev_delta = cached_delta

  # Clear cache for envs that just reset
  if hasattr(env, "reset_buf"):
    reset_mask = env.reset_buf.bool()
    if reset_mask.any():
      prev_delta = prev_delta.clone()
      prev_delta[reset_mask] = 0.0

  jerk = delta - prev_delta
  # Update cache for next step
  setattr(env, _JERK_PREV_DELTA_ATTR, delta.clone())

  return torch.sqrt(torch.mean(jerk * jerk, dim=-1))


# ---------------------------------------------------------------------------
# Cage / contact metrics
# ---------------------------------------------------------------------------


def cage_escape_frequency(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """1.0 if cube is outside AABB cage this step, else 0.0.

  Averaged over episode steps → fraction of steps outside.
  """
  counter = getattr(env, "_cage_penalty_counter", None)
  if counter is None:
    return torch.zeros(env.num_envs, device=env.device)
  return (counter > 0).float()


def fingertip_contact_count(
  env: "ManagerBasedRlEnv",
  sensor_cfg: SceneEntityCfg = SceneEntityCfg("tip_object_contact"),
) -> torch.Tensor:
  """Number of fingertips in contact with cube this step.

  Averaged over episode steps → mean contact count (0–5 range).
  """
  sensor = env.scene[sensor_cfg.name]
  if sensor.data.found is None:
    return torch.zeros(env.num_envs, device=env.device)
  return torch.sum((sensor.data.found > 0).float(), dim=-1)


# ---------------------------------------------------------------------------
# Torque / joint dynamics metrics
# ---------------------------------------------------------------------------

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


def torque_saturation_ratio(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  threshold_frac: float = 0.9,
) -> torch.Tensor:
  """Fraction of actuators exceeding threshold_frac of their force limit.

  Averaged over episode steps → mean saturation ratio (0–1).
  Falls back to zero if force_limit unavailable.
  """
  robot = env.scene[asset_cfg.name]
  torques = robot.data.actuator_force[:, asset_cfg.actuator_ids]
  # force_limit lives on individual actuator instances, not Entity.
  # NOTE: cat order assumes robot.actuators matches actuator_ids index space.
  # Safe for single actuator group (Wuji Hand); may need reorder for multi-group.
  limits_list = [getattr(act, "force_limit", None) for act in robot.actuators]
  if any(lim is None for lim in limits_list):
    return torch.zeros(env.num_envs, device=env.device)
  limits = torch.cat(limits_list, dim=-1)[:, asset_cfg.actuator_ids]
  saturated = (torques.abs() > threshold_frac * limits).float()
  return saturated.mean(dim=-1)


def joint_acceleration_rms(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """RMS of joint accelerations across all joints.

  Averaged over episode steps → mean joint acceleration RMS.
  """
  robot = env.scene[asset_cfg.name]
  acc = robot.data.joint_acc[:, asset_cfg.joint_ids]
  return torch.sqrt(torch.mean(acc * acc, dim=-1))


def joint_vel_rms(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Mean joint velocity RMS across all joints (detects "freeze" policy).

  Averaged over episode steps. Near-zero = policy learned to freeze.
  """
  robot = env.scene[asset_cfg.name]
  vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.sqrt(torch.mean(vel * vel, dim=-1))


# ---------------------------------------------------------------------------
# Cube stability metrics
# ---------------------------------------------------------------------------

_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")


def cube_height_above_palm(
  env: "ManagerBasedRlEnv",
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Cube position along palm normal direction (palm-frame z), gravity-invariant.

  Returns palm-frame z component of cube position relative to palm: positive =
  cube is along palm's local +z direction (i.e. "above" the palm regardless of
  wrist orientation), negative = below. Uses the palm's quaternion to project,
  so it is invariant to arbitrary wrist DR (world-z does not need to align with
  palm normal).

  Averaged over episode steps. Large positive = unstable lift.
  """
  obj = env.scene[object_cfg.name]
  robot = env.scene[robot_cfg.name]
  cube_pos_w = obj.data.root_link_pos_w
  palm_pos_w = robot.data.body_link_pose_w[:, robot_cfg.body_ids[0], :3]
  palm_quat_w = robot.data.body_link_pose_w[:, robot_cfg.body_ids[0], 3:7]
  cube_in_palm = quat_apply_inverse(palm_quat_w, cube_pos_w - palm_pos_w)
  return cube_in_palm[:, 2]


def finger_collision_frequency(
  env: "ManagerBasedRlEnv",
  sensor_cfg: SceneEntityCfg = SceneEntityCfg("finger_collision"),
) -> torch.Tensor:
  """1.0 if any finger-finger self-collision this step, else 0.0.

  Averaged over episode steps → fraction of steps with self-collision.
  """
  sensor = env.scene[sensor_cfg.name]
  if sensor.data.found is None:
    return torch.zeros(env.num_envs, device=env.device)
  return (torch.sum((sensor.data.found > 0).float(), dim=-1) > 0).float()


# ---------------------------------------------------------------------------
# Episode-level progress metrics
# ---------------------------------------------------------------------------


def cube_survival_steps(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Current episode step count (proxy for cube survival time).

  Averaged over episode steps → monotonic proxy for cube survival time
  (the underlying counter grows linearly until reset).
  """
  return env.episode_length_buf.float()


def success_interval(
  env: "ManagerBasedRlEnv",
  command_name: str = "reorient_command",
) -> torch.Tensor:
  """Steps since last goal switch (sawtooth proxy for success interval).

  Averaged over episode steps → ≈ interval/2 when switching is regular.
  Inflated when no success occurs (timer grows monotonically). Interpret
  as rough trend indicator, not exact interval.
  """
  command = env.command_manager.get_term(command_name)
  return command.goal_timer.float()


def curriculum_progress(
  env: "ManagerBasedRlEnv",
  command_name: str = "reorient_command",
) -> torch.Tensor:
  """Success curriculum level (success_threshold tightening progress).

  Reads the current success_threshold from the command term config.
  Averaged over episode steps → monotonic with goal_reach_count.
  """
  command = env.command_manager.get_term(command_name)
  # goal_reach_count as proxy for curriculum progress
  return command.goal_reach_count.float()
