# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_inv,
  quat_mul,
)

from wuji_mjlab.utils.math import random_quat_uniform

from .cage import read_cage_penalty_counter
from .runtime_state import get_reorient_runtime_state

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")


def tag_pose_w(robot: Entity, site_ids) -> tuple[torch.Tensor, torch.Tensor]:
  """Wrist-tag world pose read from the robot's ``*_wrist_tag`` site.

  Its pose is the single source of truth for the sim2real tag frame.
  """
  pose = robot.data.site_pose_w[:, site_ids[0]]  # (num_envs, 7): xyz + wxyz
  return pose[..., :3], pose[..., 3:7]


def joint_pos_target_error(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  soft_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
  center = 0.5 * (soft_limits[..., 0] + soft_limits[..., 1])
  half_range = 0.5 * (soft_limits[..., 1] - soft_limits[..., 0])
  normalized_pos = ((joint_pos - center) / (half_range + 1e-6)).clamp(-1.0, 1.0)

  processed = env.action_manager.get_term("joint_pos").processed_action
  target = processed[:, asset_cfg.joint_ids]
  normalized_target = ((target - center) / (half_range + 1e-6)).clamp(-1.0, 1.0)

  return normalized_pos - normalized_target


_DEFAULT_PALM_CFG = SceneEntityCfg("robot", body_names=(".*_palm_link",))
_DEFAULT_TAG_CFG = SceneEntityCfg("robot", site_names=(".*_wrist_tag",))


def _resolve_tag_site_ids(robot: Entity, cfg: SceneEntityCfg):
  """Resolve tag site index from cfg.

  Manager resolves ``site_ids`` via term params, but default kwarg values stay
  unresolved (``site_ids`` remains ``slice(None)``).
  """
  site_ids = cfg.site_ids
  if isinstance(site_ids, slice):
    resolved, _ = robot.find_sites(cfg.site_names)
    if not resolved:
      raise ValueError(
        f"reorient observations: no tag site matched {cfg.site_names!r} "
        f"on entity '{cfg.name}'."
      )
    return resolved
  return site_ids


def cube_pos_in_tag(
  env,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  robot_cfg: SceneEntityCfg = _DEFAULT_PALM_CFG,
  tag_cfg: SceneEntityCfg = _DEFAULT_TAG_CFG,
  injection_prob: float = 0.0,
) -> torch.Tensor:
  """Cube root position expressed in the tag frame."""
  obj: Entity = env.scene[object_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  tag_ids = _resolve_tag_site_ids(robot, tag_cfg)
  tag_pos_w, tag_quat_w = tag_pose_w(robot, tag_ids)

  cube_pos_w = obj.data.root_link_pos_w
  cube_pos_tag = quat_apply_inverse(tag_quat_w, cube_pos_w - tag_pos_w)

  if injection_prob > 0 and env.scene.device != "meta":
    n = cube_pos_tag.shape[0]
    mask = torch.rand(n, 1, device=cube_pos_tag.device) < injection_prob
    rand_pos = torch.empty_like(cube_pos_tag).uniform_(-0.5, 0.5)
    cube_pos_tag = torch.where(mask, rand_pos, cube_pos_tag)

  return cube_pos_tag


def goal_rot_err_6d(
  env,
  command_name: str,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  robot_cfg: SceneEntityCfg = _DEFAULT_PALM_CFG,
  tag_cfg: SceneEntityCfg = _DEFAULT_TAG_CFG,
  injection_prob: float = 0.0,
) -> torch.Tensor:
  """6D rotation error in tag frame: ``mat(cube_tag * goal_tag^-1)[3:9]``."""
  command = env.command_manager.get_term(command_name)
  obj: Entity = env.scene[object_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  tag_ids = _resolve_tag_site_ids(robot, tag_cfg)
  _, tag_quat_w = tag_pose_w(robot, tag_ids)

  tag_quat_inv = quat_inv(tag_quat_w)
  cube_in_tag = quat_mul(tag_quat_inv, obj.data.root_link_quat_w)
  goal_in_tag = quat_mul(tag_quat_inv, command.goal_quat)
  q_err_tag = quat_mul(cube_in_tag, quat_inv(goal_in_tag))

  rot = matrix_from_quat(q_err_tag)
  ori_error = rot.reshape(*rot.shape[:-2], 9)[..., 3:]

  if injection_prob > 0 and env.scene.device != "meta":
    n = ori_error.shape[0]
    mask = torch.rand(n, 1, device=ori_error.device) < injection_prob
    rand_quat = random_quat_uniform(n, device=ori_error.device)
    rand_rot = matrix_from_quat(rand_quat)
    rand_error = rand_rot.reshape(*rand_rot.shape[:-2], 9)[..., 3:]
    ori_error = torch.where(mask, rand_error, ori_error)

  return ori_error


def palm_rot_6d_w(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Palm (robot root) orientation as 6D rotation in world frame."""
  asset: Entity = env.scene[asset_cfg.name]
  palm_quat_w = asset.data.root_link_quat_w
  rot = matrix_from_quat(palm_quat_w)
  return rot.reshape(*rot.shape[:-2], 9)[..., 3:]


def previous_raw_action(env) -> torch.Tensor:
  return env.action_manager.prev_action


def command_state_progress(
  env,
  command_name: str = "reorient_command",
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  hold_progress = command.hold_counter.float() / max(command.cfg.success_hold_steps, 1)
  window_progress = command.reward_window_timer.float() / max(
    command.cfg.goal_switch_delay, 1
  )
  episode_progress = env.episode_length_buf.float() / env.max_episode_length
  goal_count = command.goal_reach_count.float()
  return torch.stack(
    [hold_progress, window_progress, episode_progress, goal_count], dim=-1
  )


def cage_counter_progress(env, max_outside_steps: int = 10) -> torch.Tensor:
  counter = read_cage_penalty_counter(env)
  if counter is None:
    return torch.zeros((env.num_envs, 1), device=env.device)
  return (counter.float() / max(max_outside_steps, 1)).clamp(max=1.0).unsqueeze(-1)


def perturbation_direction(env) -> torch.Tensor:
  return get_reorient_runtime_state(env).pert_force_dir


def perturbation_velocity(env) -> torch.Tensor:
  return get_reorient_runtime_state(env).pert_velocity_cache


def joint_pos_limit_normalized(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  soft_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
  center = 0.5 * (soft_limits[..., 0] + soft_limits[..., 1])
  half_range = 0.5 * (soft_limits[..., 1] - soft_limits[..., 0])
  return ((joint_pos - center) / (half_range + 1e-6)).clamp(-1.0, 1.0)


def root_lin_vel_w(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_w


def root_ang_vel_w(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_w


def body_pos_rel(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  offset: tuple[float, float, float] = (0.0, 0.0, 0.5),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  offset_tensor = torch.tensor(offset, device=asset.data.body_link_pos_w.device)
  return (
    asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    - env.scene.env_origins.unsqueeze(1)
    - offset_tensor
  ).flatten(start_dim=1)


def dr_params_privileged(
  env,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
  if env.scene.device == "meta":
    return torch.ones((env.num_envs, 6), device=env.device)

  robot: Entity = env.scene[robot_cfg.name]
  obj: Entity = env.scene[object_cfg.name]
  n = env.num_envs
  env_ids = torch.arange(n, device=env.device, dtype=torch.long)

  robot_geom_ids = robot.indexing.geom_ids[robot_cfg.geom_ids].long()
  env_g, geom_g = torch.meshgrid(env_ids, robot_geom_ids, indexing="ij")
  cur_friction = env.sim.model.geom_friction[env_g, geom_g, 0]
  def_friction = env.sim.get_default_field("geom_friction")[robot_geom_ids, 0]
  friction_ratio = (cur_friction / def_friction.clamp_min(1e-8)).mean(
    dim=-1, keepdim=True
  )

  obj_body_ids = obj.indexing.body_ids[object_cfg.body_ids].long()
  env_b, body_b = torch.meshgrid(env_ids, obj_body_ids, indexing="ij")
  cur_mass = env.sim.model.body_mass[env_b, body_b]
  def_mass = env.sim.get_default_field("body_mass")[obj_body_ids]
  mass_ratio = (cur_mass / def_mass.clamp_min(1e-8)).mean(dim=-1, keepdim=True)

  robot_act_ids = robot.indexing.ctrl_ids[robot_cfg.actuator_ids].long()
  env_a, act_a = torch.meshgrid(env_ids, robot_act_ids, indexing="ij")
  cur_kp = env.sim.model.actuator_gainprm[env_a, act_a, 0]
  def_kp = env.sim.get_default_field("actuator_gainprm")[robot_act_ids, 0]
  kp_ratio = (cur_kp / def_kp.clamp_min(1e-8)).mean(dim=-1, keepdim=True)

  # biasprm[:,2] is negative (e.g. -0.1); clamp(max=-1e-8) keeps it negative
  # and prevents divide-by-zero without flipping sign.
  cur_kd = env.sim.model.actuator_biasprm[env_a, act_a, 2]
  def_kd = env.sim.get_default_field("actuator_biasprm")[robot_act_ids, 2]
  kd_ratio = (cur_kd / def_kd.clamp(max=-1e-8)).mean(dim=-1, keepdim=True)

  robot_jnt_ids = robot.indexing.joint_v_adr[robot_cfg.joint_ids].long()
  env_j, jnt_j = torch.meshgrid(env_ids, robot_jnt_ids, indexing="ij")
  cur_damp = env.sim.model.dof_damping[env_j, jnt_j]
  def_damp = env.sim.get_default_field("dof_damping")[robot_jnt_ids]
  damp_ratio = (cur_damp / def_damp.clamp_min(1e-8)).mean(dim=-1, keepdim=True)

  obj_geom_ids = obj.indexing.geom_ids[object_cfg.geom_ids].long()
  env_og, geom_og = torch.meshgrid(env_ids, obj_geom_ids, indexing="ij")
  cur_size = env.sim.model.geom_size[env_og, geom_og]
  def_size = env.sim.get_default_field("geom_size")[obj_geom_ids]
  size_ratio = cur_size / def_size.clamp_min(1e-8)
  size_ratio = size_ratio.reshape(n, -1).mean(dim=-1, keepdim=True)

  return torch.cat(
    [friction_ratio, mass_ratio, kp_ratio, kd_ratio, damp_ratio, size_ratio], dim=-1
  )
