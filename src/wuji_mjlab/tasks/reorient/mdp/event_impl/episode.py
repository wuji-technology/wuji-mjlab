# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from wuji_mjlab.tasks.reorient.mdp.curriculums import get_curriculum_value
from wuji_mjlab.tasks.reorient.mdp.event_impl.curriculum import (
  _get_total_training_steps,
)
from wuji_mjlab.tasks.reorient.mdp.event_impl.state import (
  get_reorient_event_state,
)
from wuji_mjlab.tasks.reorient.mdp.event_utils import (
  get_linear_progress,
  resolve_env_ids,
)
from wuji_mjlab.utils.math import random_quat_uniform

_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")
_DEFAULT_ROBOT_PALM_CFG = SceneEntityCfg("robot", body_names=(".*_palm_link",))


def _resolve_body_id(asset: Entity, cfg: SceneEntityCfg) -> int:
  body_ids = cfg.body_ids
  if isinstance(body_ids, slice):
    resolved, _ = asset.find_bodies(cfg.body_names)
    if not resolved:
      raise ValueError(
        f"No body matched pattern {cfg.body_names} on entity '{cfg.name}'."
      )
    return int(resolved[0])
  return int(body_ids[0])


def reset_object_orientation(
  env,
  env_ids,
  pos_noise: float = 0.01,
  pos_noise_curriculum_term: str = "",
  asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> None:
  """Reset object with random SO(3) orientation and small position noise."""
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  asset: Entity = env.scene[asset_cfg.name]
  default_state = asset.data.default_root_state[env_ids].clone()

  default_state[:, 3:7] = random_quat_uniform(env_ids.numel(), env.device)

  positions = default_state[:, :3] + env.scene.env_origins[env_ids]
  if pos_noise_curriculum_term:
    pos_noise *= get_curriculum_value(env, pos_noise_curriculum_term, 0.0)
  if pos_noise > 0.0:
    positions += (
      torch.rand(env_ids.numel(), 3, device=env.device) * 2.0 - 1.0
    ) * pos_noise

  default_state[:, 7:] = 0.0

  pose = torch.cat([positions, default_state[:, 3:7]], dim=-1)
  asset.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(default_state[:, 7:], env_ids=env_ids)


def reset_object_pose_in_palm(
  env,
  env_ids,
  cube_offset_in_palm: tuple[float, float, float],
  pos_noise: float = 0.0,
  pos_noise_curriculum_term: str = "",
  asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_PALM_CFG,
) -> None:
  """Reset object pose from the current palm pose and a palm-local offset."""
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  asset: Entity = env.scene[asset_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  default_state = asset.data.default_root_state[env_ids].clone()

  palm_id = _resolve_body_id(robot, robot_cfg)
  palm_pose_w = robot.data.body_link_pose_w[env_ids, palm_id, :]
  palm_pos_w = palm_pose_w[:, :3]
  palm_quat_w = palm_pose_w[:, 3:7]

  offset = torch.tensor(
    cube_offset_in_palm, device=env.device, dtype=palm_pos_w.dtype
  ).expand(env_ids.numel(), 3)
  if pos_noise_curriculum_term:
    pos_noise *= get_curriculum_value(env, pos_noise_curriculum_term, 0.0)
  if pos_noise > 0.0:
    offset = offset + (
      torch.rand(env_ids.numel(), 3, device=env.device, dtype=palm_pos_w.dtype)
      * 2.0
      - 1.0
    ) * float(pos_noise)

  positions = palm_pos_w + quat_apply(palm_quat_w, offset)
  default_state[:, 3:7] = random_quat_uniform(env_ids.numel(), env.device)
  default_state[:, 7:] = 0.0

  pose = torch.cat([positions, default_state[:, 3:7]], dim=-1)
  asset.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(default_state[:, 7:], env_ids=env_ids)


def reset_disturbance_caches(env, env_ids) -> None:
  """Reset disturbance observation caches to zero."""
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  state = get_reorient_event_state(env)
  assert state.pert_force_dir is not None
  assert state.pert_velocity_cache is not None
  state.pert_force_dir[env_ids] = 0.0
  state.pert_velocity_cache[env_ids] = 0.0


def reset_joint_acc_cache(
  env,
  env_ids,
) -> None:
  """Reset joint acceleration cache on episode reset."""
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  state = get_reorient_event_state(env)
  if state.prev_joint_vel is None:
    return

  state.prev_joint_vel[env_ids] = 0.0


def apply_velocity_disturbance(
  env,
  env_ids,
  asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  min_speed: float = 0.05,
  max_speed: float = 0.15,
  min_ang_speed: float = 0.0,
  max_ang_speed: float = 0.0,
  warmup_time_s: float = 3.0,
  warmup_frac: float = 0.05,
  rampup_frac: float = 0.80,
  adaptive_curriculum_term: str = "",
) -> None:
  """Apply an instantaneous random velocity to the object on interval trigger."""
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  linear_progress = get_linear_progress(
    env,
    warmup_frac=warmup_frac,
    rampup_frac=rampup_frac,
    total_steps=_get_total_training_steps(),
  )
  adaptive_scale = 1.0
  if adaptive_curriculum_term:
    adaptive_scale = get_curriculum_value(env, adaptive_curriculum_term, 1.0)

  curr_max_speed = (
    float(min_speed)
    + (float(max_speed) - float(min_speed)) * linear_progress * adaptive_scale
  )
  curr_max_ang_speed = (
    float(min_ang_speed)
    + (float(max_ang_speed) - float(min_ang_speed))
    * linear_progress
    * adaptive_scale
  )
  state = get_reorient_event_state(env)
  assert state.pert_force_dir is not None
  assert state.pert_velocity_cache is not None

  episode_time = env.episode_length_buf[env_ids].float() * env.step_dt
  active_mask = episode_time >= float(warmup_time_s)
  active_local = active_mask.nonzero(as_tuple=False).reshape(-1)

  state.pert_force_dir[env_ids] = 0.0
  state.pert_velocity_cache[env_ids] = 0.0

  if active_local.numel() == 0:
    return

  active_env_ids = env_ids[active_local]

  vel_dir = torch.randn((active_env_ids.numel(), 3), device=env.device)
  vel_dir = vel_dir / vel_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
  speed = torch.empty(active_env_ids.numel(), 1, device=env.device).uniform_(
    float(min_speed),
    max(float(min_speed), curr_max_speed),
  )
  velocity = vel_dir * speed
  ang_velocity = torch.zeros_like(velocity)
  if max(float(min_ang_speed), curr_max_ang_speed) > 0.0:
    ang_dir = torch.randn((active_env_ids.numel(), 3), device=env.device)
    ang_dir = ang_dir / ang_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    ang_speed = torch.empty(active_env_ids.numel(), 1, device=env.device).uniform_(
      float(min_ang_speed),
      max(float(min_ang_speed), curr_max_ang_speed),
    )
    ang_velocity = ang_dir * ang_speed

  state.pert_force_dir[active_env_ids] = vel_dir
  state.pert_velocity_cache[active_env_ids] = torch.cat(
    [velocity, ang_velocity],
    dim=-1,
  )

  asset: Entity = env.scene[asset_cfg.name]
  root_vel = asset.data.root_link_vel_w[active_env_ids].clone()
  root_vel[:, :3] += velocity
  root_vel[:, 3:] += ang_velocity
  asset.write_root_link_velocity_to_sim(root_vel, env_ids=active_env_ids)
