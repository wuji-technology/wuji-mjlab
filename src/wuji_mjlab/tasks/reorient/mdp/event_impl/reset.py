# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)

from wuji_mjlab.tasks.reorient.mdp._env_utils import resolve_env_ids
from wuji_mjlab.tasks.reorient.mdp.runtime_state import get_reorient_runtime_state
from wuji_mjlab.utils.math import random_quat_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")
_POSE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def reset_root_pose_about_world_axes(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  pivot_in_root: tuple[float, float, float] = (0.0, 0.0, 0.0),
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> None:
  """Reset a mocap root by world-frame rotations about a root-frame pivot.

  Wuji Hand 2's palm root maps local Y onto world Z; mjlab's right-multiplied
  local pitch therefore spins around the palm normal instead of tilting it.
  """
  unknown = pose_range.keys() - _POSE_KEYS
  if unknown:
    raise ValueError(f"Unknown pose_range keys: {sorted(unknown)}")

  asset: Entity = env.scene[asset_cfg.name]
  if not asset.is_fixed_base or not asset.is_mocap:
    raise ValueError(
      f"World-axis reset requires a fixed-base mocap entity: {asset_cfg.name}"
    )

  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return
  default_state = asset.data.default_root_state[env_ids]
  ranges = torch.tensor(
    [pose_range.get(key, (0.0, 0.0)) for key in _POSE_KEYS], device=env.device
  )
  sampled = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), env.device)
  delta = quat_from_euler_xyz(sampled[:, 3], sampled[:, 4], sampled[:, 5])
  root_pos, root_quat = default_state[:, :3], default_state[:, 3:7]
  pivot = torch.as_tensor(pivot_in_root, device=env.device, dtype=root_pos.dtype)
  center = root_pos + quat_apply(root_quat, pivot.expand_as(root_pos))
  positions = (
    center
    + quat_apply(delta, root_pos - center)
    + sampled[:, :3]
    + env.scene.env_origins[env_ids]
  )
  orientations = quat_mul(delta, root_quat)
  asset.write_mocap_pose_to_sim(
    torch.cat((positions, orientations), dim=-1), env_ids=env_ids
  )


def reset_object_orientation(
  env,
  env_ids,
  pos_noise: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> None:
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  asset: Entity = env.scene[asset_cfg.name]
  default_state = asset.data.default_root_state[env_ids].clone()

  default_state[:, 3:7] = random_quat_uniform(env_ids.numel(), env.device)

  positions = default_state[:, :3] + env.scene.env_origins[env_ids]
  if pos_noise > 0.0:
    positions += (
      torch.rand(env_ids.numel(), 3, device=env.device) * 2.0 - 1.0
    ) * pos_noise

  default_state[:, 7:] = 0.0

  pose = torch.cat([positions, default_state[:, 3:7]], dim=-1)
  asset.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(default_state[:, 7:], env_ids=env_ids)


def reset_disturbance_caches(env, env_ids) -> None:
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  state = get_reorient_runtime_state(env)
  assert state.pert_force_dir is not None
  assert state.pert_velocity_cache is not None
  state.pert_force_dir[env_ids] = 0.0
  state.pert_velocity_cache[env_ids] = 0.0


def reset_joint_acc_cache(
  env,
  env_ids,
) -> None:
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  state = get_reorient_runtime_state(env)
  if state.prev_joint_vel is None:
    return

  state.prev_joint_vel[env_ids] = 0.0
