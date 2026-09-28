# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from wuji_mjlab.tasks.reorient.mdp._env_utils import resolve_env_ids

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


def resolve_joint_velocity_limits(
  asset: Entity, env_ids: torch.Tensor
) -> torch.Tensor | None:
  soft_vel_limits = getattr(asset.data, "soft_joint_vel_limits", None)
  if soft_vel_limits is not None:
    return soft_vel_limits[env_ids]

  hard_vel_limits = getattr(asset.data, "joint_vel_limits", None)
  if hard_vel_limits is not None:
    return hard_vel_limits[env_ids]

  return None


def resolve_random_bounds(
  lower: torch.Tensor,
  upper: torch.Tensor,
  range_cfg: tuple[float | None, float | None],
  operation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
  low_cfg, high_cfg = range_cfg
  if operation == "abs":
    low = lower if low_cfg is None else torch.full_like(lower, float(low_cfg))
    high = upper if high_cfg is None else torch.full_like(upper, float(high_cfg))
    return low, high

  low = lower if low_cfg is None else lower * float(low_cfg)
  high = upper if high_cfg is None else upper * float(high_cfg)
  return low, high


def sample_and_clip(
  low: torch.Tensor,
  high: torch.Tensor,
  lower_bound: torch.Tensor,
  upper_bound: torch.Tensor,
  default_offset: torch.Tensor | None = None,
) -> torch.Tensor:
  sampled = low + torch.rand_like(low) * (high - low)
  if default_offset is not None:
    sampled = sampled + default_offset
  return torch.clamp(sampled, min=lower_bound, max=upper_bound)


def reset_joints_within_limits_range(
  env,
  env_ids,
  position_range: dict[str, tuple[float | None, float | None]],
  velocity_range: dict[str, tuple[float | None, float | None]],
  use_default_offset: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  operation: str = "abs",
) -> None:
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  if operation not in ("abs", "scale"):
    raise ValueError(f"Unsupported operation '{operation}'. Expected 'abs' or 'scale'.")

  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos[env_ids].clone()
  default_joint_vel = asset.data.default_joint_vel[env_ids].clone()
  soft_pos_limits = asset.data.soft_joint_pos_limits[env_ids]

  joint_pos = default_joint_pos.clone()
  joint_vel = default_joint_vel.clone()

  for joint_name, joint_range in position_range.items():
    joint_ids, _ = asset.find_joints(joint_name)
    if len(joint_ids) == 0:
      continue

    lower = soft_pos_limits[:, joint_ids, 0]
    upper = soft_pos_limits[:, joint_ids, 1]
    low, high = resolve_random_bounds(lower, upper, joint_range, operation)
    default_offset = default_joint_pos[:, joint_ids] if use_default_offset else None
    joint_pos[:, joint_ids] = sample_and_clip(
      low,
      high,
      lower_bound=lower,
      upper_bound=upper,
      default_offset=default_offset,
    )

  joint_vel_limits = (
    resolve_joint_velocity_limits(asset, env_ids) if velocity_range else None
  )

  for joint_name, joint_range in velocity_range.items():
    joint_ids, _ = asset.find_joints(joint_name)
    if len(joint_ids) == 0:
      continue

    if joint_vel_limits is None:
      low_cfg, high_cfg = joint_range
      if operation == "scale":
        raise ValueError(
          f"Velocity operation='scale' requires joint velocity limits, but asset "
          f"'{asset_cfg.name}' does not expose soft/hard velocity limits."
        )
      if low_cfg is None or high_cfg is None:
        raise ValueError(
          f"Velocity range for pattern '{joint_name}' uses None bounds, but asset "
          f"'{asset_cfg.name}' does not expose joint velocity limits."
        )
      lower = torch.full_like(default_joint_vel[:, joint_ids], float(low_cfg))
      upper = torch.full_like(default_joint_vel[:, joint_ids], float(high_cfg))
    else:
      lower = -joint_vel_limits[:, joint_ids]
      upper = joint_vel_limits[:, joint_ids]
    low, high = resolve_random_bounds(lower, upper, joint_range, operation)
    default_offset = default_joint_vel[:, joint_ids] if use_default_offset else None
    joint_vel[:, joint_ids] = sample_and_clip(
      low,
      high,
      lower_bound=lower,
      upper_bound=upper,
      default_offset=default_offset,
    )

  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
