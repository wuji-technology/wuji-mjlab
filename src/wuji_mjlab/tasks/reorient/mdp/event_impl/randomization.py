# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.dr.geom import _recompute_geom_bounds
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from wuji_mjlab.tasks.reorient.mdp.curriculums import get_curriculum_value
from wuji_mjlab.tasks.reorient.mdp.event_utils import resolve_env_ids

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")


def _resolve_field_entity_indices(
  asset: Entity,
  asset_cfg: SceneEntityCfg,
  field: str,
) -> torch.Tensor:
  if field.startswith("geom_"):
    return asset.indexing.geom_ids[asset_cfg.geom_ids].long()
  if field.startswith("body_"):
    return asset.indexing.body_ids[asset_cfg.body_ids].long()
  if field.startswith("dof_"):
    return asset.indexing.joint_v_adr[asset_cfg.joint_ids].long()
  raise ValueError(f"Unsupported field for reorient randomize_field: {field}")


def _get_default_field_values(
  env,
  field: str,
  env_ids: torch.Tensor,
  entity_ids: torch.Tensor,
) -> torch.Tensor:
  default_field = env.sim.get_default_field(field)
  if field in getattr(env.sim, "per_world_default_fields", ()):
    env_grid, entity_grid = torch.meshgrid(env_ids.long(), entity_ids, indexing="ij")
    return default_field[env_grid, entity_grid]

  values = default_field[entity_ids].unsqueeze(0)
  return values.expand((env_ids.numel(),) + values.shape[1:])


def _sample_field_random_values(
  shape: tuple[int, ...],
  device: str | torch.device,
  low: float,
  high: float,
  distribution: str,
) -> torch.Tensor:
  if distribution == "uniform":
    return torch.empty(shape, device=device).uniform_(float(low), float(high))
  if distribution == "log_uniform":
    return torch.exp(
      torch.empty(shape, device=device).uniform_(
        float(torch.log(torch.tensor(low))),
        float(torch.log(torch.tensor(high))),
      )
    )
  if distribution == "gaussian":
    mean = 0.5 * (float(low) + float(high))
    std = 0.5 * (float(high) - float(low))
    return torch.empty(shape, device=device).normal_(mean=mean, std=std)
  raise ValueError(f"Unsupported distribution: {distribution}")


def _write_randomized_model_field(
  env,
  env_ids: torch.Tensor | None,
  field: str,
  ranges: tuple[float, float],
  distribution: str,
  operation: str,
  asset_cfg: SceneEntityCfg,
) -> None:
  resolved_env_ids: torch.Tensor = resolve_env_ids(env, env_ids)
  if resolved_env_ids.numel() == 0:
    return

  asset = env.scene[asset_cfg.name]
  entity_ids = _resolve_field_entity_indices(asset, asset_cfg, field)
  model_field = getattr(env.sim.model, field)

  env_grid, entity_grid = torch.meshgrid(resolved_env_ids, entity_ids, indexing="ij")
  indexed = model_field[env_grid, entity_grid]
  base = (
    _get_default_field_values(env, field, resolved_env_ids, entity_ids)
    if operation in ("scale", "add")
    else indexed
  )
  sampled = _sample_field_random_values(
    tuple(base.shape), env.device, ranges[0], ranges[1], distribution
  )

  if operation == "scale":
    result = base * sampled
  elif operation == "add":
    result = base + sampled
  elif operation == "abs":
    result = sampled
  else:
    raise ValueError(f"Unsupported operation: {operation}")

  model_field[env_grid, entity_grid] = result


def randomize_field(
  env,
  env_ids,
  field: str,
  ranges: tuple[float, float] | dict[int, tuple[float, float]],
  distribution: str = "uniform",
  operation: str = "abs",
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  axes: list[int] | None = None,
):
  del axes

  if isinstance(ranges, dict):
    if set(ranges.keys()) != {0, 1, 2}:
      raise ValueError(
        "Reorient randomize_field only supports dict ranges over axes {0,1,2}."
      )
    ordered = [ranges[i] for i in (0, 1, 2)]
    if len({tuple(v) for v in ordered}) != 1:
      raise ValueError(
        "Reorient randomize_field only supports equal per-axis ranges when using dict ranges."
      )
    ranges = ordered[0]

  if field == "body_ipos":
    return dr.body_com_offset(
      env,
      env_ids,
      ranges=ranges,
      asset_cfg=asset_cfg,
      distribution=distribution,
      operation=operation,
    )
  if field == "geom_friction":
    return dr.geom_friction(
      env,
      env_ids,
      ranges=ranges,
      asset_cfg=asset_cfg,
      distribution=distribution,
      operation=operation,
    )
  if field == "dof_damping":
    return dr.dof_damping(
      env,
      env_ids,
      ranges=ranges,
      asset_cfg=asset_cfg,
      distribution=distribution,
      operation=operation,
    )
  if field == "dof_armature":
    return dr.dof_armature(
      env,
      env_ids,
      ranges=ranges,
      asset_cfg=asset_cfg,
      distribution=distribution,
      operation=operation,
    )
  if field == "dof_frictionloss":
    return dr.dof_frictionloss(
      env,
      env_ids,
      ranges=ranges,
      asset_cfg=asset_cfg,
      distribution=distribution,
      operation=operation,
    )
  if field in {"body_mass", "body_inertia"}:
    return _write_randomized_model_field(
      env, env_ids, field, ranges, distribution, operation, asset_cfg
    )

  raise ValueError(f"Unsupported field for reorient randomize_field: {field}")


def randomize_pd_gains(
  env,
  env_ids,
  kp_range: tuple[float, float],
  kd_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  distribution: str = "uniform",
  operation: str = "scale",
) -> None:
  dr.pd_gains(
    env,
    env_ids,
    kp_range=kp_range,
    kd_range=kd_range,
    asset_cfg=asset_cfg,
    distribution=distribution,
    operation=operation,
  )


def randomize_encoder_bias(
  env,
  env_ids,
  bias_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> None:
  dr.encoder_bias(env, env_ids, bias_range=bias_range, asset_cfg=asset_cfg)


@requires_model_fields("geom_friction")
def randomize_object_friction_scale(
  env,
  env_ids,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("object", geom_names=("cube",)),
  scale_range: tuple[float, float] = (0.7, 1.3),
  curriculum_term: str = "adaptive_episode",
) -> None:
  """Randomize object sliding friction around nominal, gated by curriculum."""
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  progress = (
    get_curriculum_value(env, curriculum_term, 0.0) if curriculum_term else 1.0
  )
  progress = max(min(float(progress), 1.0), 0.0)
  low = 1.0 - (1.0 - float(scale_range[0])) * progress
  high = 1.0 + (float(scale_range[1]) - 1.0) * progress

  asset = env.scene[asset_cfg.name]
  geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids].long()
  default_friction = _get_default_field_values(env, "geom_friction", env_ids, geom_ids)
  scales = torch.empty(env_ids.numel(), geom_ids.numel(), device=env.device).uniform_(
    low,
    high,
  )

  env_grid, geom_grid = torch.meshgrid(env_ids.long(), geom_ids, indexing="ij")
  env.sim.model.geom_friction[env_grid, geom_grid, 0] = (
    default_friction[..., 0] * scales
  )


@requires_model_fields("geom_size", "geom_rbound", "geom_aabb")
def randomize_geom_size_uniform(
  env,
  env_ids,
  asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  scale_range: tuple[float, float] = (0.85, 1.15),
) -> None:
  """Uniformly scale all 3 dims of geom_size by one shared random factor per env."""
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  asset = env.scene[asset_cfg.name]
  geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids].long()
  default_size = env.sim.get_default_field("geom_size")

  scales = torch.empty(len(env_ids), 1, 1, device=env.device).uniform_(
    float(scale_range[0]),
    float(scale_range[1]),
  )

  env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")
  env.sim.model.geom_size[env_grid, geom_grid] = (
    default_size[geom_ids].unsqueeze(0) * scales
  )
  _recompute_geom_bounds(env, env_ids.int(), asset_cfg)


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def randomize_body_mass_and_inertia(
  env,
  env_ids,
  asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  scale_range: tuple[float, float] = (0.4, 1.6),
) -> None:
  """Randomize body mass and inertia with one shared scale per env/body."""
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  asset = env.scene[asset_cfg.name]
  body_ids = asset.indexing.body_ids[asset_cfg.body_ids].long()
  default_mass = _get_default_field_values(env, "body_mass", env_ids, body_ids)
  default_inertia = _get_default_field_values(env, "body_inertia", env_ids, body_ids)
  scales = torch.empty(env_ids.numel(), body_ids.numel(), device=env.device).uniform_(
    float(scale_range[0]), float(scale_range[1])
  )

  env_grid, body_grid = torch.meshgrid(env_ids.long(), body_ids, indexing="ij")
  env.sim.model.body_mass[env_grid, body_grid] = default_mass * scales
  env.sim.model.body_inertia[env_grid, body_grid] = default_inertia * scales.unsqueeze(
    -1
  )


@requires_model_fields("body_inertia", recompute=RecomputeLevel.set_const_0)
def randomize_body_inertia(
  env,
  env_ids,
  asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  scale_range: tuple[float, float] = (0.4, 1.6),
) -> None:
  _write_randomized_model_field(
    env,
    env_ids,
    "body_inertia",
    scale_range,
    "uniform",
    "scale",
    asset_cfg,
  )


@requires_model_fields("geom_solref", "geom_solimp")
def randomize_contact_params(
  env,
  env_ids,
  robot_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot", geom_names=(".*palm_.*", ".*finger.*_col")
  ),
  solref_timeconst_range: tuple[float, float] = (1.0, 2.0),
  solref_dampratio_range: tuple[float, float] = (0.8, 1.2),
  solimp_width_range: tuple[float, float] = (1.0, 2.0),
  solimp_dmin_range: tuple[float, float] | None = None,
) -> None:
  """Randomize contact solver parameters on hand collision geoms."""
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  asset = env.scene[robot_cfg.name]
  geom_ids = asset.indexing.geom_ids[robot_cfg.geom_ids].long()

  default_solref = env.sim.get_default_field("geom_solref")[geom_ids]
  default_solimp = env.sim.get_default_field("geom_solimp")[geom_ids]

  env_grid, geom_grid = torch.meshgrid(env_ids.long(), geom_ids, indexing="ij")

  tc_scale = torch.empty(env_ids.numel(), 1, device=env.device).uniform_(
    float(solref_timeconst_range[0]), float(solref_timeconst_range[1])
  )
  env.sim.model.geom_solref[env_grid, geom_grid, 0] = (
    default_solref[:, 0].unsqueeze(0) * tc_scale
  )

  dr_scale = torch.empty(env_ids.numel(), 1, device=env.device).uniform_(
    float(solref_dampratio_range[0]), float(solref_dampratio_range[1])
  )
  env.sim.model.geom_solref[env_grid, geom_grid, 1] = (
    default_solref[:, 1].unsqueeze(0) * dr_scale
  )

  width_scale = torch.empty(env_ids.numel(), 1, device=env.device).uniform_(
    float(solimp_width_range[0]), float(solimp_width_range[1])
  )
  env.sim.model.geom_solimp[env_grid, geom_grid, 2] = (
    default_solimp[:, 2].unsqueeze(0) * width_scale
  )

  if solimp_dmin_range is not None:
    dmin_scale = torch.empty(env_ids.numel(), 1, device=env.device).uniform_(
      float(solimp_dmin_range[0]), float(solimp_dmin_range[1])
    )
    env.sim.model.geom_solimp[env_grid, geom_grid, 0] = (
      default_solimp[:, 0].unsqueeze(0) * dmin_scale
    )
