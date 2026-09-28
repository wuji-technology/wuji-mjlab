# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import torch
from mjlab.envs.mdp.dr.geom import _recompute_geom_bounds
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from wuji_mjlab.tasks.reorient.mdp._env_utils import resolve_env_ids

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")


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


@requires_model_fields("geom_size", "geom_rbound", "geom_aabb")
def randomize_geom_size_uniform(
  env,
  env_ids,
  asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  scale_range: tuple[float, float] = (0.85, 1.15),
) -> None:
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
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  asset = env.scene[asset_cfg.name]
  body_ids = asset.indexing.body_ids[asset_cfg.body_ids].long()
  default_inertia = _get_default_field_values(env, "body_inertia", env_ids, body_ids)
  scales = torch.empty(default_inertia.shape, device=env.device).uniform_(
    float(scale_range[0]), float(scale_range[1])
  )
  env_grid, body_grid = torch.meshgrid(env_ids.long(), body_ids, indexing="ij")
  env.sim.model.body_inertia[env_grid, body_grid] = default_inertia * scales


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
