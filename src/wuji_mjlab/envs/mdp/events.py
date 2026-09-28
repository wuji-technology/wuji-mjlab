# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Event terms shared by the Wuji hand tasks."""

from __future__ import annotations

import torch
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg


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
