# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg


@dataclass(slots=True)
class ReorientRuntimeState:
  pert_force_dir: torch.Tensor | None = None
  pert_velocity_cache: torch.Tensor | None = None
  prev_joint_vel: torch.Tensor | None = None
  fingertip_bindings: dict[
    tuple[str, str, tuple[int, ...]], FingertipContactBinding
  ] = field(default_factory=dict)


def get_reorient_runtime_state(env) -> ReorientRuntimeState:
  state = getattr(env, "_reorient_runtime_state", None)
  if state is None:
    state = ReorientRuntimeState(
      pert_force_dir=torch.zeros((env.num_envs, 3), device=env.device),
      pert_velocity_cache=torch.zeros((env.num_envs, 6), device=env.device),
    )
    env._reorient_runtime_state = state
  return state


class FingertipContactBinding:
  def __init__(self, env, sensor, site_ids: list[int], site_names: tuple[str, ...]):
    self.site_ids = site_ids
    self.site_names = site_names
    self.geom_names = tuple(sensor.primary_names)
    context = f"geom names={self.geom_names}, tip site names={self.site_names}"
    site_fingers = [
      re.fullmatch(r"(.+_finger[1-5])_tip", name.rsplit("/", 1)[-1])
      for name in self.site_names
    ]
    geom_fingers = [
      re.fullmatch(
        r"(.+_finger[1-5])_(?:link4_col|tip_sensor_col)", name.rsplit("/", 1)[-1]
      )
      for name in self.geom_names
    ]
    if len(site_fingers) != 5 or not all(site_fingers) or not all(geom_fingers):
      raise ValueError(f"Cannot bind fingertip contacts: {context}")
    sites = [match.group(1) for match in site_fingers]
    geoms = [match.group(1) for match in geom_fingers]
    if len(set(sites)) != 5 or set(geoms) != set(sites):
      raise ValueError(f"Fingertip geom/site finger sets do not match: {context}")
    num_slots = sensor.cfg.num_slots
    columns = [
      [
        geom_index * num_slots + slot
        for geom_index, geom_finger in enumerate(geoms)
        if geom_finger == finger
        for slot in range(num_slots)
      ]
      for finger in sites
    ]
    width = max(map(len, columns))
    self._indices = torch.tensor(
      [group + [group[0]] * (width - len(group)) for group in columns],
      dtype=torch.long,
      device=env.device,
    )
    self._contact_shape = (env.num_envs, len(geoms) * num_slots)

  def contact_mask(self, contact: torch.Tensor) -> torch.Tensor:
    if contact.shape != self._contact_shape:
      raise ValueError(
        f"Contact shape {tuple(contact.shape)} does not match bound geom columns "
        f"{self._contact_shape}; geom names={self.geom_names}, "
        f"tip site names={self.site_names}"
      )
    return contact[:, self._indices].any(dim=-1)


def get_fingertip_contact_binding(
  env, robot_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg
) -> FingertipContactBinding:
  robot = env.scene[robot_cfg.name]
  if isinstance(robot_cfg.site_ids, slice):
    site_ids = list(range(len(robot.site_names)))[robot_cfg.site_ids]
  else:
    site_ids = list(robot_cfg.site_ids)
  key = (robot_cfg.name, sensor_cfg.name, tuple(site_ids))
  bindings = get_reorient_runtime_state(env).fingertip_bindings
  if key not in bindings:
    bindings[key] = FingertipContactBinding(
      env,
      env.scene[sensor_cfg.name],
      site_ids,
      tuple(robot.site_names[index] for index in site_ids),
    )
  return bindings[key]
