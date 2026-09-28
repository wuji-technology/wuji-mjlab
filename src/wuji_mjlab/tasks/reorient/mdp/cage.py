# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse
from mjlab.viewer.debug_visualizer import NullDebugVisualizer

from wuji_mjlab.tasks.reorient.mdp.command_visualization import (
  native_visualization_context,
)
from wuji_mjlab.tasks.reorient.reorient_constants import REORIENT_PALM_NORMAL_AXIS

if TYPE_CHECKING:
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")
_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_CAGE_COUNTER_ATTR = "_cage_penalty_counter"
_CAGE_UP_MARGIN = 0.03
_DEFAULT_CAGE_UP_AXIS = REORIENT_PALM_NORMAL_AXIS


def _compute_cage_bounds_in_palm(
  hand_in_palm: torch.Tensor,
  margin: float,
  up_margin: float = _CAGE_UP_MARGIN,
  up_axis: int = _DEFAULT_CAGE_UP_AXIS,
) -> tuple[torch.Tensor, torch.Tensor]:
  lo = hand_in_palm.min(dim=1).values - margin
  hi = hand_in_palm.max(dim=1).values + margin
  hi[:, up_axis] = hand_in_palm[:, :, up_axis].max(dim=1).values + up_margin
  return lo, hi


def _compute_cage_aabb_escape_in_palm(
  cube_in_palm: torch.Tensor,
  hand_in_palm: torch.Tensor,
  margin: float,
  up_margin: float = _CAGE_UP_MARGIN,
  up_axis: int = _DEFAULT_CAGE_UP_AXIS,
) -> torch.Tensor:
  lo, hi = _compute_cage_bounds_in_palm(
    hand_in_palm, margin=margin, up_margin=up_margin, up_axis=up_axis
  )
  return (torch.relu(lo - cube_in_palm) + torch.relu(cube_in_palm - hi)).sum(dim=-1)


def _compute_cage_escape_dist(
  env,
  object_cfg: SceneEntityCfg,
  robot_cfg: SceneEntityCfg,
  margin: float,
  up_axis: int,
) -> torch.Tensor:
  obj: Entity = env.scene[object_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]

  palm_id = robot_cfg.body_ids[0]
  palm_pos = robot.data.body_link_pose_w[:, palm_id, :3]
  palm_quat = robot.data.body_link_pose_w[:, palm_id, 3:7]

  hand_ids = robot_cfg.body_ids
  hand_pos_w = robot.data.body_link_pose_w[:, hand_ids, :3]
  hand_rel = hand_pos_w - palm_pos.unsqueeze(1)
  palm_quat_b = palm_quat.unsqueeze(1).expand(-1, hand_rel.shape[1], -1)
  hand_in_palm = quat_apply_inverse(palm_quat_b, hand_rel)

  cube_pos_w = obj.data.root_link_pos_w
  cube_rel = cube_pos_w - palm_pos
  cube_in_palm = quat_apply_inverse(palm_quat, cube_rel)

  return _compute_cage_aabb_escape_in_palm(
    cube_in_palm,
    hand_in_palm,
    margin=margin,
    up_margin=_CAGE_UP_MARGIN,
    up_axis=up_axis,
  )


_CAGE_RGBA_INSIDE = (0.2, 0.9, 0.2, 0.25)
_CAGE_RGBA_OUTSIDE = (0.95, 0.2, 0.2, 0.25)


def update_cage_penalty_counter(
  counter: torch.Tensor,
  outside: torch.Tensor,
  decay_rate: float = 0.5,
  max_count: float = 15.0,
) -> torch.Tensor:
  result = torch.where(outside, counter + 1, (counter - decay_rate).clamp(min=0))
  return result.clamp(max=max_count)


def compute_cage_escalation(
  counter: torch.Tensor,
  outside: torch.Tensor,
  max_outside_steps: int = 10,
  drop_scale: float = 4.0,
) -> torch.Tensor:
  max_outside_steps = max(int(max_outside_steps), 1)
  return outside.float() * counter / float(max_outside_steps) * drop_scale


def read_cage_penalty_counter(env):
  return getattr(env, _CAGE_COUNTER_ATTR, None)


class CageEscapePenalty(ManagerTermBase):
  def __init__(self, cfg, env):
    super().__init__(env)
    self._debug_vis_enabled = True
    if cfg.weight == 0.0:
      raise ValueError(
        "CageEscapePenalty weight must not be 0: cage_drop termination "
        "depends on the counter updated by this reward term."
      )
    params = cfg.params
    self._object_cfg: SceneEntityCfg = params.get("object_cfg", _DEFAULT_OBJECT_CFG)
    self._robot_cfg: SceneEntityCfg = params.get("robot_cfg", _DEFAULT_ROBOT_CFG)
    self._margin: float = params.get("margin", 0.01)
    self._up_axis: int = params.get("up_axis", _DEFAULT_CAGE_UP_AXIS)
    self._max_outside_steps: int = params.get("max_outside_steps", 10)
    self._drop_scale: float = params.get("drop_scale", 4.0)
    self._counter_decay_rate: float = params.get("counter_decay_rate", 0.5)
    self._max_count: float = params.get("max_count", 15.0)
    if self._max_count < self._max_outside_steps:
      raise ValueError(
        f"CageEscapePenalty max_count ({self._max_count}) must be >= "
        f"max_outside_steps ({self._max_outside_steps}), else the counter caps "
        "below the cage_drop threshold and termination can never fire."
      )
    self._penalty_counter = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids) -> None:
    self._penalty_counter[env_ids] = 0
    setattr(self._env, _CAGE_COUNTER_ATTR, self._penalty_counter)

  def __call__(
    self,
    env,
    object_cfg: SceneEntityCfg | None = None,
    robot_cfg: SceneEntityCfg | None = None,
    margin: float | None = None,
    up_axis: int | None = None,
    max_outside_steps: int | None = None,
    drop_scale: float | None = None,
    counter_decay_rate: float | None = None,
    max_count: float | None = None,
  ) -> torch.Tensor:
    object_cfg = self._object_cfg if object_cfg is None else object_cfg
    robot_cfg = self._robot_cfg if robot_cfg is None else robot_cfg
    margin = self._margin if margin is None else margin
    up_axis = self._up_axis if up_axis is None else up_axis
    max_outside_steps = (
      self._max_outside_steps if max_outside_steps is None else max_outside_steps
    )
    drop_scale = self._drop_scale if drop_scale is None else drop_scale
    counter_decay_rate = (
      self._counter_decay_rate if counter_decay_rate is None else counter_decay_rate
    )
    max_count = self._max_count if max_count is None else max_count

    escape_dist = _compute_cage_escape_dist(env, object_cfg, robot_cfg, margin, up_axis)
    outside = escape_dist > 0

    self._penalty_counter = update_cage_penalty_counter(
      self._penalty_counter, outside, counter_decay_rate, max_count
    )
    setattr(env, _CAGE_COUNTER_ATTR, self._penalty_counter)

    return compute_cage_escalation(
      self._penalty_counter, outside, max_outside_steps, drop_scale
    )

  def debug_vis(self, visualizer: "DebugVisualizer") -> None:
    if not self._debug_vis_enabled or isinstance(visualizer, NullDebugVisualizer):
      return

    env = self._env
    obj: Entity = env.scene[self._object_cfg.name]
    robot: Entity = env.scene[self._robot_cfg.name]

    palm_id = self._robot_cfg.body_ids[0]
    palm_pos = robot.data.body_link_pose_w[:, palm_id, :3]
    palm_quat = robot.data.body_link_pose_w[:, palm_id, 3:7]

    hand_ids = self._robot_cfg.body_ids
    hand_pos_w = robot.data.body_link_pose_w[:, hand_ids, :3]
    hand_rel = hand_pos_w - palm_pos.unsqueeze(1)
    palm_quat_b = palm_quat.unsqueeze(1).expand(-1, hand_rel.shape[1], -1)
    hand_in_palm = quat_apply_inverse(palm_quat_b, hand_rel)

    cube_pos_w = obj.data.root_link_pos_w
    cube_rel = cube_pos_w - palm_pos
    cube_in_palm = quat_apply_inverse(palm_quat, cube_rel)

    lo, hi = _compute_cage_bounds_in_palm(
      hand_in_palm,
      margin=self._margin,
      up_margin=_CAGE_UP_MARGIN,
      up_axis=self._up_axis,
    )
    escape = _compute_cage_aabb_escape_in_palm(
      cube_in_palm,
      hand_in_palm,
      margin=self._margin,
      up_margin=_CAGE_UP_MARGIN,
      up_axis=self._up_axis,
    )

    palm_rot = matrix_from_quat(palm_quat)

    context = native_visualization_context(visualizer)
    for env_idx in range(env.num_envs):
      if context is not None and context[1].ngeom >= context[1].maxgeom:
        break
      lo_np = lo[env_idx].detach().cpu().numpy().astype(np.float64)
      hi_np = hi[env_idx].detach().cpu().numpy().astype(np.float64)
      center_palm = 0.5 * (lo_np + hi_np)
      half_size = 0.5 * (hi_np - lo_np)

      rot = palm_rot[env_idx].detach().cpu().numpy().astype(np.float64)
      palm_pos_np = palm_pos[env_idx].detach().cpu().numpy().astype(np.float64)
      center_world = palm_pos_np + rot @ center_palm

      rgba = _CAGE_RGBA_OUTSIDE if escape[env_idx].item() > 0 else _CAGE_RGBA_INSIDE
      visualizer.add_box(
        center=center_world,
        size=half_size,
        mat=rot,
        color=rgba,
        label=f"reorient_cage_env{env_idx}",
      )


def cage_drop(
  env,
  max_outside_steps: int = 10,
) -> torch.Tensor:
  counter = read_cage_penalty_counter(env)
  if counter is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return counter >= max_outside_steps
