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

if TYPE_CHECKING:
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")
_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_CAGE_COUNTER_ATTR = "_cage_penalty_counter"
_CAGE_Z_UPPER_MARGIN = 0.03


def _compute_cage_bounds_in_palm(
  hand_in_palm: torch.Tensor,
  margin: float,
  z_upper_margin: float = _CAGE_Z_UPPER_MARGIN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  xy_min = hand_in_palm[:, :, :2].min(dim=1).values - margin
  xy_max = hand_in_palm[:, :, :2].max(dim=1).values + margin
  z_min = hand_in_palm[:, :, 2].min(dim=1).values - margin
  z_max = hand_in_palm[:, :, 2].max(dim=1).values + z_upper_margin
  return xy_min, xy_max, z_min, z_max


def _compute_cage_aabb_escape_in_palm(
  cube_in_palm: torch.Tensor,
  hand_in_palm: torch.Tensor,
  margin: float,
  z_upper_margin: float = _CAGE_Z_UPPER_MARGIN,
) -> torch.Tensor:
  """Pure-tensor AABB escape distance in palm frame."""
  xy_min, xy_max, z_min, z_max = _compute_cage_bounds_in_palm(
    hand_in_palm, margin=margin, z_upper_margin=z_upper_margin
  )

  cx, cy, cz = cube_in_palm[:, 0], cube_in_palm[:, 1], cube_in_palm[:, 2]
  dx = torch.relu(xy_min[:, 0] - cx) + torch.relu(cx - xy_max[:, 0])
  dy = torch.relu(xy_min[:, 1] - cy) + torch.relu(cy - xy_max[:, 1])
  dz = torch.relu(z_min - cz) + torch.relu(cz - z_max)
  return dx + dy + dz


def _compute_cage_escape_dist(
  env,
  object_cfg: SceneEntityCfg,
  robot_cfg: SceneEntityCfg,
  margin: float,
) -> torch.Tensor:
  """Compute how far the cube has escaped the hand AABB cage in palm frame."""
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
    cube_in_palm, hand_in_palm, margin=margin, z_upper_margin=_CAGE_Z_UPPER_MARGIN
  )


_CAGE_RGBA_INSIDE = (0.2, 0.9, 0.2, 0.25)
_CAGE_RGBA_OUTSIDE = (0.95, 0.2, 0.2, 0.25)


def update_cage_penalty_counter(
  counter: torch.Tensor,
  outside: torch.Tensor,
  decay_rate: float = 0.5,
  max_count: float = 15.0,
) -> torch.Tensor:
  """Update soft penalty counter: +1 outside, gradual decay inside."""
  result = torch.where(outside, counter + 1, (counter - decay_rate).clamp(min=0))
  return result.clamp(max=max_count)


def compute_cage_escalation(
  counter: torch.Tensor,
  outside: torch.Tensor,
  max_outside_steps: int = 10,
  drop_scale: float = 4.0,
) -> torch.Tensor:
  """Compute time-escalating penalty. Only non-zero when outside."""
  max_outside_steps = max(int(max_outside_steps), 1)
  return outside.float() * counter / float(max_outside_steps) * drop_scale


def read_cage_penalty_counter(env):
  return getattr(env, _CAGE_COUNTER_ATTR, None)


class CageEscapePenalty(ManagerTermBase):
  """Reward term that penalizes cube escape from the palm AABB cage."""

  def __init__(self, cfg, env):
    super().__init__(env)
    if cfg.weight == 0.0:
      raise ValueError(
        "CageEscapePenalty weight must not be 0: cage_drop termination "
        "depends on the counter updated by this reward term."
      )
    params = cfg.params
    self._object_cfg: SceneEntityCfg = params.get("object_cfg", _DEFAULT_OBJECT_CFG)
    self._robot_cfg: SceneEntityCfg = params.get("robot_cfg", _DEFAULT_ROBOT_CFG)
    self._margin: float = params.get("margin", 0.01)
    self._max_outside_steps: int = params.get("max_outside_steps", 10)
    self._drop_scale: float = params.get("drop_scale", 4.0)
    self._counter_decay_rate: float = params.get("counter_decay_rate", 0.5)
    self._penalty_counter = torch.zeros(env.num_envs, device=env.device)
    self._debug_vis_enabled = True

  def reset(self, env_ids) -> None:
    self._penalty_counter[env_ids] = 0
    setattr(self._env, _CAGE_COUNTER_ATTR, self._penalty_counter)

  def __call__(
    self,
    env,
    object_cfg: SceneEntityCfg | None = None,
    robot_cfg: SceneEntityCfg | None = None,
    margin: float | None = None,
    max_outside_steps: int | None = None,
    drop_scale: float | None = None,
  ) -> torch.Tensor:
    object_cfg = self._object_cfg if object_cfg is None else object_cfg
    robot_cfg = self._robot_cfg if robot_cfg is None else robot_cfg
    margin = self._margin if margin is None else margin
    max_outside_steps = (
      self._max_outside_steps if max_outside_steps is None else max_outside_steps
    )
    drop_scale = self._drop_scale if drop_scale is None else drop_scale

    escape_dist = _compute_cage_escape_dist(env, object_cfg, robot_cfg, margin)
    outside = escape_dist > 0

    self._penalty_counter = update_cage_penalty_counter(
      self._penalty_counter, outside, self._counter_decay_rate
    )
    setattr(env, _CAGE_COUNTER_ATTR, self._penalty_counter)

    return compute_cage_escalation(
      self._penalty_counter, outside, max_outside_steps, drop_scale
    )

  def debug_vis(self, visualizer: "DebugVisualizer") -> None:
    if not self._debug_vis_enabled:
      return
    if not hasattr(visualizer, "mj_model") or not hasattr(visualizer, "scn"):
      return
    import mujoco

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

    xy_min, xy_max, z_min, z_max = _compute_cage_bounds_in_palm(
      hand_in_palm, margin=self._margin, z_upper_margin=_CAGE_Z_UPPER_MARGIN
    )
    escape = _compute_cage_aabb_escape_in_palm(
      cube_in_palm,
      hand_in_palm,
      margin=self._margin,
      z_upper_margin=_CAGE_Z_UPPER_MARGIN,
    )

    palm_rot = matrix_from_quat(palm_quat)

    mj_scene = visualizer.scn
    for env_idx in range(env.num_envs):
      if mj_scene.ngeom >= mj_scene.maxgeom:
        break
      xmin = xy_min[env_idx, 0].item()
      xmax = xy_max[env_idx, 0].item()
      ymin = xy_min[env_idx, 1].item()
      ymax = xy_max[env_idx, 1].item()
      zmin = z_min[env_idx].item()
      zmax = z_max[env_idx].item()

      center_palm = np.array(
        [
          0.5 * (xmin + xmax),
          0.5 * (ymin + ymax),
          0.5 * (zmin + zmax),
        ],
        dtype=np.float64,
      )
      half_size = np.array(
        [
          0.5 * (xmax - xmin),
          0.5 * (ymax - ymin),
          0.5 * (zmax - zmin),
        ],
        dtype=np.float64,
      )

      rot = palm_rot[env_idx].detach().cpu().numpy().astype(np.float64)
      palm_pos_np = palm_pos[env_idx].detach().cpu().numpy().astype(np.float64)
      center_world = palm_pos_np + rot @ center_palm

      rgba = np.array(
        _CAGE_RGBA_OUTSIDE if escape[env_idx].item() > 0 else _CAGE_RGBA_INSIDE,
        dtype=np.float32,
      )

      geom = mj_scene.geoms[mj_scene.ngeom]
      mj_scene.ngeom += 1
      mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_BOX.value,
        half_size,
        center_world,
        rot.reshape(-1),
        rgba,
      )
      geom.category = mujoco.mjtCatBit.mjCAT_DECOR


def cage_drop(
  env,
  max_outside_steps: int = 10,
) -> torch.Tensor:
  """Terminate when the shared cage counter reaches threshold."""
  counter = read_cage_penalty_counter(env)
  if counter is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return counter >= max_outside_steps
