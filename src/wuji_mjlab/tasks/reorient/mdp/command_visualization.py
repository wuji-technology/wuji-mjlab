# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.viewer.debug_visualizer import NullDebugVisualizer
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer

if TYPE_CHECKING:
  import mujoco
  import viser
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_GOAL_VIS_Z_OFFSET = 0.15
_STATUS_VIS_X_OFFSET = 0.06
_STATUS_VIS_RADIUS = 0.012
_FRAME_TRIAD_LEN = 0.10
_FRAME_TRIAD_RADIUS = 0.005
_TAG_TRIAD_COLORS = (
  (0.0, 0.85, 0.85),
  (0.85, 0.0, 0.85),
  (0.9, 0.85, 0.0),
)
_GOAL_RGBA = np.array([1.0, 1.0, 1.0, 0.5], dtype=np.float32)
_GHOST_BOX_HALF = np.array([0.027, 0.027, 0.027], dtype=np.float64)
_WARNED_VISUALIZER_TYPES: set[type] = set()


def native_visualization_context(
  visualizer: "DebugVisualizer",
) -> tuple["mujoco.MjModel", "mujoco.MjvScene"] | None:
  if isinstance(visualizer, MujocoNativeDebugVisualizer):
    # mjlab 1.6 exposes no public scene/material API for textured debug boxes.
    return visualizer._ref_mj_model, visualizer._mjv_scene
  return None


def goal_vis_position_above_object(
  object_pos: np.ndarray,
  z_offset: float = _GOAL_VIS_Z_OFFSET,
) -> np.ndarray:
  vis_pos = np.array(object_pos, copy=True)
  vis_pos[2] += z_offset
  return vis_pos


def goal_vis_env_indices(num_envs: int) -> list[int]:
  return list(range(num_envs))


def reorient_status_color_rgba(success: bool) -> tuple[float, float, float, float]:
  if success:
    return (0.2, 0.9, 0.2, 0.95)
  return (0.95, 0.2, 0.2, 0.95)


def format_reorient_status_markdown(ori_error_rad: float, success: bool) -> str:
  ori_error_deg = float(np.degrees(ori_error_rad))
  return (
    "### Reorient Policy Status\n"
    f"- ori_error: {ori_error_deg:.1f} deg\n"
    f"- success: {success}\n"
  )


def update_reorient_status_markdown(
  status_markdown,
  get_env_idx: Callable[[], int],
  num_envs: int,
  policy_status_for_env: Callable[[int], tuple[float, bool]],
) -> None:
  if num_envs <= 0:
    return
  env_idx = int(np.clip(get_env_idx(), 0, num_envs - 1))
  ori_error_rad, is_success = policy_status_for_env(env_idx)
  status_markdown.content = format_reorient_status_markdown(
    ori_error_rad=ori_error_rad,
    success=is_success,
  )


def draw_reorient_frame_triads(
  *,
  visualizer: "DebugVisualizer",
  num_envs: int,
  palm_pose_w: "torch.Tensor",
  tag_pose_w: "torch.Tensor",
) -> None:
  palm_pos_t = palm_pose_w[:, 0:3]
  palm_quat_t = palm_pose_w[:, 3:7]
  palm_R = matrix_from_quat(palm_quat_t).detach().cpu().numpy()
  palm_pos = palm_pos_t.detach().cpu().numpy()

  tag_pos_t = tag_pose_w[:, 0:3]
  tag_quat_t = tag_pose_w[:, 3:7]
  tag_R = matrix_from_quat(tag_quat_t).detach().cpu().numpy()
  tag_pos = tag_pos_t.detach().cpu().numpy()

  for env_idx in goal_vis_env_indices(num_envs):
    visualizer.add_frame(
      position=palm_pos[env_idx].astype(np.float32),
      rotation_matrix=palm_R[env_idx].astype(np.float32),
      scale=_FRAME_TRIAD_LEN,
      axis_radius=_FRAME_TRIAD_RADIUS,
      alpha=1.0,
      label=f"palm_frame_env{env_idx}",
    )
    visualizer.add_frame(
      position=tag_pos[env_idx].astype(np.float32),
      rotation_matrix=tag_R[env_idx].astype(np.float32),
      scale=_FRAME_TRIAD_LEN,
      axis_radius=_FRAME_TRIAD_RADIUS,
      alpha=1.0,
      axis_colors=_TAG_TRIAD_COLORS,
      label=f"tag_frame_env{env_idx}",
    )


@dataclass
class ReorientCommandVisualization:
  entity_name: str
  ghost_mat_id: int = -1
  ghost_box_size: np.ndarray = field(default_factory=lambda: _GHOST_BOX_HALF.copy())
  _ghost_resolved: bool = False
  status_markdown: object | None = None
  get_env_idx: Callable[[], int] | None = None

  def create_gui(
    self,
    *,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    with server.gui.add_folder(name.capitalize()):
      self.status_markdown = server.gui.add_markdown("")
    self.get_env_idx = get_env_idx

  def attach_status_targets(
    self,
    *,
    status_markdown,
    get_env_idx: Callable[[], int],
  ) -> None:
    self.status_markdown = status_markdown
    self.get_env_idx = get_env_idx

  def update_status_gui(
    self,
    *,
    num_envs: int,
    policy_status_for_env: Callable[[int], tuple[float, bool]],
  ) -> None:
    if self.status_markdown is None or self.get_env_idx is None:
      return
    update_reorient_status_markdown(
      status_markdown=self.status_markdown,
      get_env_idx=self.get_env_idx,
      num_envs=num_envs,
      policy_status_for_env=policy_status_for_env,
    )

  def _ensure_ghost_material(self, mj_model) -> bool:
    if self._ghost_resolved:
      return self.ghost_mat_id >= 0
    self._ghost_resolved = True

    import mujoco

    cube_body = mujoco.mj_name2id(
      mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{self.entity_name}/cube"
    )

    for geom_idx in range(mj_model.ngeom):
      if (
        mj_model.geom_bodyid[geom_idx] == cube_body
        and mj_model.geom_group[geom_idx] == 2
      ):
        self.ghost_mat_id = int(mj_model.geom_matid[geom_idx])
        self.ghost_box_size = mj_model.geom_size[geom_idx].copy()
        if self.ghost_mat_id >= 0:
          return True
        break
    warnings.warn(
      f"No visual material found for {self.entity_name}/cube; "
      "the textured reorient goal cannot be drawn.",
      RuntimeWarning,
      stacklevel=2,
    )
    return False

  def draw_debug_visuals(
    self,
    *,
    visualizer: "DebugVisualizer",
    num_envs: int,
    palm_pose_w: "torch.Tensor",
    tag_pose_w: "torch.Tensor",
    object_pos_w: "torch.Tensor",
    goal_quat_w: "torch.Tensor",
    policy_status_for_env: Callable[[int], tuple[float, bool]],
  ) -> None:
    draw_reorient_frame_triads(
      visualizer=visualizer,
      num_envs=num_envs,
      palm_pose_w=palm_pose_w,
      tag_pose_w=tag_pose_w,
    )

    context = native_visualization_context(visualizer)
    if context is None:
      if (
        not isinstance(visualizer, NullDebugVisualizer)
        and type(visualizer) not in _WARNED_VISUALIZER_TYPES
      ):
        _WARNED_VISUALIZER_TYPES.add(type(visualizer))
        warnings.warn(
          f"{type(visualizer).__name__} cannot draw the textured reorient goal; "
          "use the native MuJoCo viewer.",
          RuntimeWarning,
          stacklevel=2,
        )
      return

    mj_model, mj_scene = context
    if not self._ensure_ghost_material(mj_model):
      return

    import mujoco

    goal_rot = matrix_from_quat(goal_quat_w)

    for env_idx in goal_vis_env_indices(num_envs):
      if mj_scene.ngeom >= mj_scene.maxgeom:
        break

      ori_error_rad, is_success = policy_status_for_env(env_idx)
      vis_pos = goal_vis_position_above_object(
        object_pos_w[env_idx].detach().cpu().numpy()
      )
      geom = mj_scene.geoms[mj_scene.ngeom]
      mj_scene.ngeom += 1
      geom.category = mujoco.mjtCatBit.mjCAT_DECOR
      mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_BOX.value,
        self.ghost_box_size,
        vis_pos.astype(np.float64),
        goal_rot[env_idx].detach().cpu().numpy().reshape(-1).astype(np.float64),
        _GOAL_RGBA,
      )
      # Same arucocube material as the live cube; the cube texture samples by
      # face direction on a box, so no mesh dataid / texcoords are needed.
      geom.matid = self.ghost_mat_id

      status_pos = vis_pos.copy()
      status_pos[0] += _STATUS_VIS_X_OFFSET
      visualizer.add_sphere(
        center=status_pos,
        radius=_STATUS_VIS_RADIUS,
        color=reorient_status_color_rgba(is_success),
        label=(
          f"reorient_status_env{env_idx}"
          f"_err_{np.degrees(ori_error_rad):.1f}deg"
          f"_success_{is_success}"
        ),
      )
