# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Interactive Wuji Hand 2 collision-capsule and mesh-convex-hull check."""

from __future__ import annotations

import re
from colorsys import hsv_to_rgb
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

from wuji_mjlab.tasks.reorient.config.wuji_hand2.env_cfgs import (
  get_wuji_hand2_rig_cfg,
)
from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
  HAND2_REORIENT_CAPSULES,
  REORIENT_HAND2_ROOT_POS,
  REORIENT_HAND2_ROOT_ROT,
)
from wuji_mjlab.tasks.reorient.tooling.hand2_check_scene import (
  add_viewer_environment,
  attach_robot,
  hold_home_pose,
)

_ACTIVE_RGBA = np.array((0.05, 0.95, 0.25, 1.0))
_HIDDEN_RGBA = np.array((0.0, 0.0, 0.0, 0.0))


@dataclass(frozen=True)
class CapsuleOverlayCheckConfig:
  """Interactive collision-overlay options."""

  mesh_opacity: float = 0.22
  hull_opacity: float = 0.35


def _style_robot(robot: mujoco.MjSpec, mesh_opacity: float) -> None:
  if not 0.0 < mesh_opacity < 1.0:
    raise ValueError(f"mesh_opacity must be in (0, 1), got {mesh_opacity}")

  capsule_names = {f"{body_name}_col" for body_name in HAND2_REORIENT_CAPSULES}
  for geom in robot.geoms:
    if geom.name in capsule_names:
      if geom.contype and geom.conaffinity:
        geom.rgba = _ACTIVE_RGBA
        geom.group = 1
      else:
        geom.rgba = _HIDDEN_RGBA
    elif geom.type == mujoco.mjtGeom.mjGEOM_MESH:
      if geom.group == 1:
        geom.rgba = (0.75, 0.75, 0.75, mesh_opacity)
      else:
        rgba = np.asarray(geom.rgba).copy()
        rgba[3] = 0.0
        geom.rgba = rgba


def _collision_mesh_ids(model: mujoco.MjModel) -> list[int]:
  return [
    geom_id
    for geom_id in range(model.ngeom)
    if model.body(model.geom_bodyid[geom_id]).name.startswith("robot/")
    and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
    and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
  ]


def _mesh_category(name: str) -> str:
  name = name.rsplit("/", 1)[-1]
  if "_palm_" in name:
    return "palm"
  return re.sub(r"^(left|right)_(finger\d+_)?", "", name).removesuffix("_col")


def _mesh_hull(model: mujoco.MjModel, mesh_id: int) -> tuple[np.ndarray, np.ndarray]:
  graph_start = model.mesh_graphadr[mesh_id]
  if graph_start < 0:
    raise ValueError(f"collision mesh {model.mesh(mesh_id).name!r} has no convex hull")
  graph = model.mesh_graph[graph_start:]
  nvert, nface = graph[:2]
  vertex_ids = graph[2 + nvert : 2 + 2 * nvert]
  face_start = 2 + 3 * nvert + 3 * nface
  faces = graph[face_start : face_start + 3 * nface].reshape(-1, 3)
  indices = np.full(model.mesh_vertnum[mesh_id], -1, dtype=np.int32)
  indices[vertex_ids] = np.arange(nvert)
  return model.mesh_vert[model.mesh_vertadr[mesh_id] + vertex_ids], indices[faces]


def _add_collision_hulls(
  scene: mujoco.MjSpec, model: mujoco.MjModel, hull_opacity: float
) -> None:
  geom_ids = _collision_mesh_ids(model)
  categories = sorted({_mesh_category(model.geom(i).name) for i in geom_ids})
  other_categories = [c for c in categories if c not in ("tip_sensor", "palm")]
  colors = {
    "tip_sensor": (1.0, 0.45, 0.05),
    "palm": (0.85, 0.15, 1.0),
    **{
      category: hsv_to_rgb(0.5 + 0.17 * i / len(other_categories), 0.9, 1.0)
      for i, category in enumerate(other_categories)
    },
  }
  # Inserting geoms invalidates MjSpec's compiled name-to-index lookup.
  for geom_id in geom_ids:
    scene.geom(model.geom(geom_id).name).rgba = _HIDDEN_RGBA
  for geom_id in geom_ids:
    name = model.geom(geom_id).name
    vertices, faces = _mesh_hull(model, model.geom_dataid[geom_id])
    overlay_name = f"{name}_hull_overlay"
    scene.add_mesh(
      name=overlay_name,
      uservert=vertices.ravel(),
      userface=faces.ravel(),
      smoothnormal=False,
    )
    # Compiled vertices and geom poses already include mesh scale and recentering.
    scene.body(model.body(model.geom_bodyid[geom_id]).name).add_geom(
      name=overlay_name,
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=overlay_name,
      pos=model.geom_pos[geom_id],
      quat=model.geom_quat[geom_id],
      rgba=(*colors[_mesh_category(name)], hull_opacity),
      material="",
      contype=0,
      conaffinity=0,
      group=1,
      mass=0,
      density=0,
    )


def _print_mesh_legend(model: mujoco.MjModel) -> None:
  print("Collision mesh convex hulls (RGBA; vertices/faces):")
  for geom_id in _collision_mesh_ids(model):
    name = model.geom(geom_id).name
    graph_start = model.mesh_graphadr[model.geom_dataid[geom_id]]
    nvert, nface = model.mesh_graph[graph_start : graph_start + 2]
    rgba = model.geom(f"{name}_hull_overlay").rgba
    color = ", ".join(f"{value:.2f}" for value in rgba)
    print(f"  {name} | {_mesh_category(name)} | ({color}) | {nvert}/{nface}")


def build_capsule_overlay_check_model(
  mesh_opacity: float = 0.22,
  hull_opacity: float = 0.35,
) -> mujoco.MjModel:
  """Compile highlighted capsules and the collision meshes' actual convex hulls."""
  if not 0.0 < hull_opacity < 1.0:
    raise ValueError(f"hull_opacity must be in (0, 1), got {hull_opacity}")
  scene = mujoco.MjSpec()
  scene.option.timestep = 0.01
  robot = get_wuji_hand2_rig_cfg("right").spec_fn()
  _style_robot(robot, mesh_opacity)
  attach_robot(
    scene,
    robot,
    prefix="robot/",
    root_pos=REORIENT_HAND2_ROOT_POS,
    root_rot=REORIENT_HAND2_ROOT_ROT,
    x_shift=0.0,
  )
  add_viewer_environment(scene)
  model = scene.compile()
  _add_collision_hulls(scene, model, hull_opacity)
  model = scene.compile()

  active = 0
  disabled = 0
  for body_name in HAND2_REORIENT_CAPSULES:
    geom = model.geom(f"robot/{body_name}_col")
    if model.geom_type[geom.id] != mujoco.mjtGeom.mjGEOM_CAPSULE:
      raise ValueError(f"{geom.name!r} is not a capsule")
    if model.geom_contype[geom.id] and model.geom_conaffinity[geom.id]:
      active += 1
    else:
      disabled += 1
  if (active, disabled) != (10, 5):
    raise ValueError(
      f"expected 10 active and 5 disabled capsules, got {active}/{disabled}"
    )
  _print_mesh_legend(model)
  return model


def launch_capsule_overlay_check(config: CapsuleOverlayCheckConfig) -> None:
  """Launch the native interactive capsule-overlay viewer."""
  model = build_capsule_overlay_check_model(
    mesh_opacity=config.mesh_opacity, hull_opacity=config.hull_opacity
  )
  data = mujoco.MjData(model)
  hold_home_pose(model, data, ("robot/",))
  mujoco.mj_forward(model, data)

  print(
    "Compiled capsule-overlay check: pose=home, "
    f"mesh_opacity={config.mesh_opacity:g}, "
    f"hull_opacity={config.hull_opacity:g}, capsules=15"
  )
  print(
    "Legend: GREEN = 10 active L2/L3 capsules; the 5 disabled L1 capsules are hidden."
  )
  print("Criterion: each colored collision shape should hug its gray visual mesh.")
  mujoco.viewer.launch(model, data)
