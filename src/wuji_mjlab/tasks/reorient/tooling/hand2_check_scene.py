# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Shared scene helpers for the interactive Wuji Hand 2 reorient checks."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
  REORIENT_HAND2_CUBE_INIT_POS,
  REORIENT_HAND2_JOINT_POS,
  add_frame_axes,
)


def _namespace_mesh_files(robot: mujoco.MjSpec, namespace: str) -> None:
  source_assets = dict(robot.assets)
  mesh_filenames = {Path(mesh.file).name for mesh in robot.meshes}
  assets_by_filename = {
    Path(key).name: value
    for key, value in source_assets.items()
    if Path(key).name in mesh_filenames
  }
  renamed_assets = {
    key: value
    for key, value in source_assets.items()
    if Path(key).name not in mesh_filenames
  }

  for mesh in robot.meshes:
    original_filename = Path(mesh.file).name
    try:
      asset_bytes = assets_by_filename[original_filename]
    except KeyError as exc:
      raise ValueError(f"mesh asset bytes missing for {mesh.file!r}") from exc
    mesh.file = f"{namespace}_{original_filename}"
    asset_key = f"{robot.meshdir}/{mesh.file}" if robot.meshdir else mesh.file
    renamed_assets[asset_key] = asset_bytes
  robot.assets = renamed_assets


def _add_cube_reference(
  scene: mujoco.MjSpec,
  *,
  namespace: str,
  x_shift: float,
) -> None:
  cube = scene.worldbody.add_body(
    name=f"{namespace}_cube_reference",
    pos=np.asarray(REORIENT_HAND2_CUBE_INIT_POS["right"])
    + np.array((x_shift, 0.0, 0.0)),
  )
  cube.add_geom(
    name=f"{namespace}_cube_reference",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.027, 0.027, 0.027),
    rgba=(1.0, 0.45, 0.1, 0.65),
    contype=0,
    conaffinity=0,
    group=1,
    density=0,
  )


def attach_robot(
  scene: mujoco.MjSpec,
  robot: mujoco.MjSpec,
  *,
  prefix: str,
  root_pos: tuple[float, float, float],
  root_rot: tuple[float, float, float, float],
  x_shift: float,
) -> None:
  add_frame_axes(
    robot.body("right_palm_link"),
    half_length=0.05,
    half_width=0.0025,
    origin_radius=0.006,
  )
  # MuJoCo caches file-backed meshes by basename during a combined compile.
  # Distinct basenames prevent the second hand from reusing the first hand's STL.
  _namespace_mesh_files(robot, prefix.rstrip("/"))
  while robot.keys:
    robot.delete(robot.keys[0])
  frame = scene.worldbody.add_frame()
  frame.pos = np.asarray(root_pos) + np.array((x_shift, 0.0, 0.0))
  frame.quat = np.asarray(root_rot)
  scene.attach(robot, prefix=prefix, frame=frame)
  _add_cube_reference(
    scene,
    namespace=prefix.rstrip("/"),
    x_shift=x_shift,
  )


def add_viewer_environment(spec: mujoco.MjSpec) -> None:
  skybox = spec.add_texture()
  skybox.name = "skybox"
  skybox.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
  skybox.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
  skybox.rgb1 = np.array((0.3, 0.5, 0.9))
  skybox.rgb2 = np.array((0.9, 0.95, 1.0))
  skybox.width = 800
  skybox.height = 800

  ground_texture = spec.add_texture()
  ground_texture.name = "groundplane"
  ground_texture.type = mujoco.mjtTexture.mjTEXTURE_2D
  ground_texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
  ground_texture.mark = mujoco.mjtMark.mjMARK_EDGE
  ground_texture.rgb1 = np.array((0.2, 0.3, 0.4))
  ground_texture.rgb2 = np.array((0.1, 0.2, 0.3))
  ground_texture.markrgb = np.array((0.8, 0.8, 0.8))
  ground_texture.width = 300
  ground_texture.height = 300

  ground_material = spec.add_material()
  ground_material.name = "groundplane"
  ground_material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
  ground_material.texrepeat = np.array((5.0, 5.0))
  ground_material.texuniform = True
  ground_material.reflectance = 0.2

  spec.visual.headlight.diffuse = np.array((0.8, 0.8, 0.8))
  spec.visual.headlight.ambient = np.array((0.2, 0.2, 0.2))
  spec.visual.headlight.specular = np.array((1.0, 1.0, 1.0))
  spec.visual.global_.azimuth = 120.0
  spec.visual.global_.elevation = -20.0
  spec.visual.quality.shadowsize = 8192

  light = spec.worldbody.add_light()
  light.pos = np.array((0.0, 0.0, 1.5))
  light.dir = np.array((0.0, 0.0, -1.0))
  light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

  floor = spec.worldbody.add_geom()
  floor.name = "floor"
  floor.type = mujoco.mjtGeom.mjGEOM_PLANE
  floor.size = np.array((5.0, 5.0, 0.05))
  floor.material = "groundplane"
  floor.contype = 1
  floor.conaffinity = 1


def _set_home_qpos(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  prefixes: tuple[str, ...],
) -> None:
  data.qpos[:] = model.qpos0
  for prefix in prefixes:
    for pattern, value in REORIENT_HAND2_JOINT_POS.items():
      joint_name = f"{prefix}right_{pattern.removeprefix('.*_')}"
      joint = model.joint(joint_name)
      data.qpos[joint.qposadr[0]] = value


def hold_home_pose(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  prefixes: tuple[str, ...],
) -> None:
  _set_home_qpos(model, data, prefixes)
  for actuator_id in range(model.nu):
    joint_id = model.actuator_trnid[actuator_id, 0]
    data.ctrl[actuator_id] = data.qpos[model.jnt_qposadr[joint_id]]
