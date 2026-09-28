#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Passive render viewer (process 3 of 3) — visualization only, no policy/hardware."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from wuji_reorient_deploy import config_loader, constants
from wuji_reorient_deploy.constants import load_constants
from wuji_reorient_deploy.math_utils import quat_apply, quat_mul
from wuji_reorient_deploy.repo_assets import REPO_ASSETS_DIR, load_hand_spec
from wuji_reorient_deploy.zmq_bridge import CubeReceiver, GoalReceiver, JointReceiver

_CUBE_HALF = 0.027  # 54 mm cube
_GOAL_OFFSET = np.array([0.0, 0.12, 0.0])

# MuJoCo consumes cube textures in [+X, -X, +Y, -Y, +Z, -Z] order, while the
# filenames encode physical-face labels rather than axis names.
_CUBE_FACES = [
  "RIGHT_purple",
  "LEFT_blue",
  "TOP_red",
  "BOTTOM_cyan",
  "FRONT_green",
  "BACK_black",
]


def _ry(deg: float) -> np.ndarray:
  """Quaternion (w,x,y,z) for a rotation about the world Y axis."""
  a = np.radians(deg) * 0.5
  return np.array([np.cos(a), 0.0, np.sin(a), 0.0], dtype=np.float64)


def _root_quat(const: constants.Constants, tilt_deg: float) -> np.ndarray:
  return quat_mul(_ry(-tilt_deg), const.root_quat)


def _pkg_assets_root() -> Path:
  assets = REPO_ASSETS_DIR
  if not assets.is_dir():
    raise RuntimeError(
      f"expected wuji_mjlab assets at {assets} (repo checkout layout: "
      "deploy/ and src/wuji_mjlab/ live under the same repo root). If this "
      "script was copied out of the wuji-mjlab checkout, put it back or "
      "run it from within the repo."
    )
  return assets


def _cube_tex_dir() -> Path:
  return _pkg_assets_root() / "objects" / "inhand_object" / "textures"


def _build_scene(
  const: constants.Constants, gen: int, hand_side: str, root_quat: np.ndarray
) -> mujoco.MjModel:
  spec = load_hand_spec(gen, hand_side)
  spec.visual.global_.offwidth = 1280
  spec.visual.global_.offheight = 960

  palm = spec.body(f"{hand_side}_palm_link")
  palm.pos = const.root_pos.tolist()
  palm.quat = root_quat.tolist()

  ghost_spec = load_hand_spec(gen, hand_side)
  ghost_palm = ghost_spec.body(f"{hand_side}_palm_link")
  ghost_palm.pos = const.root_pos.tolist()
  ghost_palm.quat = root_quat.tolist()
  ghost_mat = ghost_spec.add_material(name="hand_ghost")
  ghost_mat.rgba = [0.1, 0.8, 1.0, 0.25]
  for geom in ghost_spec.geoms:
    geom.material = "hand_ghost"
    geom.rgba = ghost_mat.rgba
    geom.group = 2
    geom.contype = 0
    geom.conaffinity = 0
  spec.attach(ghost_spec, prefix="target/", frame=spec.worldbody.add_frame())

  wb = spec.worldbody

  spec.add_texture(
    name="sky",
    type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
    rgb1=[0.3, 0.5, 0.7],
    rgb2=[0.0, 0.0, 0.05],
    width=512,
    height=512,
  )
  spec.add_texture(
    name="grid",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    rgb1=[0.2, 0.3, 0.4],
    rgb2=[0.1, 0.15, 0.2],
    width=512,
    height=512,
  )
  gmat = spec.add_material(name="grid")
  gmat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
  gmat.texrepeat = [6, 6]
  gmat.reflectance = 0.1

  cube_tex = _cube_tex_dir()
  ctex = spec.add_texture(name="cube", type=mujoco.mjtTexture.mjTEXTURE_CUBE)
  ctex.cubefiles = [str(cube_tex / f"{f}.png") for f in _CUBE_FACES]
  cmat = spec.add_material(name="cube")
  cmat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "cube"
  gcmat = spec.add_material(name="cube_ghost")
  gcmat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "cube"
  gcmat.rgba = [1.0, 1.0, 1.0, 0.5]

  wb.add_light(pos=[0, 0, 2.0], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7])
  wb.add_light(pos=[0.6, -0.6, 1.5], dir=[-0.4, 0.4, -1], diffuse=[0.4, 0.4, 0.4])
  # Wuji Hand 2 collision geoms inherit group 3, which the viewer hides; visualization
  # geoms must override the inherited group.
  wb.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[2, 2, 0.1],
    material="grid",
    group=2,
    contype=0,
    conaffinity=0,
  )

  L = 0.20
  axes = (
    ([1, 0, 0, 1], [L, 0, 0]),
    ([0, 1, 0, 1], [0, L, 0]),
    ([0, 0, 1, 1], [0, 0, L]),
  )
  for rgba, end in axes:
    wb.add_geom(
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      fromto=[0, 0, 0, *end],
      size=[0.007, 0, 0],
      rgba=rgba,
      group=2,
      contype=0,
      conaffinity=0,
    )

  # Wrist-tag frame triad — THE frame the cube pose (cube_pos_tag /
  # cube_quat_tag) is expressed in, so the reference to check cube obs against.
  tag_pos = const.root_pos + quat_apply(root_quat, const.tag_in_palm_pos)
  tag_quat = quat_mul(root_quat, const.tag_in_palm_quat)
  tagb = wb.add_body(name="tag_frame", pos=tag_pos.tolist(), quat=tag_quat.tolist())
  Lt = 0.08
  for rgba, end in (
    ([1, 0, 0, 1], [Lt, 0, 0]),
    ([0, 1, 0, 1], [0, Lt, 0]),
    ([0, 0, 1, 1], [0, 0, Lt]),
  ):
    tagb.add_geom(
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      fromto=[0, 0, 0, *end],
      size=[0.005, 0, 0],
      rgba=rgba,
      group=2,
      contype=0,
      conaffinity=0,
    )

  Lb = 0.12
  for rgba, end in (
    ([1, 0, 0, 1], [Lb, 0, 0]),
    ([0, 1, 0, 1], [0, Lb, 0]),
    ([0, 0, 1, 1], [0, 0, Lb]),
  ):
    palm.add_geom(
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      fromto=[0, 0, 0, *end],
      size=[0.006, 0, 0],
      rgba=rgba,
      group=2,
      contype=0,
      conaffinity=0,
    )

  wrist_pos = next((list(g.pos) for g in palm.geoms if "wrist" in (g.name or "")), None)
  if wrist_pos is None:
    wrist_pos = [0.000250158, 0.00300004, 0.0284999]
  wx, wy, wz = wrist_pos
  Lw = 0.10
  for rgba, end in (
    ([1, 0, 0, 1], [wx + Lw, wy, wz]),
    ([0, 1, 0, 1], [wx, wy + Lw, wz]),
    ([0, 0, 1, 1], [wx, wy, wz + Lw]),
  ):
    palm.add_geom(
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      fromto=[wx, wy, wz, *end],
      size=[0.0045, 0, 0],
      rgba=rgba,
      group=2,
      contype=0,
      conaffinity=0,
    )

  # Frame the default (and offscreen) camera to include BOTH the world-frame
  # triad at the origin (z=0) and the hand (z≈0.5) — not the whole 2 m floor.
  spec.stat.center = [0.0, 0.0, 0.28]
  spec.stat.extent = 0.75

  def _add_body_axes(body, L=0.055, r=0.004):
    # L > _CUBE_HALF, so the axes stick out past the cube faces.
    for rgba, end in (
      ([1, 0, 0, 1], [L, 0, 0]),
      ([0, 1, 0, 1], [0, L, 0]),
      ([0, 0, 1, 1], [0, 0, L]),
    ):
      body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0, 0, 0, *end],
        size=[r, 0, 0],
        rgba=rgba,
        group=2,
        contype=0,
        conaffinity=0,
      )

  cube = wb.add_body(name="obs_cube", pos=[0, 0, 0.6])
  cube.add_freejoint()
  cube.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[_CUBE_HALF] * 3,
    material="cube",
    group=2,
    contype=0,
    conaffinity=0,
  )
  _add_body_axes(cube)

  goal = wb.add_body(
    name="goal_cube", mocap=True, pos=(const.root_pos + _GOAL_OFFSET).tolist()
  )
  goal.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[_CUBE_HALF] * 3,
    material="cube_ghost",
    group=2,
    contype=0,
    conaffinity=0,
  )
  _add_body_axes(goal)

  wb.add_camera(name="view", pos=[0.5, -0.5, 0.8], xyaxes=[0.7, 0.7, 0, -0.3, 0.3, 0.9])
  return spec.compile()


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument(
    "--gen",
    type=int,
    choices=(1, 2),
    default=2,
    help="hand generation (default: 2)",
  )
  ap.add_argument("--hand", choices=["right"], required=True)
  ap.add_argument(
    "--tilt-deg",
    type=float,
    default=10.0,
    help="Downward pitch of the fixed hand mount (deg, about world Y)",
  )
  ap.add_argument(
    "--cube-port",
    type=int,
    default=None,
    help="cube pose SUB port (default: config/control.yaml)",
  )
  args = ap.parse_args()

  const = load_constants(args.gen, args.hand)
  root_quat = _root_quat(const, args.tilt_deg)
  model = _build_scene(const, args.gen, args.hand, root_quat)
  data = mujoco.MjData(model)

  jnames = constants.joint_names(args.hand)
  qadr = np.array([model.joint(n).qposadr[0] for n in jnames])
  target_qadr = np.array(
    [model.joint(f"target/{n}").qposadr[0] for n in jnames]
  )
  _cube_bid = model.body("obs_cube").id
  cube_qadr = model.jnt_qposadr[
    model.body_jntadr[_cube_bid]
  ]
  goal_mid = model.body("goal_cube").mocapid[0]

  tag_pos = const.root_pos + quat_apply(root_quat, const.tag_in_palm_pos)
  tag_quat = quat_mul(root_quat, const.tag_in_palm_quat)

  cube_port = args.cube_port if args.cube_port is not None else config_loader.cube_port()
  goal_port = config_loader.goal_port()
  joint_port = config_loader.joint_port()
  cube_recv = CubeReceiver(port=cube_port)
  goal_recv = GoalReceiver()
  joint_recv = JointReceiver()

  data.qpos[qadr] = const.default_joint_pos
  data.qpos[target_qadr] = const.default_joint_pos
  data.qpos[cube_qadr : cube_qadr + 7] = [*tag_pos, 1, 0, 0, 0]
  mujoco.mj_forward(model, data)

  print(
    f"[render_viewer] gen={args.gen} hand={args.hand}; subscribing "
    f"{cube_port}/{goal_port}/{joint_port}. Ctrl+C to quit."
  )
  with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [0.0, 0.0, 0.28]
    viewer.cam.distance = 1.15
    viewer.cam.azimuth = 140.0
    viewer.cam.elevation = -12.0
    last_report = 0.0
    while viewer.is_running():
      q_target = joint_recv.latest()
      q_measured = joint_recv.latest_measured()
      joint_status = "NO"
      if q_target is not None and q_target.shape[0] == const.num_joints:
        data.qpos[target_qadr] = q_target
        if q_measured is not None and q_measured.shape[0] == const.num_joints:
          data.qpos[qadr] = q_measured
          joint_status = "measured+target-ghost"
        else:
          data.qpos[qadr] = q_target
          joint_status = "target-only"
      cube_pos_tag, cube_quat_tag = cube_recv.latest()
      cube_pos_w = tag_pos + quat_apply(tag_quat, cube_pos_tag)
      cube_quat_w = quat_mul(tag_quat, cube_quat_tag)
      data.qpos[cube_qadr : cube_qadr + 3] = cube_pos_w
      data.qpos[cube_qadr + 3 : cube_qadr + 7] = cube_quat_w
      data.mocap_quat[goal_mid] = quat_mul(tag_quat, goal_recv.latest())
      mujoco.mj_forward(model, data)
      viewer.sync()
      now = time.monotonic()
      if now - last_report > 2.0:
        print(
          f"[render_viewer] cube msgs={cube_recv.count} "
          f"joints={joint_status}  "
          f"cube_pos_tag={np.round(cube_pos_tag, 3)}",
          flush=True,
        )
        last_report = now
      time.sleep(1.0 / 60.0)


if __name__ == "__main__":
  main()
