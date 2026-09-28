# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Standalone cage-pose validation for the reorient init state."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from functools import partial

import mujoco
import numpy as np

from wuji_mjlab.assets.robots.wuji_hand.wuji_hand_cfg import wuji_hand_xml_path
from wuji_mjlab.assets.robots.wuji_hand2.wuji_hand2_cfg import wuji_hand2_xml_path

CUBE_HALF = 0.027
CUBE_MASS = 0.120


def build_model(
  hand: str,
  side: str,
  root_pos: tuple[float, float, float],
  root_quat: tuple[float, float, float, float],
  cube_pos: tuple[float, float, float],
) -> mujoco.MjModel:
  xml_path = (
    wuji_hand2_xml_path(side) if hand == "wuji_hand2" else wuji_hand_xml_path(side)
  )
  if hand == "wuji_hand2":
    from wuji_mjlab.assets.robots.wuji_hand2.wuji_hand2_cfg import _get_spec
    from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
      get_hand2_reorient_spec,
    )

    spec = get_hand2_reorient_spec(partial(_get_spec, xml_path))
  else:
    spec = mujoco.MjSpec.from_file(str(xml_path))

  palm = spec.body(f"{side}_palm_link")
  palm.pos = root_pos
  palm.quat = root_quat

  cube = spec.worldbody.add_body(name="cube", pos=cube_pos)
  cube.add_freejoint(name="cube_free")
  # Mirrors assets/objects/inhand_object/xmls/cube.xml: conaffinity=2 reaches
  # the link2 geoms, priority 0 defers to the hand's contact params.
  cube.add_geom(
    name="cube",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[CUBE_HALF] * 3,
    mass=CUBE_MASS,
    friction=[0.3, 0.005, 0.0001],
    conaffinity=2,
    priority=0,
    rgba=[0.9, 0.3, 0.3, 1.0],
  )
  return spec.compile()


def set_hand_pose(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  side: str,
  joint_pos: dict[str, float],
) -> None:
  """Set hand joint qpos + actuator targets from a fingerN_jointM dict."""
  for n in range(1, 6):
    for m in range(1, 5):
      value = joint_pos[f"finger{n}_joint{m}"]
      jnt = model.joint(f"{side}_finger{n}_joint{m}")
      lo, hi = jnt.range
      value = float(np.clip(value, lo, hi))
      data.qpos[jnt.qposadr[0]] = value
      act_id = model.actuator(f"{side}_finger{n}_joint{m}_actuator").id
      data.ctrl[act_id] = value


def run_check(
  hand: str,
  side: str,
  root_pos: tuple[float, float, float],
  root_quat: tuple[float, float, float, float],
  cube_pos: tuple[float, float, float],
  joint_pos: dict[str, float],
  sim_seconds: float = 3.0,
  verbose: bool = True,
) -> dict:
  model = build_model(hand, side, root_pos, root_quat, cube_pos)
  data = mujoco.MjData(model)
  set_hand_pose(model, data, side, joint_pos)
  cube_qposadr = model.joint("cube_free").qposadr[0]
  data.qpos[cube_qposadr : cube_qposadr + 3] = cube_pos
  mujoco.mj_forward(model, data)

  tip_pos = np.array(
    [data.site(f"{side}_finger{n}_tip").xpos.copy() for n in range(1, 6)]
  )
  palm_z = data.body(f"{side}_palm_link").xpos[2]

  n_steps = int(sim_seconds / model.opt.timestep)
  for _ in range(n_steps):
    mujoco.mj_step(model, data)

  cube_final = data.qpos[cube_qposadr : cube_qposadr + 3].copy()
  drift = float(np.linalg.norm(cube_final - np.asarray(cube_pos)))
  dropped = bool(cube_final[2] < palm_z - 0.05)

  result = {
    "tip_positions": tip_pos,
    "tip_centroid": tip_pos.mean(axis=0),
    "cube_final": cube_final,
    "drift": drift,
    "dropped": dropped,
    "ok": (not dropped) and drift < 0.04,
  }
  if verbose:
    np.set_printoptions(precision=4, suppress=True)
    print(f"[{hand}/{side}] tip centroid (world): {result['tip_centroid']}")
    for i, tp in enumerate(tip_pos, start=1):
      print(f"  finger{i} tip: {tp}")
    print(f"  cube start: {np.asarray(cube_pos)}  final: {cube_final}")
    print(f"  drift: {drift * 1000:.1f} mm  dropped: {dropped}  ok: {result['ok']}")
  return result


_JOINT_NAMES = tuple(
  f"finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)
)


def get_task_joint_pos(hand: str) -> dict[str, float]:
  """Resolve the requested hand's task joint patterns to bare joint names."""
  if hand == "wuji_hand2":
    from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
      REORIENT_HAND2_JOINT_POS,
    )

    source = REORIENT_HAND2_JOINT_POS
  elif hand == "wuji_hand":
    from wuji_mjlab.tasks.reorient.reorient_constants import REORIENT_JOINT_POS

    source = REORIENT_JOINT_POS
  else:
    raise ValueError(f"unsupported hand {hand!r}")

  joint_pos: dict[str, float] = {}
  for pattern, value in source.items():
    for joint_name in _JOINT_NAMES:
      if re.fullmatch(pattern, f"right_{joint_name}"):
        joint_pos[joint_name] = value

  missing = [joint_name for joint_name in _JOINT_NAMES if joint_name not in joint_pos]
  if missing:
    raise ValueError(f"{hand}: joint pose patterns did not cover {missing}")
  return joint_pos


def get_task_root_pose(
  hand: str,
) -> tuple[
  tuple[float, float, float],
  tuple[float, float, float, float],
]:
  """Return the reorient task root pose for the requested hand generation."""
  if hand == "wuji_hand2":
    from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
      REORIENT_HAND2_ROOT_POS,
      REORIENT_HAND2_ROOT_ROT,
    )

    return REORIENT_HAND2_ROOT_POS, REORIENT_HAND2_ROOT_ROT
  if hand == "wuji_hand":
    from wuji_mjlab.tasks.reorient.reorient_constants import (
      REORIENT_ROBOT_ROOT_POS,
      REORIENT_ROBOT_ROOT_ROT,
    )

    return REORIENT_ROBOT_ROOT_POS, REORIENT_ROBOT_ROOT_ROT
  raise ValueError(f"unsupported hand {hand!r}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--hand", default="wuji_hand2", choices=("wuji_hand", "wuji_hand2")
  )
  parser.add_argument("--side", default="right", choices=("right",))
  parser.add_argument(
    "--cube-pos", type=float, nargs=3, default=None, help="world cube start pos"
  )
  parser.add_argument("--sim-seconds", type=float, default=3.0)
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  root_pos, root_quat = get_task_root_pose(args.hand)
  joint_pos = get_task_joint_pos(args.hand)

  cube_pos = args.cube_pos
  if cube_pos is None:
    probe = run_check(
      args.hand,
      args.side,
      root_pos,
      root_quat,
      (0.0, 0.0, 1.0),
      joint_pos,
      sim_seconds=0.0,
      verbose=False,
    )
    cube_pos = tuple(probe["tip_centroid"])
    print(f"placing cube at tip centroid: {np.asarray(cube_pos)}")

  run_check(
    args.hand,
    args.side,
    root_pos,
    root_quat,
    tuple(cube_pos),
    joint_pos,
    sim_seconds=args.sim_seconds,
  )


if __name__ == "__main__":
  main()
