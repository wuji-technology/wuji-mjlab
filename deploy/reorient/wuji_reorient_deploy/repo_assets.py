# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Locate public robot and object assets for deployment."""

from pathlib import Path

import mujoco
from wuji_reorient_deploy import constants


REPO_ASSETS_DIR = Path(__file__).resolve().parents[3] / "src" / "wuji_mjlab" / "assets"

_RENAMED = ("body", "joint", "site", "actuator", "mesh")
_UNCOVERED = (
  "sensors", "tendons", "equalities", "pairs", "cameras", "lights", "materials",
  "textures", "hfields", "skins", "flexes", "keys", "numerics", "texts", "tuples",
  "frames", "plugins",
)


def hand_mjcf_path(gen: int, hand_side: str) -> Path:
  assets_dir = REPO_ASSETS_DIR / "robots" / ("wuji_hand" if gen == 1 else "wuji_hand2") / "mjcf"
  if not assets_dir.is_dir():
    raise FileNotFoundError(
      f"hand MJCF dir not found: {assets_dir}. Expected it under "
      "src/wuji_mjlab/assets/robots/<hand>/mjcf/ in the repo checkout."
    )
  xml_path = assets_dir / f"{hand_side}_mjlab.xml"
  if not xml_path.exists():
    raise FileNotFoundError(
      f"viewer MJCF not found: {xml_path}. Expected it under "
      "src/wuji_mjlab/assets/robots/<hand>/mjcf/<side>_mjlab.xml in the "
      "repo checkout."
    )
  return xml_path


def _require_bijection(kind: str, present: list[str], table: dict[str, str]) -> None:
  targets = list(table.values())
  if sorted(present) != sorted(table) or len(set(targets)) != len(targets):
    raise ValueError(
      f"Wuji Hand 2 {kind} names disagree with the task-name table: "
      f"unmapped={sorted(set(present) - set(table))} absent={sorted(set(table) - set(present))} "
      f"duplicate_targets={sorted({t for t in targets if targets.count(t) > 1})}"
    )


def _apply_task_names(spec: mujoco.MjSpec, names: dict[str, dict[str, str]]) -> mujoco.MjSpec:
  uncovered = [kind for kind in _UNCOVERED if getattr(spec, kind)]
  if uncovered:
    raise ValueError(f"Wuji Hand 2 MJCF has {uncovered}; the task-name table does not cover them")
  bodies = [body for body in spec.bodies if body.name != "world"]
  elements = {
    "body": bodies,
    "joint": list(spec.joints),
    "site": list(spec.sites),
    "actuator": list(spec.actuators),
    "mesh": list(spec.meshes),
  }
  _require_bijection("model", [spec.modelname], names["model"])
  for kind in _RENAMED:
    _require_bijection(kind, [element.name for element in elements[kind]], names[kind])
  _require_bijection("collision_geom", [body.name for body in bodies], names["collision_geom"])
  if any(geom.name for geom in spec.geoms):
    raise ValueError("Wuji Hand 2 upstream geoms are expected to be unnamed")
  collision_geoms = []
  for body in bodies:
    collidable = [geom for geom in body.geoms if geom.contype or geom.conaffinity]
    if len(collidable) != 1:
      raise ValueError(f"Wuji Hand 2 body {body.name!r} has {len(collidable)} collidable geoms; expected 1")
    collision_geoms.append((collidable[0], names["collision_geom"][body.name]))
  for actuator in spec.actuators:
    if actuator.trntype != mujoco.mjtTrn.mjTRN_JOINT or actuator.refsite or actuator.slidersite:
      raise ValueError(f"Wuji Hand 2 actuator {actuator.name!r} is not a plain joint actuator")
    actuator.target = names["joint"][actuator.target]
  for exclude in spec.excludes:
    exclude.bodyname1 = names["body"][exclude.bodyname1]
    exclude.bodyname2 = names["body"][exclude.bodyname2]
  for geom in spec.geoms:
    if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
      geom.meshname = names["mesh"][geom.meshname]
  spec.modelname = names["model"][spec.modelname]
  for kind in _RENAMED:
    for element in elements[kind]:
      element.name = names[kind][element.name]
  for geom, name in collision_geoms:
    geom.name = name
  joints = {joint.name for joint in spec.joints}
  bodies_after = {body.name for body in spec.bodies}
  meshes = {mesh.name for mesh in spec.meshes}
  live = (
    sum(actuator.target in joints for actuator in spec.actuators),
    sum(e.bodyname1 in bodies_after and e.bodyname2 in bodies_after for e in spec.excludes),
    sum(g.meshname in meshes for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH),
  )
  expected = (
    len(spec.actuators),
    len(spec.excludes),
    sum(g.type == mujoco.mjtGeom.mjGEOM_MESH for g in spec.geoms),
  )
  if live != expected:
    raise ValueError(f"Wuji Hand 2 references left dangling after renaming: live={live} expected={expected}")
  return spec


def load_hand_spec(gen: int, hand_side: str) -> mujoco.MjSpec:
  xml_path = hand_mjcf_path(gen, hand_side)
  spec = mujoco.MjSpec.from_file(str(xml_path))
  if gen == 2:
    _apply_task_names(spec, constants.WUJI_HAND2_TASK_NAMES)
  return spec
