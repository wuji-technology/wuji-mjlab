# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Locate public robot and object assets for deployment."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from wuji_reorient_deploy import constants


_REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_ASSETS_DIR = _REPO_ROOT / "src" / "wuji_mjlab" / "assets"
HAND1_DESCRIPTION_DIR = REPO_ASSETS_DIR / "robots" / "wuji-description" / "hand" / "body-with-soft"
HAND2_DESCRIPTION_DIR = REPO_ASSETS_DIR / "robots" / "wuji-description" / "hand2" / "hand2_2" / "body"

_RENAMED = ("body", "joint", "site", "actuator", "mesh", "material")
_UNCOVERED = (
  "sensors", "tendons", "equalities", "pairs", "cameras", "lights", "textures",
  "hfields", "skins", "flexes", "keys", "numerics", "texts", "tuples", "frames",
  "plugins",
)
_DROPPED_SENSORS = ("r_wrist_imu_acc", "r_wrist_imu_gyro")


def hand_mjcf_path(gen: int, hand_side: str) -> Path:
  if gen == 1:
    xml_path = HAND1_DESCRIPTION_DIR / "mjcf" / f"{hand_side}_simplified.xml"
  else:
    xml_path = HAND2_DESCRIPTION_DIR / "mjcf" / f"{hand_side}.xml"
  if not xml_path.exists():
    raise FileNotFoundError(
      f"Wuji Hand {gen} MJCF not found: {xml_path}. Fetch the wuji-description "
      "submodule with `git submodule update --init`."
    )
  return xml_path


# Wuji Hand 1: the training loader's task edits, applied the same way.
_HAND1_COLLISION_GEOMS = (
  "right_palm_collision", "right_finger1_link2_col", "right_finger1_link2_softbody_col",
  "right_finger1_link3_col", "right_finger1_link4_col",
  *(f"right_finger{f}_link{link}_col" for f in range(2, 6) for link in (2, 3, 4)),
)
_HAND1_TIP_SITES = {
  "right_finger1_link4": (0.00026399, -0.00020119, 0.032808),
  "right_finger2_link4": (-0.000749345, 0.0, 0.024),
  "right_finger3_link4": (-0.00075, 0.0, 0.024),
  "right_finger4_link4": (-0.000811164, 0.0, 0.0239995),
  "right_finger5_link4": (-0.000828016, 0.0, 0.0259994),
}
_HAND1_PALM_EXCLUDES = (
  "right_finger1_link1", "right_finger1_link2", "right_finger2_link2",
  "right_finger3_link2", "right_finger4_link2", "right_finger5_link2",
)


def _apply_hand1_task_edits(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  if spec.modelname != "wujihand-right" or spec.sites or spec.excludes:
    raise ValueError(f"Wuji Hand 1 MJCF {spec.modelname!r} is not the upstream hand the task edits expect")
  spec.modelname = "wujihand-right-mjlab"
  builtin = mujoco.MjSpec()
  spec.option.jacobian = builtin.option.jacobian
  spec.option.tolerance = builtin.option.tolerance
  spec.option.timestep = 0.01
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
  spec.option.iterations = 5
  spec.option.ls_iterations = 8
  spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
  default = builtin.default.geom
  for geom in spec.geoms:
    if (
      list(geom.friction) != list(default.friction) or list(geom.solref) != list(default.solref)
      or list(geom.solimp) != list(default.solimp) or geom.priority != default.priority
    ):
      raise ValueError(f"Wuji Hand 1 geom on {geom.parent.name!r} sets its own contact")
    if geom.group == default.group:
      geom.group = 3
    geom.friction = (0.7, 0.005, 0.0001)
    geom.solref = (0.02, 1.5)
    geom.priority = 1
  for geom in spec.geoms:
    if not (geom.contype or geom.conaffinity):
      continue
    body = geom.parent.name
    if body == "right_palm_link":
      geom.name = "right_palm_collision"
    elif geom.meshname.endswith("_softbody_simplified"):
      geom.name = f"{body}_softbody_col"
    else:
      geom.name = f"{body}_col"
  named = sorted(geom.name for geom in spec.geoms if geom.name)
  if named != sorted(_HAND1_COLLISION_GEOMS):
    raise ValueError(f"Wuji Hand 1 collision geoms {named} differ from {sorted(_HAND1_COLLISION_GEOMS)}")
  for body, pos in _HAND1_TIP_SITES.items():
    spec.body(body).add_site(name=f"{body.removesuffix('_link4')}_tip", pos=pos, group=4)
  for link in _HAND1_PALM_EXCLUDES:
    spec.add_exclude(bodyname1="right_palm_link", bodyname2=link)
  return spec


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
    "material": list(spec.materials),
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
    if geom.material:
      geom.material = names["material"][geom.material]
  spec.modelname = names["model"][spec.modelname]
  for kind in _RENAMED:
    for element in elements[kind]:
      element.name = names[kind][element.name]
  for geom, name in collision_geoms:
    geom.name = name
  joints = {joint.name for joint in spec.joints}
  bodies_after = {body.name for body in spec.bodies}
  meshes = {mesh.name for mesh in spec.meshes}
  materials = {material.name for material in spec.materials}
  live = (
    sum(actuator.target in joints for actuator in spec.actuators),
    sum(e.bodyname1 in bodies_after and e.bodyname2 in bodies_after for e in spec.excludes),
    sum(g.meshname in meshes for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH),
    sum(g.material in materials for g in spec.geoms if g.material),
  )
  expected = (
    len(spec.actuators),
    len(spec.excludes),
    sum(g.type == mujoco.mjtGeom.mjGEOM_MESH for g in spec.geoms),
    sum(bool(g.material) for g in spec.geoms),
  )
  if live != expected:
    raise ValueError(f"Wuji Hand 2 references left dangling after renaming: live={live} expected={expected}")
  return spec


def _palm_rooted_xml(xml_path: Path) -> str:
  """Upstream roots Hand 2 at r_mount; root it at the palm, as training does."""
  root = ET.parse(xml_path).getroot()
  worldbody = root.find("worldbody")
  if worldbody is None or [(e.tag, e.get("name")) for e in worldbody] != [("body", "r_mount")]:
    raise ValueError(f"{xml_path}: expected the worldbody to hold only body 'r_mount'")
  mount = worldbody[0]
  palms = [e for e in mount if e.tag == "body"]
  if (
    [palm.get("name") for palm in palms] != ["r_wrist"]
    or set(mount.attrib) != {"name"}
    or any(e.tag not in ("inertial", "geom", "body") for e in mount)
    or not set(palms[0].attrib) <= {"name", "pos", "quat"}
  ):
    raise ValueError(f"{xml_path}: expected 'r_mount' at the origin holding only geoms and body 'r_wrist'")
  palm = palms[0]
  pos = np.array(palm.attrib.pop("pos", "0 0 0").split(), dtype=np.float64)
  quat = np.array(palm.attrib.pop("quat", "1 0 0 0").split(), dtype=np.float64)
  mount_pos, mount_quat = np.zeros(3), np.zeros(4)
  mujoco.mju_negPose(mount_pos, mount_quat, pos, quat / np.linalg.norm(quat))
  mount.remove(palm)
  worldbody.remove(mount)
  mount.set("pos", " ".join(repr(float(v)) for v in mount_pos))
  mount.set("quat", " ".join(repr(float(v)) for v in mount_quat))
  palm.append(mount)
  worldbody.append(palm)
  return ET.tostring(root, encoding="unicode")


def load_hand_spec(gen: int, hand_side: str) -> mujoco.MjSpec:
  xml_path = hand_mjcf_path(gen, hand_side)
  if gen == 1:
    return _apply_hand1_task_edits(mujoco.MjSpec.from_file(str(xml_path)))
  spec = mujoco.MjSpec.from_string(_palm_rooted_xml(xml_path))
  spec.modelfiledir = str(xml_path.parent)
  sensors = list(spec.sensors)
  if sorted(sensor.name for sensor in sensors) != sorted(_DROPPED_SENSORS):
    raise ValueError(f"Wuji Hand 2 sensors {sorted(s.name for s in sensors)} differ from {sorted(_DROPPED_SENSORS)}")
  for sensor in sensors:
    spec.delete(sensor)
  return _apply_task_names(spec, constants.WUJI_HAND2_TASK_NAMES)
