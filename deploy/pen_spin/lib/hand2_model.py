# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Torch-free hand2 name adapter, matching main's shared robot loader.

Keep the public MJCF and its body hierarchy intact. This deploy-local copy
avoids importing mjlab/torch into the pen-spin deployment environment; parity
against the shared loader is checked by the hand2 compatibility tests.
"""
from pathlib import Path

import mujoco

_RENAMED = ("body", "joint", "site", "actuator", "mesh")
_UNCOVERED = (
  "sensors", "tendons", "equalities", "pairs", "cameras", "lights", "materials",
  "textures", "hfields", "skins", "flexes", "keys", "numerics", "texts", "tuples",
  "frames", "plugins",
)

WUJI_HAND2_TASK_NAMES: dict[str, dict[str, str]] = {
  "model": {
    "wujihand2-right": "wujihand2-right-mjlab"
  },
  "body": {
    "r_wrist": "right_palm_link",
    "r_thumb_proximal": "right_finger1_link1",
    "r_thumb_proximal_abd": "right_finger1_link2",
    "r_thumb_middle": "right_finger1_link3",
    "r_thumb_distal": "right_finger1_link4",
    "r_thumb_tip_sensor_frame": "right_finger1_tip_sensor",
    "r_index_finger_proximal": "right_finger2_link1",
    "r_index_finger_proximal_abd": "right_finger2_link2",
    "r_index_finger_middle": "right_finger2_link3",
    "r_index_finger_distal": "right_finger2_link4",
    "r_index_finger_tip_sensor_frame": "right_finger2_tip_sensor",
    "r_middle_finger_proximal": "right_finger3_link1",
    "r_middle_finger_proximal_abd": "right_finger3_link2",
    "r_middle_finger_middle": "right_finger3_link3",
    "r_middle_finger_distal": "right_finger3_link4",
    "r_middle_finger_tip_sensor_frame": "right_finger3_tip_sensor",
    "r_ring_finger_proximal": "right_finger4_link1",
    "r_ring_finger_proximal_abd": "right_finger4_link2",
    "r_ring_finger_middle": "right_finger4_link3",
    "r_ring_finger_distal": "right_finger4_link4",
    "r_ring_finger_tip_sensor_frame": "right_finger4_tip_sensor",
    "r_pinky_proximal": "right_finger5_link1",
    "r_pinky_proximal_abd": "right_finger5_link2",
    "r_pinky_middle": "right_finger5_link3",
    "r_pinky_distal": "right_finger5_link4",
    "r_pinky_tip_sensor_frame": "right_finger5_tip_sensor"
  },
  "joint": {
    "r_thumb_cmc_flex": "right_finger1_joint1",
    "r_thumb_cmc_abd": "right_finger1_joint2",
    "r_thumb_mcp": "right_finger1_joint3",
    "r_thumb_ip": "right_finger1_joint4",
    "r_index_finger_mcp_flex": "right_finger2_joint1",
    "r_index_finger_mcp_abd": "right_finger2_joint2",
    "r_index_finger_pip": "right_finger2_joint3",
    "r_index_finger_dip": "right_finger2_joint4",
    "r_middle_finger_mcp_flex": "right_finger3_joint1",
    "r_middle_finger_mcp_abd": "right_finger3_joint2",
    "r_middle_finger_pip": "right_finger3_joint3",
    "r_middle_finger_dip": "right_finger3_joint4",
    "r_ring_finger_mcp_flex": "right_finger4_joint1",
    "r_ring_finger_mcp_abd": "right_finger4_joint2",
    "r_ring_finger_pip": "right_finger4_joint3",
    "r_ring_finger_dip": "right_finger4_joint4",
    "r_pinky_mcp_flex": "right_finger5_joint1",
    "r_pinky_mcp_abd": "right_finger5_joint2",
    "r_pinky_pip": "right_finger5_joint3",
    "r_pinky_dip": "right_finger5_joint4"
  },
  "site": {
    "r_thumb_tip": "right_finger1_tip",
    "r_index_finger_tip": "right_finger2_tip",
    "r_middle_finger_tip": "right_finger3_tip",
    "r_ring_finger_tip": "right_finger4_tip",
    "r_pinky_tip": "right_finger5_tip"
  },
  "actuator": {
    "r_THJ0": "right_finger1_joint1_actuator",
    "r_THJ1": "right_finger1_joint2_actuator",
    "r_THJ2": "right_finger1_joint3_actuator",
    "r_THJ3": "right_finger1_joint4_actuator",
    "r_FFJ0": "right_finger2_joint1_actuator",
    "r_FFJ1": "right_finger2_joint2_actuator",
    "r_FFJ2": "right_finger2_joint3_actuator",
    "r_FFJ3": "right_finger2_joint4_actuator",
    "r_MFJ0": "right_finger3_joint1_actuator",
    "r_MFJ1": "right_finger3_joint2_actuator",
    "r_MFJ2": "right_finger3_joint3_actuator",
    "r_MFJ3": "right_finger3_joint4_actuator",
    "r_RFJ0": "right_finger4_joint1_actuator",
    "r_RFJ1": "right_finger4_joint2_actuator",
    "r_RFJ2": "right_finger4_joint3_actuator",
    "r_RFJ3": "right_finger4_joint4_actuator",
    "r_LFJ0": "right_finger5_joint1_actuator",
    "r_LFJ1": "right_finger5_joint2_actuator",
    "r_LFJ2": "right_finger5_joint3_actuator",
    "r_LFJ3": "right_finger5_joint4_actuator"
  },
  "mesh": {
    "r_wrist": "r_wrist",
    "r_thumb_proximal": "r_thumb_proximal",
    "r_thumb_proximal_abd": "r_thumb_proximal_abd",
    "r_thumb_middle": "r_thumb_middle",
    "r_thumb_distal": "r_thumb_distal",
    "r_thumb_tip_sensor_frame": "r_thumb_tip_sensor_frame",
    "r_index_finger_proximal": "r_index_finger_proximal",
    "r_index_finger_proximal_abd": "r_index_finger_proximal_abd",
    "r_index_finger_middle": "r_index_finger_middle",
    "r_index_finger_distal": "r_index_finger_distal",
    "r_index_finger_tip_sensor_frame": "r_index_finger_tip_sensor_frame",
    "r_middle_finger_proximal": "r_middle_finger_proximal",
    "r_middle_finger_proximal_abd": "r_middle_finger_proximal_abd",
    "r_middle_finger_middle": "r_middle_finger_middle",
    "r_middle_finger_distal": "r_middle_finger_distal",
    "r_middle_finger_tip_sensor_frame": "r_middle_finger_tip_sensor_frame",
    "r_ring_finger_proximal": "r_ring_finger_proximal",
    "r_ring_finger_proximal_abd": "r_ring_finger_proximal_abd",
    "r_ring_finger_middle": "r_ring_finger_middle",
    "r_ring_finger_distal": "r_ring_finger_distal",
    "r_ring_finger_tip_sensor_frame": "r_ring_finger_tip_sensor_frame",
    "r_pinky_proximal": "r_pinky_proximal",
    "r_pinky_proximal_abd": "r_pinky_proximal_abd",
    "r_pinky_middle": "r_pinky_middle",
    "r_pinky_distal": "r_pinky_distal",
    "r_pinky_tip_sensor_frame": "r_pinky_tip_sensor_frame"
  },
  "collision_geom": {
    "r_wrist": "right_palm_collision",
    "r_thumb_proximal": "right_finger1_link1_col",
    "r_thumb_proximal_abd": "right_finger1_link2_col",
    "r_thumb_middle": "right_finger1_link3_col",
    "r_thumb_distal": "right_finger1_link4_col",
    "r_thumb_tip_sensor_frame": "right_finger1_tip_sensor_col",
    "r_index_finger_proximal": "right_finger2_link1_col",
    "r_index_finger_proximal_abd": "right_finger2_link2_col",
    "r_index_finger_middle": "right_finger2_link3_col",
    "r_index_finger_distal": "right_finger2_link4_col",
    "r_index_finger_tip_sensor_frame": "right_finger2_tip_sensor_col",
    "r_middle_finger_proximal": "right_finger3_link1_col",
    "r_middle_finger_proximal_abd": "right_finger3_link2_col",
    "r_middle_finger_middle": "right_finger3_link3_col",
    "r_middle_finger_distal": "right_finger3_link4_col",
    "r_middle_finger_tip_sensor_frame": "right_finger3_tip_sensor_col",
    "r_ring_finger_proximal": "right_finger4_link1_col",
    "r_ring_finger_proximal_abd": "right_finger4_link2_col",
    "r_ring_finger_middle": "right_finger4_link3_col",
    "r_ring_finger_distal": "right_finger4_link4_col",
    "r_ring_finger_tip_sensor_frame": "right_finger4_tip_sensor_col",
    "r_pinky_proximal": "right_finger5_link1_col",
    "r_pinky_proximal_abd": "right_finger5_link2_col",
    "r_pinky_middle": "right_finger5_link3_col",
    "r_pinky_distal": "right_finger5_link4_col",
    "r_pinky_tip_sensor_frame": "right_finger5_tip_sensor_col"
  }
}



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


def load_hand_spec(xml_path: Path) -> mujoco.MjSpec:
  return _apply_task_names(mujoco.MjSpec.from_file(str(xml_path)), WUJI_HAND2_TASK_NAMES)
