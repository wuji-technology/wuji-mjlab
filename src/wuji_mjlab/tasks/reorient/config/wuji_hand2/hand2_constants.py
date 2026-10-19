# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from collections.abc import Callable

import mujoco
import numpy as np
from mjlab.entity import EntityCfg

from wuji_mjlab.assets.robots.wuji_hand2.wuji_hand2_cfg import WUJI_HAND2_JIG_TABLE_MESH

REORIENT_HAND2_PALM_NORMAL_AXIS = 1


REORIENT_HAND2_ROOT_POS = (-0.0284999, 0.00300004, 0.500250158)
REORIENT_HAND2_ROOT_ROT = (0.5, 0.5, 0.5, 0.5)

# Calibrated mount frame, in the palm frame.
REORIENT_HAND2_MOUNT_POS = (-0.0030000041, -0.000250110803, 0.0285)
REORIENT_HAND2_MOUNT_ROT = (1.0, 0.0, 0.0, -0.0000081995)
REORIENT_HAND2_TAG_POS = (0.0, 0.037408, 0.034737)
REORIENT_HAND2_TAG_ROT = (0.5, -0.5, -0.5, -0.5)
REORIENT_HAND2_JIG_POS = (0.0, -0.1062088, 0.0178309)
REORIENT_HAND2_JIG_ROT = (0.541668573, -0.454527399, 0.454527399, 0.541668573)

HAND2_REORIENT_GEOM_DEFAULTS = {
  "group": 3,
  "friction": (0.7, 0.005, 0.0001),
  "solref": (0.02, 1.5),
  "solimp": (0.9, 0.95, 0.001, 0.5, 2.0),
  "priority": 1,
}

HAND2_REORIENT_ACTUATOR_GAINS = {
  "right_finger1_joint1": (1.354985, 0.04769),
  "right_finger1_joint2": (2.198821, 0.062838),
  "right_finger1_joint3": (0.886467, 0.027982),
  "right_finger1_joint4": (0.654026, 0.020457),
  "right_finger2_joint1": (1.345207, 0.047402),
  "right_finger2_joint2": (0.88946, 0.027239),
  "right_finger2_joint3": (0.779361, 0.024489),
  "right_finger2_joint4": (0.656424, 0.020508),
  "right_finger3_joint1": (1.299313, 0.045699),
  "right_finger3_joint2": (0.942641, 0.027846),
  "right_finger3_joint3": (0.73725, 0.023408),
  "right_finger3_joint4": (0.710314, 0.022366),
  "right_finger4_joint1": (1.276892, 0.046209),
  "right_finger4_joint2": (1.013334, 0.028704),
  "right_finger4_joint3": (0.859159, 0.027436),
  "right_finger4_joint4": (0.648064, 0.020448),
  "right_finger5_joint1": (1.102915, 0.043816),
  "right_finger5_joint2": (0.910201, 0.026593),
  "right_finger5_joint3": (0.861968, 0.027476),
  "right_finger5_joint4": (0.656704, 0.020611),
}

HAND2_REORIENT_ACTUATOR_FORCE_LIMITS = {
  "right_finger1_joint1": 1.02,
  "right_finger1_joint2": 0.83,
  "right_finger1_joint3": 0.35,
  "right_finger1_joint4": 0.21,
  "right_finger2_joint1": 1.05,
  "right_finger2_joint2": 0.38,
  "right_finger2_joint3": 0.31,
  "right_finger2_joint4": 0.30,
  "right_finger3_joint1": 1.06,
  "right_finger3_joint2": 0.39,
  "right_finger3_joint3": 0.32,
  "right_finger3_joint4": 0.29,
  "right_finger4_joint1": 1.02,
  "right_finger4_joint2": 0.41,
  "right_finger4_joint3": 0.28,
  "right_finger4_joint4": 0.30,
  "right_finger5_joint1": 0.87,
  "right_finger5_joint2": 0.36,
  "right_finger5_joint3": 0.28,
  "right_finger5_joint4": 0.30,
}

HAND2_REORIENT_CAPSULES = {
  "right_finger1_link1": (
    (
      0.0001600994,
      -0.0025818024,
      -0.001025414,
      0.0010229472,
      0.00022010189,
      -0.014007877,
    ),
    0.008326602,
  ),
  "right_finger1_link2": (
    (
      1.549657e-06,
      -3.15346e-06,
      -0.03143319,
      1.556016e-06,
      -3.148208e-06,
      -0.004805587,
    ),
    0.00874906,
  ),
  "right_finger1_link3": (
    (
      -5.1675061e-05,
      -9.0148385e-07,
      -0.034401508,
      -5.1675061e-05,
      -9.0148385e-07,
      -0.00053389498,
    ),
    0.008768497,
  ),
  "right_finger2_link1": (
    (
      2.0003946e-20,
      -0.0012000001,
      -0.00087331082,
      -7.5588199e-19,
      -0.0012000001,
      -0.01354449,
    ),
    0.009388893,
  ),
  "right_finger2_link2": (
    (
      1.43177e-05,
      1.68479e-07,
      -0.003500009,
      -3.205759e-06,
      1.684791e-07,
      -0.03710302,
    ),
    0.007507256,
  ),
  "right_finger2_link3": (
    (
      -1.805026e-18,
      -1.1052597e-34,
      -0.029478312,
      2.1212593e-21,
      1.2988967e-37,
      3.4642794e-05,
    ),
    0.00801817,
  ),
  "right_finger3_link1": (
    (
      2.0003946e-20,
      -0.0012000001,
      -0.00087331082,
      -7.5588199e-19,
      -0.0012000001,
      -0.01354449,
    ),
    0.009388893,
  ),
  "right_finger3_link2": (
    (
      1.43177e-05,
      1.68479e-07,
      -0.003500009,
      -3.205759e-06,
      1.684791e-07,
      -0.03710302,
    ),
    0.007507256,
  ),
  "right_finger3_link3": (
    (
      -1.805026e-18,
      -1.1052597e-34,
      -0.029478312,
      2.1212593e-21,
      1.2988967e-37,
      3.4642794e-05,
    ),
    0.00801817,
  ),
  "right_finger4_link1": (
    (
      2.0003946e-20,
      -0.0012000001,
      -0.00087331082,
      -7.5588199e-19,
      -0.0012000001,
      -0.01354449,
    ),
    0.009388893,
  ),
  "right_finger4_link2": (
    (
      1.43177e-05,
      1.68479e-07,
      -0.003500009,
      -3.205759e-06,
      1.684791e-07,
      -0.03710302,
    ),
    0.007507256,
  ),
  "right_finger4_link3": (
    (
      -1.805026e-18,
      -1.1052597e-34,
      -0.029478312,
      2.1212593e-21,
      1.2988967e-37,
      3.4642794e-05,
    ),
    0.00801817,
  ),
  "right_finger5_link1": (
    (
      2.0003975e-20,
      -0.0012000005,
      -0.00087331082,
      -7.5588196e-19,
      -0.0012000005,
      -0.01354449,
    ),
    0.009407771,
  ),
  "right_finger5_link2": (
    (
      -2.479326e-08,
      9.787299e-07,
      -0.03333651,
      -2.479326e-08,
      9.787299e-07,
      -0.003263492,
    ),
    0.007500022,
  ),
  "right_finger5_link3": (
    (
      -1.6782627e-18,
      -1.0276395e-34,
      -0.02740811,
      -2.177578e-21,
      -1.333382e-37,
      -3.5562547e-05,
    ),
    0.008049908,
  ),
}


def apply_hand2_reorient_actuator_gains(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Apply the Wuji Hand 2 gains measured for the reorient task."""
  actuators = {actuator.name: actuator for actuator in spec.actuators}
  for joint_name, (kp, kv) in HAND2_REORIENT_ACTUATOR_GAINS.items():
    actuator_name = f"{joint_name}_actuator"
    try:
      actuator = actuators[actuator_name]
    except KeyError as exc:
      raise ValueError(f"missing Wuji Hand 2 actuator {actuator_name!r}") from exc
    actuator.set_to_position(kp=kp, kv=kv)
  return spec


def apply_hand2_reorient_force_limits(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Joint actuator force limits must match actuator limits or they can clip the requested torque before the actuator limit is reached."""
  actuators = {actuator.name: actuator for actuator in spec.actuators}
  for joint_name, limit in HAND2_REORIENT_ACTUATOR_FORCE_LIMITS.items():
    actuator_name = f"{joint_name}_actuator"
    try:
      actuator = actuators[actuator_name]
    except KeyError as exc:
      raise ValueError(f"missing Wuji Hand 2 actuator {actuator_name!r}") from exc
    actuator.forcerange = [-limit, limit]
    actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    joint = spec.joint(joint_name)
    joint.actfrcrange = [-limit, limit]
    joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
  return spec


def apply_hand2_reorient_collision_overlay(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  collision_meshes = {geom.name: geom for geom in spec.geoms}
  for body_name, (fromto, radius) in HAND2_REORIENT_CAPSULES.items():
    geom_name = f"{body_name}_col"
    try:
      collision_mesh = collision_meshes[geom_name]
    except KeyError as exc:
      raise ValueError(f"missing hand2 collision mesh {geom_name!r}") from exc
    collision_mesh.name = f"{geom_name}_mesh_disabled"
    collision_mesh.contype = 0
    collision_mesh.conaffinity = 0

    link_number = int(body_name[-1])
    enabled = link_number != 1
    spec.body(body_name).add_geom(
      name=geom_name,
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      fromto=fromto,
      size=(radius, 0.0, 0.0),
      contype=int(enabled),
      conaffinity=int(enabled),
      group=3,
      density=0,
    )

  while spec.excludes:
    spec.delete(spec.excludes[0])
  spec.add_exclude(
    bodyname1="right_finger1_link2",
    bodyname2="right_palm_link",
  )
  mount = spec.geom("right_mount_collision")
  mount.contype = 0
  mount.conaffinity = 0
  return spec


def add_frame_axes(
  body,
  *,
  half_length: float = 0.03,
  half_width: float = 0.0015,
  origin_radius: float = 0.0035,
) -> None:
  """Add visual-only XYZ axes to a body frame."""
  body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(origin_radius, 0.0, 0.0),
    rgba=(1.0, 1.0, 0.0, 1.0),
    contype=0,
    conaffinity=0,
    group=1,
    density=0,
  )
  for axis, rgba in (
    (0, (1.0, 0.0, 0.0, 1.0)),
    (1, (0.0, 1.0, 0.0, 1.0)),
    (2, (0.0, 0.0, 1.0, 1.0)),
  ):
    size = [half_width, half_width, half_width]
    size[axis] = half_length
    pos = [0.0, 0.0, 0.0]
    pos[axis] = half_length
    body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=size,
      pos=pos,
      rgba=rgba,
      contype=0,
      conaffinity=0,
      group=1,
      density=0,
    )


def _in_mount(mount: mujoco.MjsBody, pos, quat) -> tuple[np.ndarray, np.ndarray]:
  """Express a pose given in the calibrated mount frame in the mount body."""
  palm_to_body = np.zeros(3), np.zeros(4)
  mujoco.mju_negPose(*palm_to_body, mount.pos, mount.quat / np.linalg.norm(mount.quat))
  calibrated = np.zeros(3), np.zeros(4)
  mujoco.mju_mulPose(
    *calibrated,
    np.asarray(REORIENT_HAND2_MOUNT_POS, dtype=np.float64),
    np.asarray(REORIENT_HAND2_MOUNT_ROT, dtype=np.float64),
    np.asarray(pos, dtype=np.float64),
    np.asarray(quat, dtype=np.float64),
  )
  out = np.zeros(3), np.zeros(4)
  mujoco.mju_mulPose(*out, *palm_to_body, *calibrated)
  return out


def inject_hand2_reorient_rig(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Attach the calibration jig and wrist tag to the hand2 mount."""
  filename = WUJI_HAND2_JIG_TABLE_MESH.name
  spec.add_mesh(name="jig_table", content_type="model/stl", file=filename)
  asset_key = f"{spec.meshdir}/{filename}" if spec.meshdir else filename
  spec.assets = {**spec.assets, asset_key: WUJI_HAND2_JIG_TABLE_MESH.read_bytes()}

  mount = spec.body("right_mount")
  jig_pos, jig_quat = _in_mount(mount, REORIENT_HAND2_JIG_POS, REORIENT_HAND2_JIG_ROT)
  mount.add_geom(
    name="jig_table_visual",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="jig_table",
    pos=jig_pos,
    quat=jig_quat,
    contype=0,
    conaffinity=0,
    group=1,
    density=0,
    rgba=(0.35, 0.55, 0.9, 0.55),
  )

  tag_pos, tag_quat = _in_mount(mount, REORIENT_HAND2_TAG_POS, REORIENT_HAND2_TAG_ROT)
  tag = mount.add_body(name="right_wrist_tag_frame", pos=tag_pos, quat=tag_quat)
  tag.add_site(
    name="right_wrist_tag",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.0252, 0.0252, 0.0006),
    rgba=(1.0, 0.9, 0.0, 0.5),
    group=1,
  )
  add_frame_axes(tag)
  return spec


def apply_hand2_reorient_physics(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  defaults = spec.default.geom
  for geom in spec.geoms:
    if geom.group == defaults.group:
      geom.group = HAND2_REORIENT_GEOM_DEFAULTS["group"]
    if (geom.friction == defaults.friction).all():
      geom.friction = HAND2_REORIENT_GEOM_DEFAULTS["friction"]
    if (geom.solref == defaults.solref).all():
      geom.solref = HAND2_REORIENT_GEOM_DEFAULTS["solref"]
    if (geom.solimp == defaults.solimp).all():
      geom.solimp = HAND2_REORIENT_GEOM_DEFAULTS["solimp"]
    if geom.priority == defaults.priority:
      geom.priority = HAND2_REORIENT_GEOM_DEFAULTS["priority"]
  for finger in range(1, 6):
    spec.site(f"right_finger{finger}_tip").group = 4
  spec.option.timestep = 0.01
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
  spec.option.iterations = 5
  spec.option.ls_iterations = 8
  spec.option.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
  spec.option.tolerance = 1e-8
  spec.option.disableflags = int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
  return spec


def get_hand2_reorient_spec(
  base_spec_fn: Callable[[], mujoco.MjSpec],
) -> mujoco.MjSpec:
  """Build the robot spec with the reorient calibration rig attached."""
  spec = inject_hand2_reorient_rig(base_spec_fn())
  spec = apply_hand2_reorient_collision_overlay(spec)
  spec = apply_hand2_reorient_physics(spec)
  spec = apply_hand2_reorient_actuator_gains(spec)
  return apply_hand2_reorient_force_limits(spec)


# Cage joint pose (finger1=thumb ... finger5=pinky; joint2 = abduction).
REORIENT_HAND2_JOINT_POS: dict[str, float] = {
  ".*_finger1_joint1": 0.8,
  ".*_finger1_joint2": -0.3,
  ".*_finger1_joint3": 0.5,
  ".*_finger1_joint4": 0.4,
  ".*_finger2_joint1": 0.45,
  ".*_finger2_joint2": -0.15,
  ".*_finger2_joint3": 0.8,
  ".*_finger2_joint4": 0.5,
  ".*_finger3_joint1": 0.45,
  ".*_finger3_joint2": 0.0,
  ".*_finger3_joint3": 0.9,
  ".*_finger3_joint4": 0.3,
  ".*_finger4_joint1": 0.6,
  ".*_finger4_joint2": 0.15,
  ".*_finger4_joint3": 1.1,
  ".*_finger4_joint4": 0.2,
  ".*_finger5_joint1": 1.2,
  ".*_finger5_joint2": 0.2,
  ".*_finger5_joint3": 0.8,
  ".*_finger5_joint4": 0.4,
}

REORIENT_HAND2_ROBOT_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=REORIENT_HAND2_ROOT_POS,
  rot=REORIENT_HAND2_ROOT_ROT,
  joint_pos=REORIENT_HAND2_JOINT_POS,
  joint_vel={".*": 0.0},
)

REORIENT_HAND2_CUBE_INIT_POS = {"right": (-0.108, 0.0072, 0.5616)}
REORIENT_HAND2_CUBE_INIT_ROT = (1.0, 0.0, 0.0, 0.0)


def hand2_cube_init_state(hand_side: str) -> EntityCfg.InitialStateCfg:
  if hand_side != "right":
    raise ValueError(f"Unsupported hand_side '{hand_side}'. Only 'right' is supported.")
  return EntityCfg.InitialStateCfg(
    pos=REORIENT_HAND2_CUBE_INIT_POS[hand_side],
    rot=REORIENT_HAND2_CUBE_INIT_ROT,
  )
