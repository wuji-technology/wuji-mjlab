# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Identified Wuji Hand 2 actuator and joint parameters used by pen-spin.

These values are the identified calibration for the right Wuji Hand 2 hand.  They
are applied to the shared MJCF at load time so the wuji-description geometry
remains reusable by other tasks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import mujoco
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import GeomCfg

from wuji_mjlab.assets.robots.wuji_hand2 import get_wuji_hand2_cfg

RIGHT_FINGER_JOINT_NAMES = tuple(
  f"right_finger{finger}_joint{joint}"
  for finger in range(1, 6)
  for joint in range(1, 5)
)
RIGHT_HAND_LINK_NAMES = ("right_palm_link",) + tuple(
  f"right_finger{finger}_link{link}" for finger in range(1, 6) for link in range(1, 5)
)
# Physical hand2 wrist pose, shared by the clip reference and simulator.
RIGHT_FIXED_WRIST_POSE_W = (
  -0.028110361960901435,
  0.00300004,
  0.4952974018391883,
  0.5416752204197018,
  0.45451947767204376,
  0.45451947767204376,
  0.5416752204197018,
)
PEN_SPIN_HAND2_ROBOT_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=RIGHT_FIXED_WRIST_POSE_W[:3],
  rot=RIGHT_FIXED_WRIST_POSE_W[3:],
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)
# Hand collision geoms of the shared right_mjlab.xml. The pen is a cylinder, so
# each fingertip touches it at a single point; with condim 3 a two-finger pinch
# leaves the pen free to pivot about the contact normal, which is the twirl
# axis. Apply pen-spin's contact tuning at task load time: the upstream hand2
# XML intentionally contains no task-specific friction/solver settings.
# Keep hand2's nail and fleshy tip as independent mesh colliders. Contact
# tuning and DR cover both pieces; the self-contact penalty merges each pair.
PEN_SPIN_FINGER_CONTACT_GEOMS = (r"^right_finger[1-5]_(link[1-4]|tip_sensor)_col$",)
PEN_SPIN_HAND_CONTACT_GEOMS = (
  r"^right_palm_collision$",
) + PEN_SPIN_FINGER_CONTACT_GEOMS
HAND2_CONTACT_CONDIM = 4
HAND2_MUJOCO_IMPRATIO = 2.0
HAND2_NCONMAX = 192
HAND2_NJMAX = 4096


# Same hand2 body-local capsule definitions as Cube Reorient. Kept task-local
# so changing a PenSpin capsule does not alter the cube task.
HAND2_PEN_SPIN_CAPSULES = {
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


@dataclass(frozen=True)
class ActuatorCalibration:
  kp: float
  kv: float
  torque_limit: float
  damping: float
  frictionloss: float
  armature: float


# MIT gains and current limit; must match gains: in
# deploy/pen_spin/config/control.yaml.
PEN_SPIN_HAND2_MIT_KP = 3.0
# joint1-joint4 of every finger.
PEN_SPIN_HAND2_MIT_KD = (0.075, 0.07, 0.05, 0.05)
PEN_SPIN_HAND2_CURRENT_LIMIT_A = 1.5
PEN_SPIN_HAND2_AMPS_PER_MIT_UNIT = 1.9114709702965849
# Output torque constants, N*m/A.
PEN_SPIN_HAND2_TORQUE_CONSTANT = (
  0.4136,
  0.3545,
  0.2216,
  0.1329,
  0.4136,
  0.1673,
  0.1673,
  0.1673,
  0.4136,
  0.1673,
  0.1673,
  0.1673,
  0.4136,
  0.1673,
  0.1673,
  0.1673,
  0.4461,
  0.1115,
  0.1115,
  0.1115,
)
PEN_SPIN_HAND2_KP = tuple(
  kt * PEN_SPIN_HAND2_AMPS_PER_MIT_UNIT * PEN_SPIN_HAND2_MIT_KP
  for kt in PEN_SPIN_HAND2_TORQUE_CONSTANT
)
PEN_SPIN_HAND2_KV = tuple(
  kt * PEN_SPIN_HAND2_AMPS_PER_MIT_UNIT * PEN_SPIN_HAND2_MIT_KD[i % 4]
  for i, kt in enumerate(PEN_SPIN_HAND2_TORQUE_CONSTANT)
)
PEN_SPIN_HAND2_TORQUE_LIMIT = tuple(
  kt * PEN_SPIN_HAND2_CURRENT_LIMIT_A for kt in PEN_SPIN_HAND2_TORQUE_CONSTANT
)
# Joint caps retained by the original pen-spin baseline's right_mjlab.xml,
# before the shared asset switched to hand2. Keep these separate from the
# identified actuator limits: XmlActuator effort DR scales only the latter.
PEN_SPIN_BASELINE_JOINT_TORQUE_LIMITS = {
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
PEN_SPIN_HAND2_ARMATURE = (
  1.457162e-4,
  1.366389e-4,
  1.273553e-4,
  8.768902e-5,
  1.130474e-4,
  5.127161e-5,
  4.843783e-5,
  7.906193e-5,
  2.011048e-4,
  7.106655e-5,
  6.054746e-5,
  5.659272e-5,
  1.393897e-4,
  3.544108e-5,
  5.233974e-5,
  6.047948e-5,
  9.706259e-5,
  3.826054e-5,
  4.664456e-5,
  4.743778e-5,
)
PEN_SPIN_HAND2_DAMPING = (
  0.0026915050,
  0.0042812198,
  0.0014135711,
  0.0005131806,
  0.0010510923,
  0.0,
  0.0010302809,
  0.0000663526,
  0.0021587672,
  0.0008842859,
  0.0012610488,
  0.0001128493,
  0.0014506542,
  0.0,
  0.0007149635,
  0.0,
  0.0,
  0.0,
  0.0016246495,
  0.0,
)
PEN_SPIN_HAND2_FRICTIONLOSS = (
  0.0367355430,
  0.0342266392,
  0.0187921182,
  0.0122545518,
  0.0377084683,
  0.0177932607,
  0.0297927367,
  0.0161822467,
  0.0355644840,
  0.0240616629,
  0.0127755677,
  0.0109169800,
  0.0528257763,
  0.0266927477,
  0.0148677454,
  0.0215947297,
  0.0338251954,
  0.0261331247,
  0.0142008972,
  0.0129471168,
)


def get_calibration(joint_names: tuple[str, ...]) -> dict[str, ActuatorCalibration]:
  if joint_names != RIGHT_FINGER_JOINT_NAMES:
    raise ValueError(
      f"Wuji Hand 2 joint order mismatch: expected {RIGHT_FINGER_JOINT_NAMES}, got {joint_names}"
    )
  return {
    name: ActuatorCalibration(
      PEN_SPIN_HAND2_KP[i],
      PEN_SPIN_HAND2_KV[i],
      PEN_SPIN_HAND2_TORQUE_LIMIT[i],
      PEN_SPIN_HAND2_DAMPING[i],
      PEN_SPIN_HAND2_FRICTIONLOSS[i],
      PEN_SPIN_HAND2_ARMATURE[i],
    )
    for i, name in enumerate(joint_names)
  }


def apply_hand2_identified(spec_fn) -> mujoco.MjSpec:
  """Load the shared MJCF with pen-spin's identified Wuji Hand 2 parameters."""
  spec = spec_fn()
  calibration = get_calibration(tuple(actuator.target for actuator in spec.actuators))
  for actuator in spec.actuators:
    params = calibration[actuator.target]
    actuator.gainprm[0] = params.kp
    actuator.biasprm[1] = -params.kp
    actuator.biasprm[2] = -params.kv
    actuator.forcerange = (-params.torque_limit, params.torque_limit)
    joint = spec.joint(actuator.target)
    joint_limit = PEN_SPIN_BASELINE_JOINT_TORQUE_LIMITS[actuator.target]
    joint.actfrcrange = (-joint_limit, joint_limit)
    joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
    joint.damping[0] = params.damping
    joint.frictionloss = params.frictionloss
    joint.armature = params.armature
  return spec


def apply_hand2_pen_spin_collision_overlay(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  collision_meshes = {geom.name: geom for geom in spec.geoms}
  for body_name, (fromto, radius) in HAND2_PEN_SPIN_CAPSULES.items():
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


def get_hand2_pen_spin_spec(
  base_spec_fn: Callable[[], mujoco.MjSpec],
) -> mujoco.MjSpec:
  """Load hand2 with PenSpin calibration and the Cube-style collision overlay."""
  spec = apply_hand2_identified(base_spec_fn)
  return apply_hand2_pen_spin_collision_overlay(spec)


def get_pen_spin_hand2_cfg() -> EntityCfg:
  """The shared right Wuji Hand 2 on its fixed pen mount, with torsional friction."""
  cfg = get_wuji_hand2_cfg()
  cfg.spec_fn = partial(get_hand2_pen_spin_spec, cfg.spec_fn)
  cfg.init_state = PEN_SPIN_HAND2_ROBOT_INIT_STATE
  cfg.geoms = (
    GeomCfg(
      geom_names_expr=PEN_SPIN_HAND_CONTACT_GEOMS,
      condim=HAND2_CONTACT_CONDIM,
      priority=1,
      friction=(0.7, 0.005, 0.0001),
      solref=(0.02, 1.5),
      solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
    ),
  )
  return cfg
