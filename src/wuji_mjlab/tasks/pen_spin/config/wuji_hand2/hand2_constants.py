# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Identified Wuji Hand 2 actuator and joint parameters used by pen-spin.

These values are the identified calibration for the right Wuji Hand 2 hand.  They
are applied to the shared MJCF at load time so the checked-in geometry XML
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


PEN_SPIN_HAND2_KP = (
  2.4261130417,
  2.0794416666,
  1.2998710108,
  0.7795706558,
  2.4261130417,
  0.9813556864,
  0.9813556864,
  0.9813556864,
  2.4261130417,
  0.9813556864,
  0.9813556864,
  0.9813556864,
  2.4261130417,
  0.9813556864,
  0.9813556864,
  0.9813556864,
  2.6167529689,
  0.6540415961,
  0.6540415961,
  0.6540415961,
)
PEN_SPIN_HAND2_KV = (
  0.0808704359,
  0.0693147233,
  0.0433290343,
  0.0259856889,
  0.0808704359,
  0.0327118567,
  0.0327118567,
  0.0327118567,
  0.0808704359,
  0.0327118567,
  0.0327118567,
  0.0327118567,
  0.0808704359,
  0.0327118567,
  0.0327118567,
  0.0327118567,
  0.0872251003,
  0.0218013869,
  0.0218013869,
  0.0218013869,
)
PEN_SPIN_HAND2_TORQUE_LIMIT = (
  0.6204,
  0.53175,
  0.3324,
  0.19935,
  0.6204,
  0.25095,
  0.25095,
  0.25095,
  0.6204,
  0.25095,
  0.25095,
  0.25095,
  0.6204,
  0.25095,
  0.25095,
  0.25095,
  0.66915,
  0.16725,
  0.16725,
  0.16725,
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
PEN_SPIN_HAND2_DAMPING = (
  0.0011481331,
  0.0028281253,
  0.0014018148,
  0.0010513064,
  0.0026457962,
  0.0001960150,
  0.0010739417,
  0.0006737062,
  0.0012442763,
  0.0,
  0.0003244629,
  0.0009038801,
  0.0026002904,
  0.0,
  0.0005600658,
  0.0000185669,
  0.0016515482,
  0.0,
  0.0001594732,
  0.0000986090,
)
PEN_SPIN_HAND2_FRICTIONLOSS = (
  0.0483964915,
  0.0347770044,
  0.0227845884,
  0.0379860220,
  0.0384875828,
  0.0199457769,
  0.0221054269,
  0.0105610665,
  0.0407692067,
  0.0201052160,
  0.0131031763,
  0.0191064985,
  0.0340672352,
  0.0288227380,
  0.0132650550,
  0.0174435280,
  0.0304631362,
  0.0236187851,
  0.0133160780,
  0.0109171043,
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
