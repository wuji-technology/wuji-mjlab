# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Reorient deploy constants, generation-parametrized (gen1 wuji_hand, gen2 wuji_hand2)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# 20 joints, finger-major (finger1=thumb … finger5=pinky), joint1..4 within finger.
# This is the ONNX action/obs order AND the wuji_sdk (5,4) row-major order.
_JOINTS_PER_FINGER = 4
_NUM_FINGERS = 5
NUM_JOINTS = _NUM_FINGERS * _JOINTS_PER_FINGER


def joint_names(hand_side: str) -> list[str]:
  """Build the ordered joint names for a hand."""
  if hand_side != "right":
    raise ValueError(f"hand_side must be 'right', got {hand_side!r}")
  return [
    f"{hand_side}_finger{f}_joint{j}"
    for f in range(1, _NUM_FINGERS + 1)
    for j in range(1, _JOINTS_PER_FINGER + 1)
  ]


HISTORY_LENGTH = 3
# Term order + per-frame dims: the group is term-major, history flattened
# oldest→newest.
OBS_TERMS = (
  ("noisy_joint_angles", NUM_JOINTS),
  ("qpos_error", NUM_JOINTS),
  ("cube_pos_in_tag", 3),
  ("cube_ori_error", 6),
  ("action_history", NUM_JOINTS),
)
OBS_DIM = HISTORY_LENGTH * sum(d for _, d in OBS_TERMS)
ACTION_DIM = NUM_JOINTS

assert OBS_DIM == HISTORY_LENGTH * 69 and ACTION_DIM == 20

_TAG_POSE = {
  1: ((0.0262, 0.0, -0.0563), (0.70710678, 0.0, 0.70710678, 0.0)),
  2: (
    (-0.00299939065, 0.0371578892, 0.063237),
    (0.4999959, -0.5000041, -0.4999959, -0.5000041),
  ),
}


@lru_cache(maxsize=2)
def _load_tag_pose(gen: int) -> tuple[np.ndarray, np.ndarray]:
  pos, quat = _TAG_POSE[gen]
  return np.array(pos, dtype=np.float64), np.array(quat, dtype=np.float64)


@dataclass
class Constants:
  """Per-generation deploy constants; build one with ``load_constants``."""

  num_joints: int
  default_joint_pos: np.ndarray
  soft_lower: np.ndarray
  soft_upper: np.ndarray
  soft_center: np.ndarray
  soft_half_range: np.ndarray
  root_pos: np.ndarray           # (3,)  viewer-only; policy obs are frame-agnostic
  root_quat: np.ndarray          # (4,)  (w,x,y,z)
  tag_in_palm_pos: np.ndarray    # (3,)  wrist AprilTag pose relative to palm_link
  tag_in_palm_quat: np.ndarray   # (4,)  (w,x,y,z)
  history_length: int
  obs_terms: tuple[tuple[str, int], ...]
  obs_dim: int
  action_dim: int
  sim_kp: np.ndarray
  sim_kd: np.ndarray





_GEN2_SOFT_LOWER = np.array([
  -1.0631, -1.3749, -0.91615, -0.91615,
  -0.91615, -0.6282, -0.88995, -0.91615,
  -0.91615, -0.6282, -0.88995, -0.91615,
  -0.91615, -0.6282, -0.88995, -0.91615,
  -0.91615, -0.6282, -0.88995, -0.91615,
], dtype=np.float64)

_GEN2_SOFT_UPPER = np.array([
  1.1671, 0.5889, 1.43915, 1.43915,
  1.43915, 0.6282, 1.93695, 1.43915,
  1.43915, 0.6282, 1.93695, 1.43915,
  1.43915, 0.6282, 1.93695, 1.43915,
  1.43915, 0.6282, 1.93695, 1.43915,
], dtype=np.float64)

_GEN2_DEFAULT_JOINT_POS = np.array([
  0.8, -0.3, 0.5, 0.4,
  0.45, -0.15, 0.8, 0.5,
  0.45, 0.0, 0.9, 0.3,
  0.6, 0.15, 1.1, 0.2,
  1.2, 0.2, 0.8, 0.4,
], dtype=np.float64)

# Per-joint position-actuator gains applied by the Wuji Hand 2 reorient task layer,
# in the same finger-major order as everything else here.
_GEN2_SIM_KP = np.array([
  1.354985, 2.198821, 0.886467, 0.654026,
  1.345207, 0.889460, 0.779361, 0.656424,
  1.299313, 0.942641, 0.737250, 0.710314,
  1.276892, 1.013334, 0.859159, 0.648064,
  1.102915, 0.910201, 0.861968, 0.656704,
], dtype=np.float64)

_GEN2_SIM_KD = np.array([
  0.047690, 0.062838, 0.027982, 0.020457,
  0.047402, 0.027239, 0.024489, 0.020508,
  0.045699, 0.027846, 0.023408, 0.022366,
  0.046209, 0.028704, 0.027436, 0.020448,
  0.043816, 0.026593, 0.027476, 0.020611,
], dtype=np.float64)



# Palm-up root pose of the Wuji Hand 2 palm in the world frame.
_GEN2_ROOT_POS = np.array(
  [-0.0284999, 0.00300004, 0.500250158], dtype=np.float64
)
_GEN2_ROOT_QUAT = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)





_GEN1_SOFT_LOWER = np.array([
  0.12529, -0.085145, -0.362875, -0.368565,
  -0.072555, -0.333, -0.37639, -0.36612,
  -0.0786, -0.333, -0.372645, -0.366255,
  -0.069705, -0.333, -0.37524, -0.375645,
  -0.076545, -0.333, -0.37551, -0.36621,
], dtype=np.float64)

_GEN1_SOFT_UPPER = np.array([
  1.52551, 0.878845, 1.460975, 1.455465,
  1.474455, 0.333, 1.44719, 1.47312,
  1.4658, 0.333, 1.449945, 1.472355,
  1.472805, 0.333, 1.44744, 1.461345,
  1.472445, 0.333, 1.44771, 1.47141,
], dtype=np.float64)

# = REORIENT_JOINT_POS (reorient_constants.py), the gen1 cage home pose.
_GEN1_DEFAULT_JOINT_POS = np.array([
  0.8, -0.0215, 0.285, 0.582,
  0.441, 0.163, 0.822, 0.494,
  0.448, -0.0832, 0.883, 0.305,
  0.594, -0.268, 1.13, 0.18,
  1.2, -0.228, 0.794, 0.407,
], dtype=np.float64)

# Per-joint training-time MJCF position-actuator gains (assets/robots/wuji_hand/
# mjcf/right_mjlab.xml <position kp=... kv=...>).
_GEN1_SIM_KP = np.array([
  0.408447, 0.685860, 0.239111, 0.207361,
  0.373522, 0.455924, 0.243684, 0.180270,
  0.368709, 0.416425, 0.222186, 0.194276,
  0.357182, 0.429773, 0.249302, 0.228503,
  0.365533, 0.413931, 0.227294, 0.196472,
], dtype=np.float64)

_GEN1_SIM_KD = np.array([
  0.020882, 0.030611, 0.010181, 0.009097,
  0.018823, 0.019798, 0.010477, 0.008240,
  0.018486, 0.018033, 0.009592, 0.009153,
  0.018377, 0.018677, 0.010595, 0.009918,
  0.018617, 0.018732, 0.009510, 0.009017,
], dtype=np.float64)


# reorient_constants.REORIENT_ROBOT_ROOT_POS/ROT (same palm-up mount as gen2).
_GEN1_ROOT_POS = np.array([0.0, 0.0, 0.5], dtype=np.float64)
_GEN1_ROOT_QUAT = np.array([0.70710678, 0.0, -0.70710678, 0.0], dtype=np.float64)  # R_y(-90°)


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


def load_constants(gen: int, hand_side: str) -> Constants:
  """Build the frozen deploy ``Constants`` for (gen, hand_side)."""
  if gen not in (1, 2):
    raise ValueError(f"gen must be 1 or 2, got {gen!r}")
  if hand_side != "right":
    raise ValueError(f"hand_side must be 'right', got {hand_side!r}")

  if gen == 2:
    default = _GEN2_DEFAULT_JOINT_POS
    lower, upper = _GEN2_SOFT_LOWER, _GEN2_SOFT_UPPER
    root_pos, root_quat = _GEN2_ROOT_POS, _GEN2_ROOT_QUAT
    tag_pos, tag_quat = _load_tag_pose(gen)
    sim_kp, sim_kd = _GEN2_SIM_KP, _GEN2_SIM_KD
  else:
    default = _GEN1_DEFAULT_JOINT_POS
    lower, upper = _GEN1_SOFT_LOWER, _GEN1_SOFT_UPPER
    root_pos, root_quat = _GEN1_ROOT_POS, _GEN1_ROOT_QUAT
    tag_pos, tag_quat = _load_tag_pose(gen)
    sim_kp, sim_kd = _GEN1_SIM_KP, _GEN1_SIM_KD

  return Constants(
    num_joints=NUM_JOINTS,
    default_joint_pos=default.copy(),
    soft_lower=lower.copy(),
    soft_upper=upper.copy(),
    soft_center=0.5 * (lower + upper),
    soft_half_range=0.5 * (upper - lower),
    root_pos=root_pos.copy(),
    root_quat=root_quat.copy(),
    tag_in_palm_pos=tag_pos.copy(),
    tag_in_palm_quat=tag_quat.copy(),
    history_length=HISTORY_LENGTH,
    obs_terms=OBS_TERMS,
    obs_dim=OBS_DIM,
    action_dim=ACTION_DIM,
    sim_kp=sim_kp.copy(),
    sim_kd=sim_kd.copy(),
  )
