# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Reorient constants for Wuji Hand 1.

The root rotation below reproduces the mount's ``R_y(-90°)`` relationship.
"""

from mjlab.entity import EntityCfg

REORIENT_PALM_NORMAL_AXIS = 0


REORIENT_ROBOT_ROOT_POS = (0.0, 0.0, 0.5)
REORIENT_ROBOT_ROOT_ROT = (0.70710678, 0.0, -0.70710678, 0.0)

# Wuji Hand 1's reorient tag belongs to the task rig, not the shared robot XML.
REORIENT_WRIST_TAG_POS = (0.0262, 0.0, -0.0563)
REORIENT_WRIST_TAG_ROT = (0.70710678, 0.0, 0.70710678, 0.0)


REORIENT_CUBE_INIT_POS = (-0.0967, 0.0100, 0.5599)
REORIENT_CUBE_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

REORIENT_CUBE_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=REORIENT_CUBE_INIT_POS,
  rot=REORIENT_CUBE_INIT_ROT,
)

REORIENT_JOINT_POS: dict[str, float] = {
  ".*_finger1_joint1": 0.8,
  ".*_finger1_joint2": -0.0215,
  ".*_finger1_joint3": 0.285,
  ".*_finger1_joint4": 0.582,
  ".*_finger2_joint1": 0.441,
  ".*_finger2_joint2": 0.163,
  ".*_finger2_joint3": 0.822,
  ".*_finger2_joint4": 0.494,
  ".*_finger3_joint1": 0.448,
  ".*_finger3_joint2": -0.0832,
  ".*_finger3_joint3": 0.883,
  ".*_finger3_joint4": 0.305,
  ".*_finger4_joint1": 0.594,
  ".*_finger4_joint2": -0.268,
  ".*_finger4_joint3": 1.13,
  ".*_finger4_joint4": 0.18,
  ".*_finger5_joint1": 1.2,
  ".*_finger5_joint2": -0.228,
  ".*_finger5_joint3": 0.794,
  ".*_finger5_joint4": 0.407,
}

REORIENT_ROBOT_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=REORIENT_ROBOT_ROOT_POS,
  rot=REORIENT_ROBOT_ROOT_ROT,
  joint_pos=REORIENT_JOINT_POS,
  joint_vel={".*": 0.0},
)
