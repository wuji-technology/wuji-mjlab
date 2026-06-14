# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Robot-specific bindings for the shared in-hand reorient task."""

from __future__ import annotations

from dataclasses import dataclass

from wuji_mjlab.tasks.reorient.reorient_constants import (
  TAG_IN_PALM_POS,
  TAG_IN_PALM_QUAT_WXYZ,
)


@dataclass(frozen=True)
class ReorientRobotBinding:
  """Robot naming/contact metadata required by the shared reorient MDP."""

  name: str
  joint_names: tuple[str, ...]
  palm_body_names: tuple[str, ...]
  palm_body_pattern: str
  viewer_body_name: str
  tip_site_names: tuple[str, ...]
  tip_body_names: tuple[str, ...]
  tip_collision_geoms: tuple[str, ...]
  palm_object_found_bodies: tuple[str, ...]
  distal_finger_object_found_bodies: tuple[str, ...]
  undesired_object_contact_bodies: tuple[str, ...]
  finger_collision_primary_geoms: tuple[str, ...]
  finger_collision_subtree_body: str
  cage_body_names: tuple[str, ...]
  dr_robot_geoms: tuple[str, ...]
  geom_size_randomization_geoms: tuple[str, ...] | None
  contact_params_palm_thumb_geoms: tuple[str, ...]
  contact_params_fingers_geoms: tuple[str, ...]
  contact_params_palm_thumb_width_range: tuple[float, float]
  contact_params_fingers_width_range: tuple[float, float]
  joint_reset_position_range: dict[str, tuple[float | None, float | None]]
  tag_in_palm_pos: tuple[float, float, float]
  tag_in_palm_quat: tuple[float, float, float, float]


WUJI_RIGHT_JOINT_NAMES: tuple[str, ...] = tuple(
  f"right_finger{finger}_joint{joint}"
  for finger in range(1, 6)
  for joint in range(1, 5)
)

WUJI_RIGHT_HAND_BINDING = ReorientRobotBinding(
  name="wuji_right",
  joint_names=WUJI_RIGHT_JOINT_NAMES,
  palm_body_names=(".*_palm_link",),
  palm_body_pattern=".*_palm_link",
  viewer_body_name="right_palm_link",
  tip_site_names=(".*_finger[1-5]_tip",),
  tip_body_names=(".*_finger[1-5]_link4",),
  tip_collision_geoms=(".*_finger[1-5]_link4_col",),
  palm_object_found_bodies=(".*_palm_link", ".*_finger.*_link[12]"),
  distal_finger_object_found_bodies=(".*_finger.*_link[34]",),
  undesired_object_contact_bodies=(
    ".*_palm_link",
    ".*_finger1_link1",
    ".*_finger2_link1",
    ".*_finger2_link2",
    ".*_finger2_link3",
    ".*_finger3_link1",
    ".*_finger3_link2",
    ".*_finger3_link3",
    ".*_finger4_link1",
    ".*_finger4_link2",
    ".*_finger4_link3",
    ".*_finger5_link1",
    ".*_finger5_link2",
  ),
  finger_collision_primary_geoms=(".*finger.*_col",),
  finger_collision_subtree_body="right_palm_link",
  cage_body_names=(".*_palm_link", ".*_finger.*_link[1-4]"),
  dr_robot_geoms=(".*palm_.*", ".*finger.*_col"),
  geom_size_randomization_geoms=(r".*finger[1-5]_link[2-3]_col",),
  contact_params_palm_thumb_geoms=(
    "right_palm_collision",
    "right_finger1_link2_col",
    "right_finger1_link2_softbody_col",
    "right_finger1_link3_col",
    "right_finger1_link4_col",
  ),
  contact_params_fingers_geoms=(r".*finger[2-5]_link[2-4]_col",),
  contact_params_palm_thumb_width_range=(2.0, 5.0),
  contact_params_fingers_width_range=(1.0, 2.0),
  joint_reset_position_range={
    ".*_joint1": (-0.3, -0.1),
    ".*_joint2": (0.0, 0.0),
    ".*_joint3": (-0.3, -0.1),
    ".*_joint4": (-0.3, -0.1),
  },
  tag_in_palm_pos=TAG_IN_PALM_POS,
  tag_in_palm_quat=TAG_IN_PALM_QUAT_WXYZ,
)

REVO3_RIGHT_JOINT_NAMES: tuple[str, ...] = (
  "right_thumb_CMP_joint",
  "right_thumb_CMR_joint",
  "right_thumb_MCP_joint",
  "right_thumb_PIP_joint",
  "right_thumb_DIP_joint",
  "right_index_MPR_joint",
  "right_index_MCP_joint",
  "right_index_PIP_joint",
  "right_index_DIP_joint",
  "right_middle_MPR_joint",
  "right_middle_MCP_joint",
  "right_middle_PIP_joint",
  "right_middle_DIP_joint",
  "right_ring_MPR_joint",
  "right_ring_MCP_joint",
  "right_ring_PIP_joint",
  "right_ring_DIP_joint",
  "right_little_MPR_joint",
  "right_little_MCP_joint",
  "right_little_PIP_joint",
  "right_little_DIP_joint",
)

REVO3_DISTAL_BODIES: tuple[str, ...] = (
  "right_thumb_DIP_Link",
  "right_index_DIP_Link",
  "right_middle_DIP_Link",
  "right_ring_DIP_Link",
  "right_little_DIP_Link",
)

REVO3_PROXIMAL_BODIES: tuple[str, ...] = (
  "right_hand_base_link",
  "right_palm",
  "right_thumb_CMP_Link",
  "right_thumb_CMR_Link",
  "right_thumb_MCP_Link",
  "right_thumb_PIP_Link",
  "right_index_MPR_Link",
  "right_index_MCP_Link",
  "right_index_PIP_Link",
  "right_middle_MPR_Link",
  "right_middle_MCP_Link",
  "right_middle_PIP_Link",
  "right_ring_MPR_Link",
  "right_ring_MCP_Link",
  "right_ring_PIP_Link",
  "right_little_MPR_Link",
  "right_little_MCP_Link",
  "right_little_PIP_Link",
)

REVO3_RIGHT_HAND_BINDING = ReorientRobotBinding(
  name="revo3_right",
  joint_names=REVO3_RIGHT_JOINT_NAMES,
  palm_body_names=("right_palm",),
  palm_body_pattern="right_palm",
  viewer_body_name="right_palm",
  tip_site_names=(
    "right_thumb_tip",
    "right_index_tip",
    "right_middle_tip",
    "right_ring_tip",
    "right_little_tip",
  ),
  tip_body_names=REVO3_DISTAL_BODIES,
  tip_collision_geoms=(
    "right_thumb_DIP_Link_collision_0",
    "right_index_DIP_Link_collision_0",
    "right_middle_DIP_Link_collision_0",
    "right_ring_DIP_Link_collision_0",
    "right_little_DIP_Link_collision_0",
  ),
  palm_object_found_bodies=REVO3_PROXIMAL_BODIES,
  distal_finger_object_found_bodies=REVO3_DISTAL_BODIES,
  undesired_object_contact_bodies=REVO3_PROXIMAL_BODIES,
  finger_collision_primary_geoms=("right_.*_Link_collision_0",),
  finger_collision_subtree_body="right_hand_base_link",
  cage_body_names=("right_hand_base_link", "right_palm", "right_.*_Link"),
  dr_robot_geoms=("right_hand_base_link_collision_.*", "right_.*_Link_collision_0"),
  geom_size_randomization_geoms=None,
  contact_params_palm_thumb_geoms=(
    "right_hand_base_link_collision_.*",
    "right_thumb_.*_Link_collision_0",
  ),
  contact_params_fingers_geoms=(
    "right_index_.*_Link_collision_0",
    "right_middle_.*_Link_collision_0",
    "right_ring_.*_Link_collision_0",
    "right_little_.*_Link_collision_0",
  ),
  contact_params_palm_thumb_width_range=(1.0, 2.0),
  contact_params_fingers_width_range=(1.0, 2.0),
  joint_reset_position_range={
    "right_.*_MPR_joint": (-0.1, 0.1),
    "right_.*_MCP_joint": (-0.3, -0.1),
    "right_.*_PIP_joint": (-0.3, -0.1),
    "right_.*_DIP_joint": (-0.3, -0.1),
    "right_thumb_CMP_joint": (-0.1, 0.1),
    "right_thumb_CMR_joint": (-0.1, 0.1),
  },
  tag_in_palm_pos=(0.0, 0.0, 0.0),
  tag_in_palm_quat=(1.0, 0.0, 0.0, 0.0),
)
