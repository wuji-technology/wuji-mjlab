# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji Hand 2 Reorient environment configurations."""

from functools import partial

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg

from wuji_mjlab.assets.objects.inhand_object.object_cfg import get_inhand_object_cfg
from wuji_mjlab.assets.robots.wuji_hand2.wuji_hand2_cfg import get_wuji_hand2_cfg
from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
  REORIENT_HAND2_MOUNT_POS,
  REORIENT_HAND2_PALM_NORMAL_AXIS,
  REORIENT_HAND2_ROBOT_INIT_STATE,
  get_hand2_reorient_spec,
  hand2_cube_init_state,
)
from wuji_mjlab.tasks.reorient.reorient_env_cfg import make_reorient_env_cfg
from wuji_mjlab.tasks.reorient.reorient_terms import HAND1_GEOM_SIZE_DR_GEOMS

HAND2_TIP_COLLISION_GEOMS = (
  ".*_finger[1-5]_link4_col",
  ".*_finger[1-5]_tip_sensor_col",
)

HAND2_TIP_CONTACT_BODY_NAMES = (
  ".*_finger[1-5]_link4",
  ".*_finger[1-5]_tip_sensor",
)

HAND2_DISTAL_FINGER_OBJECT_BODY_NAMES = (
  ".*_finger.*_link3",
  *HAND2_TIP_CONTACT_BODY_NAMES,
)

HAND2_FINGER_CONTACT_PARAM_GEOMS = (
  r".*finger[2-5]_link[2-4]_col",
  r".*finger[2-5]_tip_sensor_col",
)


def get_wuji_hand2_rig_cfg(hand_side: str = "right") -> EntityCfg:
  """Build the hand entity with the task-level calibration rig attached."""
  robot_cfg = get_wuji_hand2_cfg(hand_side)
  robot_cfg.spec_fn = partial(get_hand2_reorient_spec, robot_cfg.spec_fn)
  return robot_cfg


def wuji_hand2_reorient_env_cfg(
  hand_side: str = "right",
  play: bool = False,
  num_envs: int = 8192,
) -> ManagerBasedRlEnvCfg:
  """Create the Wuji Hand 2 Reorient task configuration.

  Args:
    hand_side: Only ``"right"`` exists in the MJCF."""
  if hand_side != "right":
    raise ValueError(f"Unsupported hand_side '{hand_side}'. Only 'right' is supported.")
  cfg = make_reorient_env_cfg(
    play=play,
    num_envs=num_envs,
    palm_subtree_body=f"{hand_side}_palm_link",
    soft_pad_geoms=(
      f"{hand_side}_palm_collision",
      f"{hand_side}_finger1_link2_col",
      f"{hand_side}_finger1_link3_col",
      f"{hand_side}_finger1_link4_col",
      f"{hand_side}_finger1_tip_sensor_col",
    ),
    # The overlay replaces L2/L3 collision meshes with capsules, so size DR is valid.
    geom_size_dr_geoms=HAND1_GEOM_SIZE_DR_GEOMS,
    tip_collision_geoms=HAND2_TIP_COLLISION_GEOMS,
    tip_contact_body_names=HAND2_TIP_CONTACT_BODY_NAMES,
    distal_finger_object_body_names=HAND2_DISTAL_FINGER_OBJECT_BODY_NAMES,
    finger_contact_param_geoms=HAND2_FINGER_CONTACT_PARAM_GEOMS,
    cage_up_axis=REORIENT_HAND2_PALM_NORMAL_AXIS,
    robot_tilt_pivot_in_root=REORIENT_HAND2_MOUNT_POS,
  )
  robot_cfg = get_wuji_hand2_rig_cfg(hand_side)
  cfg.scene.entities = {
    "robot": robot_cfg,
    "object": get_inhand_object_cfg(),
  }
  cfg.scene.entities["robot"].init_state = REORIENT_HAND2_ROBOT_INIT_STATE
  cfg.scene.entities["object"].init_state = hand2_cube_init_state(hand_side)
  cfg.viewer.body_name = f"{hand_side}_palm_link"
  return cfg
