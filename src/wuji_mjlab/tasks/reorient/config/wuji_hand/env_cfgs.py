# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji Hand Reorient environment configurations."""

from collections.abc import Callable
from functools import partial

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg

from wuji_mjlab.assets.objects.inhand_object.object_cfg import get_inhand_object_cfg
from wuji_mjlab.assets.robots.wuji_hand.wuji_hand_cfg import get_wuji_hand_cfg
from wuji_mjlab.tasks.reorient.reorient_constants import (
  REORIENT_CUBE_INIT_STATE,
  REORIENT_PALM_NORMAL_AXIS,
  REORIENT_ROBOT_INIT_STATE,
  REORIENT_WRIST_TAG_POS,
  REORIENT_WRIST_TAG_ROT,
)
from wuji_mjlab.tasks.reorient.reorient_env_cfg import make_reorient_env_cfg


def _get_reorient_spec(
  base_spec_fn: Callable[[], mujoco.MjSpec], hand_side: str
) -> mujoco.MjSpec:
  spec = base_spec_fn()
  spec.body(f"{hand_side}_palm_link").add_site(
    name=f"{hand_side}_wrist_tag",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=REORIENT_WRIST_TAG_POS,
    quat=REORIENT_WRIST_TAG_ROT,
    size=(0.0252, 0.0252, 0.0006),
    rgba=(1.0, 0.9, 0.0, 0.5),
    group=1,
  )
  return spec


def get_wuji_hand_rig_cfg(hand_side: str = "right") -> EntityCfg:
  """Build the Wuji Hand 1 entity with its reorient wrist-tag site."""
  robot_cfg = get_wuji_hand_cfg(hand_side)
  robot_cfg.spec_fn = partial(_get_reorient_spec, robot_cfg.spec_fn, hand_side)
  return robot_cfg


def wuji_hand_reorient_env_cfg(
  play: bool = False, num_envs: int = 8192
) -> ManagerBasedRlEnvCfg:
  """Create the Wuji Hand Reorient task configuration."""
  cfg = make_reorient_env_cfg(
    play=play,
    num_envs=num_envs,
    cage_up_axis=REORIENT_PALM_NORMAL_AXIS,
  )
  cfg.scene.entities = {
    "robot": get_wuji_hand_rig_cfg(),
    "object": get_inhand_object_cfg(),
  }
  cfg.scene.entities["robot"].init_state = REORIENT_ROBOT_INIT_STATE
  cfg.scene.entities["object"].init_state = REORIENT_CUBE_INIT_STATE
  cfg.viewer.body_name = "right_palm_link"
  return cfg
