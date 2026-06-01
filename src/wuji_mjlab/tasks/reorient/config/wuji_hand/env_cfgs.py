# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji Hand Reorient environment configurations.

Thin binding layer: attaches the Wuji Hand robot + cube object entities
and the robot-specific viewer target to the task cfg produced by
``make_reorient_env_cfg``. All task design (sensors, rewards, DR events,
play-mode overrides) lives in ``reorient_terms`` / ``reorient_env_cfg``.
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from wuji_mjlab.assets.objects.inhand_object.object_cfg import get_inhand_object_cfg
from wuji_mjlab.assets.robots.wuji_hand.wuji_hand_cfg import get_wuji_hand_cfg
from wuji_mjlab.tasks.reorient.reorient_constants import (
  REORIENT_CUBE_INIT_STATE,
  REORIENT_ROBOT_INIT_STATE,
)
from wuji_mjlab.tasks.reorient.reorient_env_cfg import make_reorient_env_cfg


def wuji_hand_reorient_env_cfg(
  play: bool = False, num_envs: int = 8192
) -> ManagerBasedRlEnvCfg:
  """Create Wuji Hand Reorient task configuration."""
  cfg = make_reorient_env_cfg(play=play, num_envs=num_envs)
  cfg.scene.entities = {
    "robot": get_wuji_hand_cfg(),
    "object": get_inhand_object_cfg(),
  }
  cfg.scene.entities["robot"].init_state = REORIENT_ROBOT_INIT_STATE
  cfg.scene.entities["object"].init_state = REORIENT_CUBE_INIT_STATE
  cfg.viewer.body_name = "right_palm_link"
  return cfg
