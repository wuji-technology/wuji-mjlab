# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Revo3 right-hand Reorient environment configurations."""

import math

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from wuji_mjlab.assets.objects.inhand_object.object_cfg import get_inhand_object_cfg
from wuji_mjlab.assets.robots.revo3_hand.revo3_hand_cfg import get_revo3_hand_cfg
from wuji_mjlab.tasks.reorient import mdp
from wuji_mjlab.tasks.reorient.reorient_env_cfg import make_reorient_env_cfg
from wuji_mjlab.tasks.reorient.reorient_terms import (
  ReorientRobustRandomizationCfg,
  apply_reorient_robust_randomization,
  build_repose_reward_finetune_reorient_rewards,
  build_repose_reward_reorient_rewards,
)
from wuji_mjlab.tasks.reorient.robot_bindings import REVO3_RIGHT_HAND_BINDING

REVO3_ROBOT_ROOT_POS = (0.0, 0.0, 0.5)
_SQRT_HALF = math.sqrt(0.5)
REVO3_ROBOT_ROOT_ROT = (_SQRT_HALF, 0.0, -_SQRT_HALF, 0.0)
REVO3_PALM_POS_IN_ROOT = (0.0155, 0.0, 0.055)
REVO3_CUBE_OFFSET_IN_PALM = (0.055, 0.005, 0.075)
REVO3_CUBE_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
REVO3_FINE_TUNE_ROBUST_RANDOMIZATION_CFG = ReorientRobustRandomizationCfg()


def _quat_apply(
  quat_wxyz: tuple[float, float, float, float],
  vec_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
  w, x, y, z = quat_wxyz
  vx, vy, vz = vec_xyz
  tx = 2.0 * (y * vz - z * vy)
  ty = 2.0 * (z * vx - x * vz)
  tz = 2.0 * (x * vy - y * vx)
  return (
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx),
  )


def _add3(
  lhs: tuple[float, float, float],
  rhs: tuple[float, float, float],
) -> tuple[float, float, float]:
  return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


def _round_pose(vec_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
  return tuple(round(v, 10) for v in vec_xyz)


REVO3_CUBE_INIT_POS = _round_pose(
  _add3(
    REVO3_ROBOT_ROOT_POS,
    _quat_apply(
      REVO3_ROBOT_ROOT_ROT,
      _add3(REVO3_PALM_POS_IN_ROOT, REVO3_CUBE_OFFSET_IN_PALM),
    ),
  )
)

REVO3_REORIENT_CUBE_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=REVO3_CUBE_INIT_POS,
  rot=REVO3_CUBE_INIT_ROT,
)

REVO3_REORIENT_ROBOT_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=REVO3_ROBOT_ROOT_POS,
  rot=REVO3_ROBOT_ROOT_ROT,
  joint_pos={
    "right_thumb_CMP_joint": 0.8,
    "right_thumb_CMR_joint": 0.4,
    "right_thumb_MCP_joint": 0.285,
    "right_thumb_PIP_joint": 0.582,
    "right_thumb_DIP_joint": 0.582,
    "right_index_MPR_joint": 0.163,
    "right_index_MCP_joint": 0.441,
    "right_index_PIP_joint": 0.822,
    "right_index_DIP_joint": 0.494,
    "right_middle_MPR_joint": -0.0832,
    "right_middle_MCP_joint": 0.448,
    "right_middle_PIP_joint": 0.883,
    "right_middle_DIP_joint": 0.305,
    "right_ring_MPR_joint": -0.2618,
    "right_ring_MCP_joint": 0.594,
    "right_ring_PIP_joint": 1.13,
    "right_ring_DIP_joint": 0.18,
    "right_little_MPR_joint": -0.228,
    "right_little_MCP_joint": 1.2,
    "right_little_PIP_joint": 0.794,
    "right_little_DIP_joint": 0.407,
  },
  joint_vel={".*": 0.0},
)


def revo3_right_hand_reorient_env_cfg(
  play: bool = False, num_envs: int = 8192
) -> ManagerBasedRlEnvCfg:
  """Create Revo3 right-hand Reorient task configuration."""
  cfg = make_reorient_env_cfg(
    play=play,
    num_envs=num_envs,
    robot_binding=REVO3_RIGHT_HAND_BINDING,
  )
  cfg.scene.entities = {
    "robot": get_revo3_hand_cfg(),
    "object": get_inhand_object_cfg(),
  }
  cfg.scene.entities["robot"].init_state = REVO3_REORIENT_ROBOT_INIT_STATE
  cfg.scene.entities["object"].init_state = REVO3_REORIENT_CUBE_INIT_STATE
  cfg.events["reset_object_pose"].func = mdp.reset_object_pose_in_palm
  cfg.events["reset_object_pose"].params = {
    "cube_offset_in_palm": REVO3_CUBE_OFFSET_IN_PALM,
    "pos_noise": 0.0,
    "asset_cfg": SceneEntityCfg("object"),
    "robot_cfg": SceneEntityCfg(
      "robot", body_names=REVO3_RIGHT_HAND_BINDING.palm_body_names
    ),
  }
  cfg.viewer.body_name = REVO3_RIGHT_HAND_BINDING.viewer_body_name
  return cfg


def revo3_right_hand_reorient_repose_reward_env_cfg(
  play: bool = False, num_envs: int = 8192
) -> ManagerBasedRlEnvCfg:
  """Create the copied Revo3 Reorient task with RevoLab repose rewards."""
  cfg = revo3_right_hand_reorient_env_cfg(play=play, num_envs=num_envs)
  cfg.rewards = build_repose_reward_reorient_rewards(
    cube_offset_in_palm=REVO3_CUBE_OFFSET_IN_PALM,
    robot_binding=REVO3_RIGHT_HAND_BINDING,
  )
  return cfg


def revo3_right_hand_reorient_repose_reward_finetune_env_cfg(
  play: bool = False,
  num_envs: int = 8192,
  robust_randomization: ReorientRobustRandomizationCfg | None = (
    REVO3_FINE_TUNE_ROBUST_RANDOMIZATION_CFG
  ),
) -> ManagerBasedRlEnvCfg:
  """Create the Revo3 ReposeReward fine-tune task with added penalties."""
  cfg = revo3_right_hand_reorient_env_cfg(play=play, num_envs=num_envs)
  cfg.rewards = build_repose_reward_finetune_reorient_rewards(
    cube_offset_in_palm=REVO3_CUBE_OFFSET_IN_PALM,
    robot_binding=REVO3_RIGHT_HAND_BINDING,
  )
  if not play and robust_randomization is not None:
    apply_reorient_robust_randomization(cfg, robust_randomization)
  return cfg
