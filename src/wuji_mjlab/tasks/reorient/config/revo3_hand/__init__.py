# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Revo3 right-hand reorient task registration."""

from mjlab.tasks.registry import register_mjlab_task

from wuji_mjlab.rl.runner import WujiOnPolicyRunner

from .env_cfgs import (
  revo3_right_hand_reorient_env_cfg,
  revo3_right_hand_reorient_repose_reward_env_cfg,
  revo3_right_hand_reorient_repose_reward_finetune_env_cfg,
)
from .rsl_rl.ppo import revo3_right_hand_reorient_ppo_runner_cfg

register_mjlab_task(
  task_id="Revo3RightHand_Reorient",
  env_cfg=revo3_right_hand_reorient_env_cfg(num_envs=8192),
  play_env_cfg=revo3_right_hand_reorient_env_cfg(play=True),
  rl_cfg=revo3_right_hand_reorient_ppo_runner_cfg(max_iterations=5000),
  runner_cls=WujiOnPolicyRunner,
)

register_mjlab_task(
  task_id="Revo3RightHand_Reorient_Light",
  env_cfg=revo3_right_hand_reorient_env_cfg(num_envs=4096),
  play_env_cfg=revo3_right_hand_reorient_env_cfg(play=True),
  rl_cfg=revo3_right_hand_reorient_ppo_runner_cfg(
    run_name="Revo3_Reorient_Light", max_iterations=7500
  ),
  runner_cls=WujiOnPolicyRunner,
)

register_mjlab_task(
  task_id="Revo3RightHand_Reorient_ReposeReward",
  env_cfg=revo3_right_hand_reorient_repose_reward_env_cfg(num_envs=8192),
  play_env_cfg=revo3_right_hand_reorient_repose_reward_env_cfg(play=True),
  rl_cfg=revo3_right_hand_reorient_ppo_runner_cfg(
    run_name="Revo3_Reorient_ReposeReward",
    max_iterations=5000,
  ),
  runner_cls=WujiOnPolicyRunner,
)

register_mjlab_task(
  task_id="Revo3RightHand_Reorient_ReposeReward_FineTune",
  env_cfg=revo3_right_hand_reorient_repose_reward_finetune_env_cfg(num_envs=8192),
  play_env_cfg=revo3_right_hand_reorient_repose_reward_finetune_env_cfg(play=True),
  rl_cfg=revo3_right_hand_reorient_ppo_runner_cfg(
    run_name="Revo3_Reorient_ReposeReward_FineTune",
    max_iterations=5000,
  ),
  runner_cls=WujiOnPolicyRunner,
)
