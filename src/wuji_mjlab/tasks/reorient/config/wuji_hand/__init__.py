# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji Hand reorient task registration."""

from mjlab.tasks.registry import register_mjlab_task

from wuji_mjlab.rl.runner import WujiOnPolicyRunner

from .env_cfgs import wuji_hand_reorient_env_cfg
from .rsl_rl.ppo import wuji_hand_reorient_ppo_runner_cfg

register_mjlab_task(
  task_id="WujiHand_Reorient",
  env_cfg=wuji_hand_reorient_env_cfg(num_envs=8192),
  play_env_cfg=wuji_hand_reorient_env_cfg(play=True),
  rl_cfg=wuji_hand_reorient_ppo_runner_cfg(max_iterations=5000),
  runner_cls=WujiOnPolicyRunner,
)
