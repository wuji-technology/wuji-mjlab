# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji Hand 2 pen-spin task registration."""

from dataclasses import replace

from mjlab.tasks.registry import register_mjlab_task

from wuji_mjlab.rl.runner import WujiOnPolicyRunner

from .env_cfgs import wuji_hand2_pen_spin_env_cfg
from .rsl_rl.ppo import wuji_hand2_pen_spin_ppo_runner_cfg


def _pen_spin_rl_cfg():
  cfg = wuji_hand2_pen_spin_ppo_runner_cfg()
  return replace(cfg, experiment_name="wuji_pen_spin_hand2")


register_mjlab_task(
  task_id="WujiHand2_PenSpin",
  env_cfg=wuji_hand2_pen_spin_env_cfg(),
  play_env_cfg=wuji_hand2_pen_spin_env_cfg(play=True),
  rl_cfg=_pen_spin_rl_cfg(),
  runner_cls=WujiOnPolicyRunner,
)

__all__ = ["wuji_hand2_pen_spin_env_cfg", "wuji_hand2_pen_spin_ppo_runner_cfg"]
