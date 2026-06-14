# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""RL configuration for the Revo3 right-hand Reorient task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from wuji_mjlab.tasks.reorient.config.wuji_hand.rsl_rl.ppo import (
  wuji_hand_reorient_ppo_runner_cfg,
)


def revo3_right_hand_reorient_ppo_runner_cfg(
  run_name: str = "Revo3_Reorient",
  max_iterations: int = 5000,
) -> RslRlOnPolicyRunnerCfg:
  return wuji_hand_reorient_ppo_runner_cfg(
    run_name=run_name,
    max_iterations=max_iterations,
    experiment_name="revo3_reorient",
    wandb_project="revo3_reorient_mjlab",
  )
