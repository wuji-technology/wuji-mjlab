# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji Hand 2 reorient task registration (right hand only)."""

from dataclasses import replace

from mjlab.tasks.registry import register_mjlab_task

from wuji_mjlab.rl.runner import WujiOnPolicyRunner

from .env_cfgs import wuji_hand2_reorient_env_cfg
from .rsl_rl.ppo import wuji_hand2_reorient_ppo_runner_cfg


def _hand2_rl_cfg(run_name: str, max_iterations: int):
  cfg = wuji_hand2_reorient_ppo_runner_cfg(
    run_name=run_name, max_iterations=max_iterations
  )
  return replace(cfg, experiment_name="wuji_reorient_hand2")


register_mjlab_task(
  task_id="WujiHand2_Reorient",
  env_cfg=wuji_hand2_reorient_env_cfg(num_envs=8192),
  play_env_cfg=wuji_hand2_reorient_env_cfg(play=True),
  rl_cfg=_hand2_rl_cfg(run_name="Reorient_Hand2", max_iterations=5000),
  runner_cls=WujiOnPolicyRunner,
)


_RATE_RATIO = 2.5  # 50 Hz / 20 Hz control steps


def _reorient_hand2_50hz_env_cfg(num_envs: int = 8192, play: bool = False):
  """50 Hz deployment recipe for right-hand operation.

  The hold / goal-switch / cage / episode window knobs are rescaled together
  (×2.5) to preserve their wall-clock duration; retune them together, not
  knob-by-knob.
  """
  cfg = wuji_hand2_reorient_env_cfg(hand_side="right", num_envs=num_envs, play=play)
  cfg.decimation = 2
  cfg.actions["joint_pos"].ema_alpha = 0.3
  cfg.rewards["action_rate"].weight = 0.0
  cfg.rewards["action_hf"].weight = -30.0
  cfg.commands["reorient_command"].success_hold_steps = 13  # 5 × 2.5
  cfg.commands["reorient_command"].goal_switch_delay = 50  # 20 × 2.5
  # per-step-count reward — ÷2.43 to hold the per-cycle integral (Σ1..N ∝ N²)
  cfg.rewards["hold_escalation"].weight = 4.694  # 11.4 × (210/1275) × 2.5
  # Curriculum is cleared in play/eval mode, so guard the key.
  if "adaptive_episode" in cfg.curriculum:
    cfg.curriculum["adaptive_episode"].params["target_steps"] = 2000  # 800 × 2.5
  cfg.rewards["cage_escape"].params["max_outside_steps"] = 25  # 10 × 2.5
  cfg.rewards["cage_escape"].params["max_count"] = 37.5  # 1.5 × 25
  cfg.terminations["cage_drop"].params["max_outside_steps"] = 25
  cfg.observations["critic"].terms["cage_counter"].params["max_outside_steps"] = 25
  return cfg


def _reorient_hand2_50hz_rl_cfg():
  cfg = _hand2_rl_cfg(run_name="Reorient_Hand2_50Hz", max_iterations=5000)
  cfg.num_steps_per_env = 100  # 40 × 2.5 → same 2 s rollout horizon
  cfg.algorithm.gamma = 0.996  # 0.99 ** 0.4 → same real-time discount horizon
  return cfg


register_mjlab_task(
  task_id="WujiHand2_Reorient_50Hz",
  env_cfg=_reorient_hand2_50hz_env_cfg(num_envs=8192),
  play_env_cfg=_reorient_hand2_50hz_env_cfg(play=True),
  rl_cfg=_reorient_hand2_50hz_rl_cfg(),
  runner_cls=WujiOnPolicyRunner,
)
