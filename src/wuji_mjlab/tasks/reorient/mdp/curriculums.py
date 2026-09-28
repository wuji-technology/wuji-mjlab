# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import torch

from wuji_mjlab.tasks.reorient.mdp._env_utils import resolve_env_ids
from wuji_mjlab.utils.curriculum import get_curriculum_value  # noqa: F401 -- re-export

__all__ = [
  "adaptive_episode_curriculum",
  "get_curriculum_value",
  "get_linear_progress",
  "reorient_success_curriculum",
  "resolve_env_ids",
]


def get_linear_progress(
  env,
  warmup_frac: float = 0.05,
  rampup_frac: float = 0.60,
) -> float:
  if not hasattr(env, "max_common_steps"):
    raise ValueError(
      "Set env.max_common_steps from the training runner's "
      "max_iterations * num_steps_per_env before using the time ramp."
    )
  effective_total_steps = max(float(env.max_common_steps), 1.0)
  step = float(getattr(env, "common_step_counter", 0))
  warmup = float(warmup_frac) * effective_total_steps
  rampup = float(rampup_frac) * effective_total_steps
  if step <= warmup:
    return 0.0
  if step >= rampup:
    return 1.0
  return float((step - warmup) / max(rampup - warmup, 1.0))


def reorient_success_curriculum(
  env,
  env_ids,
  command_name: str = "reorient_command",
  count_threshold: int = 3,
  delta_per_loop: float = 0.01,
) -> dict[str, float]:
  if not hasattr(env, "_reorient_success_curriculum"):
    env._reorient_success_curriculum = 0.0

  env_ids = resolve_env_ids(env, env_ids)

  command = env.command_manager.get_term(command_name)
  goal_reach_count = command.metrics["goal_reach_count"][env_ids]
  if goal_reach_count.numel() == 0:
    return {"value": float(env._reorient_success_curriculum)}

  delta_per_env = float(delta_per_loop) / float(env.num_envs)
  delta = torch.where(
    goal_reach_count >= float(count_threshold),
    torch.full_like(goal_reach_count, delta_per_env, dtype=torch.float32),
    torch.full_like(goal_reach_count, -delta_per_env, dtype=torch.float32),
  )
  delta_sum = float(delta.sum().item())
  env._reorient_success_curriculum = float(
    max(min(env._reorient_success_curriculum + delta_sum, 1.0), 0.0)
  )
  return {
    "value": float(env._reorient_success_curriculum),
    "mean_goal_reach_count": float(goal_reach_count.float().mean().item()),
    "success_env_frac": float(
      (goal_reach_count >= float(count_threshold)).float().mean().item()
    ),
    "delta": delta_sum,
  }


def adaptive_episode_curriculum(
  env,
  env_ids,
  target_steps: int = 800,
  inc_rate: float = 0.1,
  dec_rate: float = 0.2,
  min_scale: float = 0.05,
  drop_gamma: float = 1.0,
) -> dict[str, float]:
  if not hasattr(env, "_adaptive_episode_curriculum"):
    env._adaptive_episode_curriculum = float(min_scale)

  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return {"value": float(env._adaptive_episode_curriculum)}

  target_steps = max(int(target_steps), 1)
  episode_steps = env.episode_length_buf[env_ids].float()
  progress = torch.clamp(episode_steps / float(target_steps), 0.0, 1.0)
  reached_target = episode_steps >= float(target_steps)

  terminated = getattr(env.termination_manager, "terminated", None)
  if terminated is None:
    terminated = torch.zeros_like(reached_target)
  else:
    terminated = terminated[env_ids]
  early_drop = terminated & ~reached_target

  inc = reached_target.float() * float(inc_rate)
  pen = (
    early_drop.float() * torch.pow(1.0 - progress, float(drop_gamma)) * float(dec_rate)
  )

  delta = float((inc - pen).sum().item() / float(env.num_envs))
  env._adaptive_episode_curriculum = float(
    max(min(env._adaptive_episode_curriculum + delta, 1.0), float(min_scale))
  )
  return {
    "value": float(env._adaptive_episode_curriculum),
    "mean_episode_steps": float(episode_steps.mean().item()),
    "mean_progress": float(progress.mean().item()),
    "reached_target_frac": float(reached_target.float().mean().item()),
    "early_drop_frac": float(early_drop.float().mean().item()),
    "target_steps": float(target_steps),
    "delta": delta,
  }
