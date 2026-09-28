# SPDX-License-Identifier: Apache-2.0
"""Opt-in action-boundary diagnostics for controlled training investigations."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from mjlab.rl import RslRlVecEnvWrapper


class ActionDiagnosticsWrapper(RslRlVecEnvWrapper):
  """Observe raw/clipped actions without changing the upstream PPO samples.

  Emit one sample per rollout, reducing maxima across DDP ranks. Each rank also
  retains a JSONL record and up to five outlier inputs for later reproduction.
  Clipping itself remains the standard RslRlVecEnvWrapper implementation.
  """

  def __init__(self, env, clip_actions, *, rollout_steps: int, output_dir: Path):
    if clip_actions is None or clip_actions <= 0 or rollout_steps < 1:
      raise ValueError("diagnostics require a positive clip and rollout length")
    super().__init__(env, clip_actions=clip_actions)
    self._rollout_steps = rollout_steps
    self._output_dir = Path(output_dir)
    self._output_dir.mkdir(parents=True, exist_ok=True)
    # The runner initializes NCCL after constructing this environment wrapper.
    self._rank = (
      dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))
    )
    self._iteration = 0
    self._saved_outliers = 0
    self._step_stats: list[torch.Tensor] = []
    self._candidates: list[dict[str, torch.Tensor]] = []
    self._last_obs = None

  def get_observations(self):
    obs = super().get_observations()
    self._last_obs = obs
    return obs

  def step(self, actions):
    bounded = actions.clamp(-self.clip_actions, self.clip_actions)
    manager = self.unwrapped.action_manager
    rate = (
      (bounded - manager.action).square()
      + (bounded - 2 * manager.action + manager.prev_action).square()
    ).sum(-1)
    worst_id = actions.abs().amax(-1).argmax()
    candidate = {"env_id": worst_id.clone(), "raw_action": actions[worst_id].clone()}
    if self._last_obs is not None and "policy" in self._last_obs:
      candidate["policy_obs"] = self._last_obs["policy"][worst_id].clone()
    motion = self.unwrapped.command_manager.get_term("motion")
    candidate["clip_id"] = motion.trajectory_ids[worst_id].clone()
    candidate["frame"] = motion.time_steps[worst_id].clone()
    # Capture before stepping: autoreset can clear the action histories.
    stat = torch.stack(
      (
        actions.abs().max(),
        bounded.abs().max(),
        rate.max(),
        (actions.abs() > self.clip_actions).float().mean(),
      )
    )
    obs, rewards, dones, extras = super().step(actions)
    self._step_stats.append(torch.cat((stat, -rewards.min().reshape(1))))
    self._candidates.append(candidate)
    self._last_obs = obs
    if len(self._step_stats) == self._rollout_steps:
      self._flush(extras)
    return obs, rewards, dones, extras

  def _flush(self, extras):
    steps = torch.stack(self._step_stats)
    local = steps.amax(0)
    local[3] = steps[:, 3].mean()
    combined = local.clone()
    if dist.is_initialized():
      # Equal rollout lengths and environment counts on both ranks.
      fraction = combined[3].clone()
      dist.all_reduce(combined, op=dist.ReduceOp.MAX)
      dist.all_reduce(fraction, op=dist.ReduceOp.SUM)
      combined[3] = fraction / dist.get_world_size()
    combined[4] *= -1
    local[4] *= -1
    names = (
      "raw_action_max",
      "applied_action_max",
      "action_rate_combined_max",
      "clip_fraction",
      "reward_min",
    )
    extras["log"] = dict(extras.get("log", {}))
    extras["log"].update(
      {
        f"ActionBoundary/{name}": value.clone()
        for name, value in zip(names, combined, strict=True)
      }
    )
    values = local.cpu().tolist()
    record = {
      "iteration": self._iteration,
      "rank": self._rank,
      **dict(zip(names, values, strict=True)),
    }
    with (self._output_dir / f"rank{self._rank}.jsonl").open("a") as stream:
      stream.write(json.dumps(record) + "\n")
    if (
      values[0] > 100 or not all(map(math.isfinite, values))
    ) and self._saved_outliers < 5:
      torch.save(
        {
          "iteration": self._iteration,
          "stats": steps.cpu(),
          "inputs": [
            {k: v.detach().cpu() for k, v in row.items()} for row in self._candidates
          ],
        },
        self._output_dir / f"outlier_rank{self._rank}_iter{self._iteration}.pt",
      )
      self._saved_outliers += 1
    self._step_stats.clear()
    self._candidates.clear()
    self._iteration += 1
