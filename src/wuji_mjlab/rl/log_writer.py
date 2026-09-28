# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""W&B log writer that also archives the on-disk params/*.yaml."""

from __future__ import annotations

import os
import pathlib

import wandb
from rsl_rl.utils import WandbLogWriter

# rsl-rl suppresses these duplicate wall-clock metrics only when logger_type is
# literally "WandbLogWriter". A dotted plugin class name bypasses that guard,
# making wall-clock seconds compete with iteration indices as W&B global steps.
_WALL_CLOCK_TAGS = frozenset(
  {"Train/mean_reward/time", "Train/mean_episode_length/time"}
)


def require_wandb_project() -> str:
  """Read the W&B project after checking explicit environment configuration.

  Returns:
    The project named by WANDB_PROJECT.

  Raises:
    ValueError: WANDB_PROJECT or WANDB_API_KEY is missing or blank.
  """
  missing = [
    name
    for name in ("WANDB_PROJECT", "WANDB_API_KEY")
    if not os.environ.get(name, "").strip()
  ]
  if missing:
    raise ValueError(
      f"W&B logging requires environment variables: {', '.join(missing)}. "
      "Set them explicitly or use --agent.logger tensorboard."
    )
  return os.environ["WANDB_PROJECT"].strip()


class WujiWandbLogWriter(WandbLogWriter):
  def __init__(self, log_dir: str, project_name: str) -> None:
    project = require_wandb_project()
    if project_name != project:
      raise ValueError("W&B project_name must match WANDB_PROJECT")
    super().__init__(log_dir, project_name)
    self._log_dir = log_dir

  def add_scalar(
    self,
    tag: str,
    scalar_value: float,
    global_step: int | None = None,
    walltime: float | None = None,
    new_style: bool = False,
  ) -> None:
    if tag in _WALL_CLOCK_TAGS:
      return
    super().add_scalar(tag, scalar_value, global_step, walltime, new_style)

  def store_config(self, env_cfg, train_cfg: dict) -> None:
    super().store_config(env_cfg, train_cfg)
    # wandb.config's JSON conversion loses the observation-term order that
    # deploy uses for canonical concatenation. base_path also preserves the
    # downstream-required "params/" prefix instead of flattening the filenames.
    for name in ("env.yaml", "agent.yaml"):
      path = pathlib.Path(self._log_dir) / "params" / name
      if path.exists():
        wandb.save(str(path), base_path=self._log_dir, policy="now")
