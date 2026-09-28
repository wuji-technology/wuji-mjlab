# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Thin play wrapper with IsaacLab-style env/agent overrides."""

from __future__ import annotations

import argparse
import sys

import mjlab
import tyro
import wuji_mjlab.tasks  # noqa: F401
from mjlab.scripts.play import PlayConfig
from mjlab.tasks.registry import list_tasks
from wuji_mjlab.utils.cli_override_utils import (
  DEFAULT_TASK_ALIASES,
  resolve_task,
  split_special_args,
)
from wuji_mjlab.utils.play_runner import run_play_with_cfg
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs
from wuji_mjlab.utils.train_config_utils import load_train_config


def run_play_entry(task_id: str, cfg: PlayConfig, overrides: list[str]) -> None:
  env_cfg, agent_cfg = prepare_task_cfgs(task_id, overrides, play=True)
  run_play_with_cfg(task_id, cfg, env_cfg, agent_cfg)


def main() -> None:
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--task", type=str, default=None)
  parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Optional YAML whose task selection is used in play mode "
    "(schema: wuji_mjlab.utils.train_config_utils). Prefer --task.",
  )
  known, remaining = parser.parse_known_args()

  config_task: str | None = None
  if known.config is not None:
    config_task, _, _ = load_train_config(known.config)

  task_from_equals, tyro_args, overrides = split_special_args(remaining)
  chosen_task = known.task or task_from_equals or config_task or "reorient"
  task_id = resolve_task(
    chosen_task,
    task_aliases=DEFAULT_TASK_ALIASES,
    all_tasks=list_tasks(),
  )

  cfg = tyro.cli(
    PlayConfig,
    args=tyro_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" --task {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )

  run_play_entry(task_id, cfg, overrides)


if __name__ == "__main__":
  main()
