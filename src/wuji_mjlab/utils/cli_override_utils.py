# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Shared thin CLI helpers for task wrappers."""

from __future__ import annotations

DEFAULT_TASK_ALIASES = {
  "reorient": "WujiHand_Reorient",
  "reorient/reorient": "WujiHand_Reorient",
  "revo3": "Revo3RightHand_Reorient",
  "revo3/reorient": "Revo3RightHand_Reorient",
}


def resolve_task(
  task_name: str,
  *,
  task_aliases: dict[str, str] | None = None,
  all_tasks: list[str] | tuple[str, ...],
) -> str:
  aliases = DEFAULT_TASK_ALIASES if task_aliases is None else task_aliases
  resolved = aliases.get(task_name, task_name)
  if resolved not in all_tasks:
    raise ValueError(f"Unknown task '{task_name}'. Available tasks: {list(all_tasks)}")
  return resolved


def split_special_args(args: list[str]) -> tuple[str | None, list[str], list[str]]:
  task_name: str | None = None
  passthrough: list[str] = []
  overrides: list[str] = []

  for arg in args:
    if arg.startswith("task="):
      task_name = arg.split("=", 1)[1]
      continue
    if "=" in arg and (arg.startswith("env.") or arg.startswith("agent.")):
      overrides.append(arg)
      continue
    passthrough.append(arg)

  return task_name, passthrough, overrides
