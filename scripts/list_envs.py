# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""List registered Wuji MJLab tasks."""

from __future__ import annotations

import argparse

from prettytable import PrettyTable

import wuji_mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--all",
    action="store_true",
    help="Include upstream MJLab example tasks",
  )
  args = parser.parse_args()

  table = PrettyTable(["#", "Task ID"])
  table.title = (
    "All Registered MJLab Tasks" if args.all else "Available Wuji MJLab Tasks"
  )
  table.align["Task ID"] = "l"

  task_ids = list_tasks()
  if not args.all:
    task_ids = [task_id for task_id in task_ids if task_id.startswith("Wuji")]
  for idx, task_id in enumerate(task_ids, start=1):
    table.add_row([idx, task_id])

  print(table)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
