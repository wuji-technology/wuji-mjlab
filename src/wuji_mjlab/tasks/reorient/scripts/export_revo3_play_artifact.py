# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""CLI for exporting and validating the actual Revo3 216-D play artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from wuji_mjlab.tasks.reorient.tooling.revo3_play_artifact import (
  add_export_args,
  export_artifact,
  options_from_args,
  validate_artifact_dir,
)


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Export/validate Revo3 raw-obs 216-D policy artifacts."
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  export_parser = subparsers.add_parser("export", help="Export artifact directory.")
  add_export_args(export_parser)

  validate_parser = subparsers.add_parser("validate", help="Validate artifact directory.")
  validate_parser.add_argument("--artifact-dir", required=True, type=Path)

  args = parser.parse_args()
  if args.command == "export":
    artifact_dir = export_artifact(options_from_args(args))
    print(f"Exported artifact: {artifact_dir}")
    return
  if args.command == "validate":
    validate_artifact_dir(args.artifact_dir)
    print(f"Validation passed: {args.artifact_dir}")
    return
  raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
  main()
