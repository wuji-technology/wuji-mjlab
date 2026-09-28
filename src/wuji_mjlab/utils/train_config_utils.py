# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""YAML training-config support for the train/play CLI wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _flatten_overrides(prefix: str, tree: Any, out: list[str]) -> None:
  if not isinstance(tree, dict):
    value = tree if isinstance(tree, str) else repr(tree)
    out.append(f"{prefix}={value}")
    return
  for key, sub in tree.items():
    _flatten_overrides(f"{prefix}.{key}", sub, out)


def load_train_config(path: str | Path) -> tuple[str | None, list[str], list[str]]:
  """Load a YAML training config."""
  path = Path(path)
  data = yaml.safe_load(path.read_text()) or {}
  if not isinstance(data, dict):
    raise ValueError(f"Training config {path} must be a YAML mapping.")

  known_keys = {"task", "env", "agent", "args"}
  unknown = set(data) - known_keys
  if unknown:
    raise ValueError(
      f"Unknown keys {sorted(unknown)} in training config {path}. "
      f"Supported keys: {sorted(known_keys)}."
    )

  task = data.get("task")
  overrides: list[str] = []
  for root in ("env", "agent"):
    if root in data and data[root] is not None:
      _flatten_overrides(root, data[root], overrides)

  extra_args = [str(a) for a in (data.get("args") or [])]
  return task, overrides, extra_args
