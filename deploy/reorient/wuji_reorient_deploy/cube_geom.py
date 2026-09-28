# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Shared helpers for cube geometry: cube_tags JSON resolution."""

from __future__ import annotations

import json
import os
from typing import Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CUBE_CONFIG_FILE = os.path.join(ROOT_DIR, "config", "cube_tags.json")


def resolve_cube_config_path(arg: str | None) -> str:
  """Resolve a ``--cube`` CLI argument to an absolute cube_tags JSON path."""
  if not arg or arg == "default":
    if not os.path.exists(DEFAULT_CUBE_CONFIG_FILE):
      raise FileNotFoundError(
        f"default cube config missing: {DEFAULT_CUBE_CONFIG_FILE!r}"
      )
    return DEFAULT_CUBE_CONFIG_FILE
  if os.path.exists(arg):
    return os.path.abspath(arg)
  raise FileNotFoundError(f"--cube={arg!r}: path does not exist.")


def load_cube_config(path: str) -> dict[str, Any]:
  """Load and parse a cube_tags JSON file."""
  with open(path) as f:
    return json.load(f)
