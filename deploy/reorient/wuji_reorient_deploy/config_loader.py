# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Read deploy ports and hardware knobs from config/control.yaml."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONTROL_YAML = os.path.join(_ROOT, "config", "control.yaml")


@lru_cache(maxsize=1)
def _cfg() -> dict[str, Any]:
  if not os.path.exists(_CONTROL_YAML):
    return {}
  with open(_CONTROL_YAML) as f:
    return yaml.safe_load(f) or {}


def _zmq(key: str, default: int) -> int:
  return int(_cfg().get("zmq", {}).get(key, default))


def cube_port() -> int:
  """Return the configured ZeroMQ cube-pose port."""
  return _zmq("cube_port", 5555)


def goal_port() -> int:
  """Return the configured ZeroMQ goal-orientation port."""
  return _zmq("goal_port", 5556)


def joint_port() -> int:
  """Return the configured ZeroMQ joint-position port."""
  return _zmq("joint_port", 5557)


def serial_number(hand_side: str) -> str:
  """Return the configured serial number for a hand."""
  return str(_cfg().get("serial_numbers", {}).get(hand_side, "") or "")


def control() -> dict[str, Any]:
  """Return configured control-loop parameters."""
  return dict(_cfg().get("control", {}) or {})


def goal() -> dict[str, Any]:
  """Return configured goal-detection parameters."""
  return dict(_cfg().get("goal", {}) or {})
