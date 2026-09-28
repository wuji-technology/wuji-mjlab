# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Locate the Hikvision MVS SDK Python bindings (``MvImport``)."""

from __future__ import annotations

import os
import sys

DEFAULT_MVS_PYTHON_PATH = "/opt/MVS/Samples/64/Python"

_INSTALL_HINT = (
  "install Hikvision MVS SDK (https://www.hikrobotics.com) or set "
  "MVS_PYTHON_PATH to the directory containing MvImport/"
)


def mvs_python_path() -> str:
  """Return the configured MVS python-bindings directory."""
  return os.environ.get("MVS_PYTHON_PATH", DEFAULT_MVS_PYTHON_PATH)


def find_mvs_python_path() -> str | None:
  """Return the MVS python-bindings directory if it actually holds MvImport/."""
  path = mvs_python_path()
  return path if os.path.isdir(os.path.join(path, "MvImport")) else None


def ensure_mvs_importable() -> str:
  """Put the MVS bindings on ``sys.path`` so ``import MvImport`` resolves."""
  path = find_mvs_python_path()
  if path is None:
    raise RuntimeError(
      f"MvImport not found at {mvs_python_path()}/MvImport. {_INSTALL_HINT}."
    )
  if path not in sys.path:
    sys.path.insert(0, path)
  return path
