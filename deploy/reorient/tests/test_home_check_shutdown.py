# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from wuji_reorient_deploy.constants import load_constants
from wuji_reorient_deploy.drivers.base import MockHandDriver


def _load_home_check():
  path = Path(__file__).parents[1] / "tools" / "home_check.py"
  spec = importlib.util.spec_from_file_location("home_check_shutdown_test", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_home_failure_propagates_after_closing_driver_once(monkeypatch):
  home_check = _load_home_check()

  class FailingHome(MockHandDriver):
    def __init__(self):
      super().__init__(load_constants(2, "right"), "right")
      self.close_calls = 0

    def home(self, *args, **kwargs):
      raise RuntimeError("simulated home failure (joint jam)")

    def close(self):
      self.close_calls += 1

  driver = FailingHome()
  monkeypatch.setattr(home_check, "make_driver", lambda *args, **kwargs: driver)
  monkeypatch.setattr(sys, "argv", ["home_check.py", "--hand", "right", "--mock"])

  with pytest.raises(RuntimeError, match="simulated home failure"):
    home_check.main()

  assert driver.close_calls == 1
