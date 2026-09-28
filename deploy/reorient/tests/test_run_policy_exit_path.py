# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_RUN_POLICY = Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py"


def _load_run_policy():
  spec = importlib.util.spec_from_file_location("run_policy", _RUN_POLICY)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


rp = _load_run_policy()


class _Driver:
  def __init__(self, close_error: BaseException | None = None):
    self.close_calls = 0
    self._close_error = close_error

  def close(self):
    self.close_calls += 1
    if self._close_error is not None:
      raise self._close_error


def _record_with_tick():
  return {"t": [0.1], "target": [[0.0, 1.0]]}


@pytest.mark.parametrize("save_error", [OSError("disk full"), KeyboardInterrupt()])
def test_log_save_failure_cannot_skip_or_escape_close(monkeypatch, save_error):
  driver = _Driver()

  def fail_save(*args, **kwargs):
    raise save_error

  monkeypatch.setattr(rp.np, "savez", fail_save)
  rp._shutdown(driver, _record_with_tick(), "/tmp/run-policy-log.npz")

  assert driver.close_calls == 1


def test_close_failure_does_not_escape_or_prevent_log_save(monkeypatch):
  driver = _Driver(RuntimeError("disable failed"))
  saved_paths = []

  def save(path, **arrays):
    saved_paths.append(path)

  monkeypatch.setattr(rp.np, "savez", save)
  rp._shutdown(driver, _record_with_tick(), "/tmp/run-policy-log.npz")

  assert driver.close_calls == 1
  assert saved_paths == ["/tmp/run-policy-log.npz"]


@pytest.mark.parametrize("rec", [None, {"t": []}])
def test_no_records_skips_log_write_but_still_closes(monkeypatch, rec):
  driver = _Driver()

  def unexpected_save(*args, **kwargs):
    raise AssertionError("np.savez must not be called without recorded ticks")

  monkeypatch.setattr(rp.np, "savez", unexpected_save)
  rp._shutdown(driver, rec, "/tmp/run-policy-log.npz")

  assert driver.close_calls == 1
