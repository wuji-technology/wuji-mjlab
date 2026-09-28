# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

_RUN_POLICY = Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py"


def _load_run_policy():
  spec = importlib.util.spec_from_file_location("run_policy_vision_guard", _RUN_POLICY)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


rp = _load_run_policy()


class _Driver:
  def __init__(self, default_q: np.ndarray, fault_after: int | None = None):
    self.q = default_q.copy()
    self.fault_after = fault_after
    self.fault_calls = 0
    self.close_calls = 0

  def write_target(self, qpos: np.ndarray) -> None:
    self.q = np.asarray(qpos, dtype=np.float64).copy()

  def read_encoders(self) -> np.ndarray:
    return self.q.copy()

  def home(self) -> None:
    pass

  def fault(self) -> str | None:
    self.fault_calls += 1
    if self.fault_after is not None and self.fault_calls >= self.fault_after:
      return "test stop"
    return None

  def temp_summary(self) -> None:
    return None

  def close(self) -> None:
    self.close_calls += 1


class _Policy:
  def __init__(self, action_dim: int):
    self.config = {
      "action_scale": 0.5,
      "ema_alpha": 0.5,
      "warmup_time_s": 0.0,
      "ctrl_dt": 0.02,
    }
    self.action_dim = action_dim
    self.calls = 0

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    self.calls += 1
    return np.zeros(self.action_dim, dtype=np.float64)


class _Publisher:
  def publish(self, *args) -> None:
    pass


class _CubeReceiver:
  def __init__(self, *, fix: bool, age: float | None):
    self.fix = fix
    self.age = age
    self.wait_calls = 0
    self.latest_calls = 0
    self.count = 0

  def wait_for_fix(self, timeout_s: float) -> bool:
    self.wait_calls += 1
    return self.fix

  def age_s(self) -> float | None:
    return self.age

  def latest(self) -> tuple[np.ndarray, np.ndarray]:
    self.latest_calls += 1
    return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])


def _install_runtime(monkeypatch, *, fix: bool, age: float | None,
                     fault_after: int | None = None):
  const = rp.load_constants(2, "right")
  driver = _Driver(const.default_joint_pos, fault_after=fault_after)
  policy = _Policy(const.action_dim)
  cube = _CubeReceiver(fix=fix, age=age)

  monkeypatch.setattr(rp, "make_driver", lambda *args, **kwargs: driver)
  monkeypatch.setattr(rp, "ONNXPolicy", lambda *args, **kwargs: policy)
  monkeypatch.setattr(rp, "GoalPublisher", _Publisher)
  monkeypatch.setattr(rp, "JointPublisher", _Publisher)
  monkeypatch.setattr(rp, "CubeReceiver", lambda: cube)
  return driver, policy, cube


def _argv(monkeypatch, *extra: str) -> None:
  monkeypatch.setattr(
    sys,
    "argv",
    ["run_policy.py", "--gen", "2", "--hand", "right", "--mock", *extra],
  )


def test_initial_fix_timeout_exits_through_shutdown(monkeypatch):
  driver, policy, cube = _install_runtime(monkeypatch, fix=False, age=None)
  _argv(monkeypatch, "--cube-wait", "0")

  with pytest.raises(SystemExit):
    rp.main()

  assert cube.wait_calls == 1
  assert cube.latest_calls == 0
  assert policy.calls == 0
  assert driver.close_calls == 1


@pytest.mark.parametrize("age", [1.0, None])
def test_stale_or_never_received_pose_stops_before_inference(monkeypatch, age):
  driver, policy, cube = _install_runtime(
    monkeypatch, fix=True, age=age, fault_after=2,
  )
  _argv(monkeypatch, "--cube-timeout", "0.5")

  rp.main()

  assert cube.wait_calls == 1
  assert driver.fault_calls == 1
  assert policy.calls == 0
  assert driver.close_calls == 1


def test_zero_cube_timeout_disables_stale_pose_stop(monkeypatch):
  driver, policy, cube = _install_runtime(
    monkeypatch, fix=True, age=1000.0, fault_after=3,
  )
  _argv(monkeypatch, "--cube-timeout", "0")

  rp.main()

  assert cube.wait_calls == 1
  assert driver.fault_calls == 3
  assert policy.calls == 2
  assert driver.close_calls == 1


def test_servo_targets_keep_interpolation_without_substep_encoder_reads(monkeypatch):
  driver, policy, _ = _install_runtime(
    monkeypatch, fix=True, age=0.0, fault_after=3,
  )
  initial = driver.q.copy()
  targets = []
  reads_after_targets = []
  displayed_encoders = []
  write_target = driver.write_target
  read_encoders = driver.read_encoders

  def write(qpos):
    targets.append(qpos.copy())
    write_target(qpos)

  def read():
    reads_after_targets.append(len(targets))
    return read_encoders()

  def infer(self, obs):
    self.calls += 1
    return np.full(self.action_dim, 0.1 * self.calls)

  class JointPublisher:
    def publish(self, target, encoders):
      displayed_encoders.append(encoders.copy())

  monkeypatch.setattr(driver, "write_target", write)
  monkeypatch.setattr(driver, "read_encoders", read)
  monkeypatch.setattr(_Policy, "__call__", infer)
  monkeypatch.setattr(rp, "JointPublisher", JointPublisher)
  monkeypatch.setattr(rp.time, "sleep", lambda seconds: None)
  _argv(monkeypatch, "--servo-hz", "100", "--ema-alpha", "1", "--action-scale", "1")

  rp.main()

  const = rp.load_constants(2, "right")
  first = np.clip(initial + 0.1, const.soft_lower, const.soft_upper)
  second = np.clip(initial + 0.2, const.soft_lower, const.soft_upper)
  np.testing.assert_allclose(targets, [
    (initial + first) / 2, first, (first + second) / 2, second,
  ])
  assert reads_after_targets == [0, 2, 4]
  np.testing.assert_allclose(displayed_encoders, [initial, initial, first, first])
  assert policy.calls == 2
  assert driver.close_calls == 1
