# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from wuji_reorient_deploy import drivers as drivers_module
from wuji_reorient_deploy.constants import load_constants
from wuji_reorient_deploy.drivers import hand2 as hand2_module
from wuji_reorient_deploy.drivers.hand2 import WujiHand2Driver


_ALL_NIDS = tuple(
  finger * 5 + joint + 1
  for finger in range(5)
  for joint in range(4)
)


def _load_check_setup():
  path = Path(__file__).parents[1] / "tools" / "check_setup.py"
  spec = importlib.util.spec_from_file_location("hand2_feedback_check_setup_test", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class _Clock:
  def __init__(self):
    self.now = 100.0

  def monotonic(self):
    return self.now

  def sleep(self, seconds):
    self.now += seconds

  def advance(self, seconds):
    self.now += seconds


class _Setting:
  def __init__(self, value=None):
    self.value = value

  def get(self):
    return self.value

  def set(self, value):
    self.value = value


class _Subscription:
  def __init__(self):
    self.closed = False

  def close(self):
    self.closed = True


class _CallbackFeed:
  def __init__(self, joints):
    self.joints = joints

  def subscribe_with_callback(self, callback):
    callback(SimpleNamespace(joints=self.joints))
    return _Subscription()


class _Publisher:
  def __init__(self):
    self.sent = []
    self.closed = False

  def send(self, command):
    self.sent.append(command)

  def close(self):
    self.closed = True


class _CommandFeed:
  def __init__(self, publisher):
    self.publisher = publisher

  def publish(self):
    return self.publisher


class _FakeHand:
  def __init__(self, online_count, state_nids):
    self.serial_number = "FAKE-SN"
    self.enable_calls = 0
    self.disable_calls = 0
    self._online = _Setting(online_count)
    self._effort_limit = _Setting()
    self._mit_params = _Setting()
    self._publisher = _Publisher()
    self._states = _CallbackFeed([
      SimpleNamespace(nid=nid, position=0.01 * nid)
      for nid in state_nids
    ])
    self._diagnostics = _CallbackFeed([])

  def online_joints_count(self):
    return self._online

  def effort_limit(self):
    return self._effort_limit

  def mit_params(self):
    return self._mit_params

  def enable(self):
    self.enable_calls += 1

  def disable(self):
    self.disable_calls += 1

  def joint_command(self):
    return _CommandFeed(self._publisher)

  def joint_states(self):
    return self._states

  def joint_diagnostics(self):
    return self._diagnostics

  def describe_error(self, code):
    return {"name": f"error-{code}", "severity": "error"}


class _FakeManager:
  def __init__(self, hand):
    self.hand = hand
    self.disconnect_calls = 0

  def auto_connect(self, *, device_name):
    return self.hand

  def connect(self, *, sn, device_name, options):
    return self.hand

  def disconnect_all(self):
    self.disconnect_calls += 1


class _ConnectOptions:
  def __init__(self, *, enable_bridge):
    self.enable_bridge = enable_bridge


class _JointCommand:
  def __init__(self, position, velocity, effort):
    self.position = position
    self.velocity = velocity
    self.effort = effort


def _install_fake_sdk(monkeypatch, *, online_count=20, state_nids=_ALL_NIDS):
  hand = _FakeHand(online_count, state_nids)
  manager = _FakeManager(hand)
  sdk = ModuleType("wuji_sdk")
  sdk.SdkManager = SimpleNamespace(instance=lambda: manager)
  sdk.ConnectOptions = _ConnectOptions
  sdk.JointCommand = _JointCommand
  clock = _Clock()
  monkeypatch.setitem(sys.modules, "wuji_sdk", sdk)
  monkeypatch.setattr(hand2_module, "time", clock)
  return hand, clock


def _construct_driver(*, energize=True):
  return WujiHand2Driver(
    load_constants(2, "right"),
    "right",
    energize=energize,
  )


def test_rejects_incomplete_online_joint_count(monkeypatch):
  _install_fake_sdk(monkeypatch, online_count=1, state_nids=(1,))

  with pytest.raises(RuntimeError, match="1/20"):
    _construct_driver()


def test_constructor_reports_specific_missing_joint_feedback(monkeypatch):
  _install_fake_sdk(monkeypatch, state_nids=(1,))

  with pytest.raises(RuntimeError, match="right_finger1_joint2"):
    _construct_driver()


def test_all_joint_feedback_is_complete(monkeypatch):
  _install_fake_sdk(monkeypatch)
  driver = _construct_driver()
  try:
    assert driver.missing_feedback() == ()
  finally:
    driver.close()


def test_stale_per_joint_feedback_becomes_fault(monkeypatch):
  _, clock = _install_fake_sdk(monkeypatch)
  driver = _construct_driver()
  try:
    clock.advance(0.51)
    assert driver.missing_feedback(max_age_s=1e-6)
    assert driver.fault() is not None
  finally:
    driver.close()


def test_default_driver_energizes_motors(monkeypatch):
  hand, _ = _install_fake_sdk(monkeypatch)
  driver = _construct_driver()
  try:
    assert hand.enable_calls == 1
  finally:
    driver.close()


def test_check_setup_uses_read_only_hand2_connection(monkeypatch):
  hand, _ = _install_fake_sdk(monkeypatch)

  check = _load_check_setup()._check_connect(2, "right", None)

  assert check.ok
  assert hand.enable_calls == 0
  assert "motors NOT energized (read-only connect)" in check.detail


def test_check_setup_treats_undeclared_driver_as_energized(monkeypatch):
  const = load_constants(2, "right")

  class DriverWithoutEnergizedFlag:
    def read_encoders(self):
      return const.default_joint_pos.copy()

    def fault(self):
      return None

    def close(self):
      pass

  monkeypatch.setattr(
    drivers_module,
    "make_driver",
    lambda *args, **kwargs: DriverWithoutEnergizedFlag(),
  )

  check = _load_check_setup()._check_connect(2, "right", None)

  assert check.ok
  assert "motors WERE energized" in check.detail


def _install_fake_hand1_sdk(
  monkeypatch, *, fail_read=False, fail_open=False, on_enable=None,
  on_disable=None, on_close=None,
):
  hand = SimpleNamespace(enable_calls=0, disable_calls=0, close_calls=0,
                         controller_active=False, events=[],
                         warning_before_enable=False)

  class FakeController:
    def __enter__(self):
      if isinstance(fail_open, BaseException):
        raise fail_open
      if fail_open:
        raise RuntimeError("controller unavailable")
      hand.controller_active = True
      return SimpleNamespace()

    def __exit__(self, exc_type, exc_value, traceback):
      hand.close_calls += 1
      hand.controller_active = False
      hand.events.append("close")
      if on_close is not None:
        on_close()

  class FakeHand:
    def write_joint_effort_limit(self, limit):
      pass

    def write_joint_enabled(self, enabled):
      if enabled:
        hand.enable_calls += 1
        if on_enable is not None:
          hand.warning_before_enable = on_enable()
      else:
        hand.disable_calls += 1
        hand.events.append("disable")
        if on_disable is not None:
          on_disable()

    def realtime_controller(self, **kwargs):
      return FakeController()

    def read_joint_actual_position(self):
      if fail_read:
        raise RuntimeError("feedback unavailable")
      return load_constants(1, "right").default_joint_pos.reshape(5, 4)

  sdk = ModuleType("wujihandpy")
  sdk.Hand = FakeHand
  sdk.filter = SimpleNamespace(LowPass=lambda **kwargs: None)
  monkeypatch.setitem(sys.modules, "wujihandpy", sdk)
  return hand


def test_check_setup_gen1_refuses_connection_without_energize_consent(monkeypatch):
  hand = _install_fake_hand1_sdk(monkeypatch)

  check = _load_check_setup()._check_connect(1, "right", None)

  assert not check.ok
  assert hand.enable_calls == 0


def test_check_setup_gen1_opt_in_energizes_and_disables(monkeypatch, capsys):
  hand = _install_fake_hand1_sdk(
    monkeypatch, on_enable=lambda: "WARNING" in capsys.readouterr().err)

  check = _load_check_setup()._check_connect(1, "right", None, allow_energize=True)

  assert check.ok
  assert hand.enable_calls == 1
  assert hand.warning_before_enable
  assert hand.disable_calls == 1
  assert hand.close_calls == 1


def test_check_setup_gen1_read_failure_still_disables_and_closes(monkeypatch):
  hand = _install_fake_hand1_sdk(monkeypatch, fail_read=True)

  check = _load_check_setup()._check_connect(1, "right", None, allow_energize=True)

  assert not check.ok
  assert hand.enable_calls == 1
  assert hand.disable_calls == 1
  assert hand.close_calls == 1


def test_check_setup_gen1_controller_failure_still_disables_and_closes(monkeypatch):
  hand = _install_fake_hand1_sdk(monkeypatch, fail_open=True)

  check = _load_check_setup()._check_connect(1, "right", None, allow_energize=True)

  assert not check.ok
  assert hand.enable_calls == 1
  assert hand.disable_calls == 1
  assert hand.close_calls == 1


@pytest.mark.parametrize("error", [RuntimeError("enable failed"), KeyboardInterrupt()])
def test_hand1_enable_failure_disables_without_leaving_controller(monkeypatch, error):
  def fail_enable():
    raise error

  hand = _install_fake_hand1_sdk(monkeypatch, on_enable=fail_enable)

  from wuji_reorient_deploy.drivers.hand1 import WujiHand1Driver

  with pytest.raises(BaseException) as caught:
    WujiHand1Driver(load_constants(1, "right"))

  assert caught.value is error
  assert hand.enable_calls == 1
  assert hand.disable_calls == 1
  assert not hand.controller_active


def test_hand1_cleanup_interrupt_keeps_original_error_and_runs_each_step(monkeypatch):
  original = RuntimeError("controller unavailable")

  def interrupt():
    raise SystemExit("interrupted cleanup")

  hand = _install_fake_hand1_sdk(
    monkeypatch, fail_open=original, on_disable=interrupt, on_close=interrupt
  )

  from wuji_reorient_deploy.drivers.hand1 import WujiHand1Driver

  with pytest.raises(RuntimeError) as caught:
    WujiHand1Driver(load_constants(1, "right"))

  assert caught.value is original
  assert hand.events == ["close", "disable"]
  assert not hand.controller_active
