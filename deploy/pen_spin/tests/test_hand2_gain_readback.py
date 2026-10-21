# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Hand 2 gain and effort read-back gate the motor enable."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from lib.hand2_adapter import Hand2Compat


class _Setting:
    def __init__(self, readback=None):
        self.value = None
        self._readback = readback

    def get(self):
        return self._readback(self.value) if self._readback else self.value

    def set(self, value):
        self.value = value


class _Subscription:
    def close(self):
        pass


class _StateFeed:
    def subscribe_with_callback(self, callback):
        callback(SimpleNamespace(joints=[
            SimpleNamespace(nid=finger * 5 + joint + 1, position=0.0)
            for finger in range(5)
            for joint in range(4)
        ]))
        return _Subscription()


class _Publisher:
    def send(self, command):
        pass

    def close(self):
        pass


class _CommandFeed:
    def publish(self):
        return _Publisher()


class _FakeHand:
    def __init__(self, mit_readback=None, effort_readback=None):
        self.serial_number = "FAKE-SN"
        self.enable_calls = 0
        self._online = _Setting()
        self._online.set(20)
        self._effort_limit = _Setting(effort_readback)
        self._mit_params = _Setting(mit_readback)

    def online_joints_count(self):
        return self._online

    def effort_limit(self):
        return self._effort_limit

    def mit_params(self):
        return self._mit_params

    def enable(self):
        self.enable_calls += 1

    def disable(self):
        pass

    def joint_command(self):
        return _CommandFeed()

    def joint_states(self):
        return _StateFeed()


class _FakeManager:
    def __init__(self, hand):
        self.hand = hand

    def connect(self, *, sn, device_name, options):
        return self.hand

    def disconnect_all(self):
        pass


def _install_fake_sdk(monkeypatch, hand):
    sdk = ModuleType("wuji_sdk")
    sdk.SdkManager = SimpleNamespace(instance=lambda: _FakeManager(hand))
    sdk.ConnectOptions = lambda *, enable_bridge: SimpleNamespace()
    sdk.MitParam = lambda kp, kd: SimpleNamespace(kp=kp, kd=kd)
    sdk.JointCommand = lambda position, velocity, effort: (position, velocity, effort)
    monkeypatch.setitem(sys.modules, "wuji_sdk", sdk)


def _construct(**overrides):
    return Hand2Compat(serial_number="FAKE-SN", kp=20.0, kd=0.1, **overrides)


def _kd_drift(params):
    back = list(params)
    back[1] = SimpleNamespace(kp=back[1].kp, kd=0.05)
    return back


def _kp_offline(params):
    back = list(params)
    back[17] = None
    return back


def _effort_drift(limits):
    back = list(limits)
    back[2] = 0.4
    return back


@pytest.mark.parametrize(
    ("overrides", "hand_kwargs", "joint"),
    [
        ({"kd_overrides": {1: 0.075}}, {"mit_readback": _kd_drift},
         "finger1_joint2"),
        ({"kp_overrides": {17: 15.0}}, {"mit_readback": _kp_offline},
         "finger5_joint2"),
        ({"effort_overrides": {2: 0.8}}, {"effort_readback": _effort_drift},
         "finger1_joint3"),
    ],
)
def test_readback_mismatch_refuses_to_enable(
    monkeypatch, overrides, hand_kwargs, joint
):
    hand = _FakeHand(**hand_kwargs)
    _install_fake_sdk(monkeypatch, hand)
    adapter = _construct(**overrides)
    try:
        with pytest.raises(RuntimeError, match=joint):
            adapter.write_joint_enabled(True)
        assert hand.enable_calls == 0
    finally:
        adapter.close()


def test_matching_readback_enables_once(monkeypatch):
    hand = _FakeHand()
    _install_fake_sdk(monkeypatch, hand)
    adapter = _construct(
        kp_overrides={17: 15.0},
        kd_overrides={1: 0.075, 2: 0.07},
        effort_overrides={2: 0.8},
    )
    try:
        adapter.write_joint_enabled(True)
        assert hand.enable_calls == 1
    finally:
        adapter.close()
