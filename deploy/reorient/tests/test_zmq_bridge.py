# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import numpy as np
import pytest
import zmq
from wuji_reorient_deploy import config_loader
from wuji_reorient_deploy import zmq_bridge as bridge


def _stop_receiver(receiver) -> None:
  receiver.close()
  receiver._t.join(timeout=0.5)
  receiver._sock.close(0)


def test_cube_freshness_tracks_only_valid_world_fixes(monkeypatch):
  monkeypatch.setattr(zmq.Socket, "connect", lambda self, endpoint: None)
  receiver = bridge.CubeReceiver(port=65431)
  try:
    assert receiver.age_s() is None
    assert receiver.wait_for_fix(0.05) is False

    now = [100.0]
    monkeypatch.setattr(bridge.time, "monotonic", lambda: now[0])
    valid = {
      "world_fixed": True,
      "cube1": {
        "position": {"x": 1.0, "y": 2.0, "z": 3.0},
        "orientation": {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9},
      },
    }
    with receiver._lock:
      receiver._on_msg(valid)

    now[0] = 100.1
    age = receiver.age_s()
    assert isinstance(age, float)
    assert age < 0.5
    assert receiver.count == 1
    assert receiver.wait_for_fix(0.0) is True

    invalid = {
      "world_fixed": False,
      "cube1": {
        "position": {"x": 9.0, "y": 9.0, "z": 9.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
      },
    }
    now[0] = 100.2
    with receiver._lock:
      receiver._on_msg(invalid)

    assert receiver.age_s() == pytest.approx(0.2)
    assert receiver.count == 1
  finally:
    _stop_receiver(receiver)


def test_joint_publisher_emits_optional_encoder_measurements(monkeypatch):
  sent = []
  publisher = bridge.JointPublisher.__new__(bridge.JointPublisher)
  monkeypatch.setattr(publisher, "_send", sent.append)

  publisher.publish(np.array([1.0, 2.0]))
  publisher.publish(np.array([3.0, 4.0]), np.array([3.1, 4.1]))

  assert sent == [
    {"joint_pos": [1.0, 2.0]},
    {"joint_pos": [3.0, 4.0], "joint_enc": [3.1, 4.1]},
  ]


def test_joint_receiver_supports_old_and_measured_messages(monkeypatch):
  monkeypatch.setattr(zmq.Socket, "connect", lambda self, endpoint: None)
  receiver = bridge.JointReceiver(port=65432)
  try:
    with receiver._lock:
      receiver._on_msg({"joint_pos": [1.0, 2.0]})
    np.testing.assert_allclose(receiver.latest(), [1.0, 2.0])
    assert receiver.latest_measured() is None

    with receiver._lock:
      receiver._on_msg({"joint_pos": [3.0, 4.0], "joint_enc": [3.1, 4.1]})
    np.testing.assert_allclose(receiver.latest(), [3.0, 4.0])
    np.testing.assert_allclose(receiver.latest_measured(), [3.1, 4.1])
  finally:
    _stop_receiver(receiver)


def test_default_ports_follow_control_config(monkeypatch):
  config_loader._cfg.cache_clear()
  monkeypatch.setattr(
    config_loader,
    "_cfg",
    lambda: {
      "zmq": {"cube_port": 5655, "goal_port": 5656, "joint_port": 5657}
    },
  )

  connected = []
  bound = []
  monkeypatch.setattr(
    zmq.Socket, "connect", lambda self, endpoint: connected.append(endpoint)
  )
  monkeypatch.setattr(zmq.Socket, "bind", lambda self, endpoint: bound.append(endpoint))

  receivers = [
    bridge.CubeReceiver(),
    bridge.GoalReceiver(),
    bridge.JointReceiver(),
  ]
  publishers = [bridge.GoalPublisher(), bridge.JointPublisher()]
  try:
    assert connected == [
      "tcp://localhost:5655",
      "tcp://localhost:5656",
      "tcp://localhost:5657",
    ]
    assert bound == ["tcp://*:5656", "tcp://*:5657"]
  finally:
    for receiver in receivers:
      _stop_receiver(receiver)
    for publisher in publishers:
      publisher.close()
