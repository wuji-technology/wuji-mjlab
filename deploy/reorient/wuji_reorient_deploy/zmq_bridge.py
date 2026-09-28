# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""ZMQ pub/sub for the 3-process deploy.

All quaternions are (w, x, y, z).
Cube message schema (published by cube_world_observer.py):
  {"world_fixed": bool, "cube1": {"position": {x,y,z},
                                  "orientation": {x,y,z,w}}}   # scipy xyzw
Joint message schema (published by run_policy.py):
  {"joint_pos": [...], "joint_enc": [...]}  # joint_enc is optional
"""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import zmq

from .config_loader import cube_port, goal_port, joint_port


class _SubThread:
  """Run a background ZeroMQ subscriber loop and cache decoded messages."""

  def __init__(self, port: int, host: str = "localhost"):
    """Connect a subscriber socket and start its background receive thread."""
    self._ctx = zmq.Context.instance()
    self._sock = self._ctx.socket(zmq.SUB)
    self._sock.connect(f"tcp://{host}:{port}")
    self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
    self._sock.setsockopt(zmq.RCVTIMEO, 200)
    self._lock = threading.Lock()
    self._stop = False
    self._t = threading.Thread(target=self._run, daemon=True)
    self._t.start()

  def _run(self):
    while not self._stop:
      try:
        msg = json.loads(self._sock.recv_string())
      except zmq.Again:
        continue
      except Exception:
        continue
      with self._lock:
        self._on_msg(msg)

  def _on_msg(self, msg: dict) -> None:
    ...

  def close(self):
    """Stop the background receive loop."""
    self._stop = True


class CubeReceiver(_SubThread):
  """Receive and cache cube poses in the tag frame."""

  def __init__(self, port: int | None = None, host: str = "localhost"):
    """Start receiving cube pose messages."""
    self._pos = np.zeros(3)
    self._quat = np.array([1.0, 0.0, 0.0, 0.0])
    self._count = 0
    self._last_t: float | None = None
    self._first_fix = threading.Event()
    super().__init__(cube_port() if port is None else port, host)

  def _on_msg(self, msg: dict) -> None:
    if not msg.get("world_fixed", False):
      return
    c = msg.get("cube1")
    if not c:
      return
    p, o = c["position"], c["orientation"]
    self._pos = np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
    self._quat = np.array([o["w"], o["x"], o["y"], o["z"]], dtype=np.float64)
    self._count += 1
    self._last_t = time.monotonic()
    self._first_fix.set()

  def latest(self) -> tuple[np.ndarray, np.ndarray]:
    """Return the most recently received cube pose."""
    with self._lock:
      return self._pos.copy(), self._quat.copy()

  def age_s(self) -> float | None:
    """Seconds since the last accepted cube pose."""
    with self._lock:
      return None if self._last_t is None else time.monotonic() - self._last_t

  def wait_for_fix(self, timeout_s: float) -> bool:
    """Block until the first valid cube pose arrives."""
    return self._first_fix.wait(timeout_s)

  @property
  def count(self) -> int:
    """Return the number of valid cube pose messages received."""
    return self._count


class GoalReceiver(_SubThread):
  """Receive and cache goal orientations in the tag frame."""
  def __init__(self, port: int | None = None, host: str = "localhost"):
    """Start receiving goal orientation messages."""
    self._quat = np.array([1.0, 0.0, 0.0, 0.0])
    super().__init__(goal_port() if port is None else port, host)

  def _on_msg(self, msg: dict) -> None:
    o = msg["orientation"]
    self._quat = np.array([o["w"], o["x"], o["y"], o["z"]], dtype=np.float64)

  def latest(self) -> np.ndarray:
    """Return the most recently received goal orientation."""
    with self._lock:
      return self._quat.copy()


class JointReceiver(_SubThread):
  """Receive and cache commanded and measured joint positions."""
  def __init__(self, port: int | None = None, host: str = "localhost"):
    """Start receiving joint-position messages."""
    self._q = None
    self._q_measured = None
    super().__init__(joint_port() if port is None else port, host)

  def _on_msg(self, msg: dict) -> None:
    self._q = np.asarray(msg["joint_pos"], dtype=np.float64)
    joint_enc = msg.get("joint_enc")
    self._q_measured = (
      None if joint_enc is None else np.asarray(joint_enc, dtype=np.float64)
    )

  def latest(self) -> np.ndarray | None:
    """Return the most recently received joint positions."""
    with self._lock:
      return None if self._q is None else self._q.copy()

  def latest_measured(self) -> np.ndarray | None:
    """Return the measured joint encoders from the latest message."""
    with self._lock:
      return None if self._q_measured is None else self._q_measured.copy()


class _Publisher:
  """Bind a ZeroMQ publisher socket for deployment telemetry."""
  def __init__(self, port: int):
    """Bind a publisher socket."""
    self._ctx = zmq.Context.instance()
    self._sock = self._ctx.socket(zmq.PUB)
    self._sock.bind(f"tcp://*:{port}")

  def _send(self, obj: dict) -> None:
    self._sock.send_string(json.dumps(obj), flags=zmq.NOBLOCK)

  def close(self):
    """Close the publisher socket."""
    self._sock.close(0)


class GoalPublisher(_Publisher):
  """Publish goal orientations in the tag frame."""
  def __init__(self, port: int | None = None):
    """Start publishing goal orientation messages."""
    super().__init__(goal_port() if port is None else port)

  def publish(self, quat_wxyz: np.ndarray) -> None:
    """Publish one goal orientation."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    self._send({"orientation": {"w": w, "x": x, "y": y, "z": z}})


class JointPublisher(_Publisher):
  """Publish commanded joint positions."""
  def __init__(self, port: int | None = None):
    """Start publishing joint-position messages."""
    super().__init__(joint_port() if port is None else port)

  def publish(self, joint_pos, joint_enc=None) -> None:
    """Publish the commanded target and, when available, the measured encoders."""
    msg = {
      "joint_pos": [float(v) for v in np.asarray(joint_pos).reshape(-1)]
    }
    if joint_enc is not None:
      msg["joint_enc"] = [
        float(v) for v in np.asarray(joint_enc).reshape(-1)
      ]
    self._send(msg)
