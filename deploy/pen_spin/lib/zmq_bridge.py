# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""ZMQ SUB for the camera-fed pen pose.

  :5555  pen pose (wrist-tag frame)   observer --> policy runner (+ viewer)

Wire schema (published by scripts/pen_world_observer.py):
  {"world_fixed": bool,
   "pen": {"position": {"x":..,"y":..,"z":..},
           "orientation": {"x":..,"y":..,"z":..,"w":..}}}   # scipy xyzw

Quaternions cross this boundary as scipy (x,y,z,w) and leave as (w,x,y,z);
this module is the ONLY place that conversion happens.

Staleness is a CALLER policy decision — this layer reports age only, the
same latest-sample contract used by runtime/streams.py, so the control loop
logic carries over unchanged.
"""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import zmq

PEN_PORT = 5555


class PenReceiver:
  """Latest pen pose in the wrist-tag frame; caches the last valid sample."""

  def __init__(self, port: int = PEN_PORT, host: str = "localhost"):
    self._pos = np.zeros(3)
    self._quat = np.array([1.0, 0.0, 0.0, 0.0])
    self._recv_t: float | None = None
    self.seen = False
    self.rejected = 0
    self.untracked = 0
    self._lock = threading.Lock()
    self._stop = False
    self._ctx = zmq.Context.instance()
    self._sock = self._ctx.socket(zmq.SUB)
    self._sock.connect(f"tcp://{host}:{port}")
    self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
    self._sock.setsockopt(zmq.RCVTIMEO, 200)
    self._t = threading.Thread(target=self._run, daemon=True)
    self._t.start()

  def _run(self) -> None:
    while not self._stop:
      try:
        msg = json.loads(self._sock.recv_string())
      except zmq.Again:
        continue
      except Exception:
        continue
      self._on_msg(msg)

  def _on_msg(self, msg: dict) -> None:
    if not msg.get("world_fixed", False):
      # Observer running, tag out of frame — distinct from silence on the
      # socket. Not a `rejected` frame: the schema is fine, the world just
      # is not fixed yet.
      self.untracked += 1
      return
    body = msg.get("pen")
    if not body:
      self.untracked += 1
      return
    try:
      p, o = body["position"], body["orientation"]
      pos = np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
      quat = np.array([o["w"], o["x"], o["y"], o["z"]], dtype=np.float64)
    except (KeyError, TypeError):
      self.rejected += 1
      return
    # A single corrupt frame must never reach the policy obs.
    if not (np.isfinite(pos).all() and np.isfinite(quat).all()):
      self.rejected += 1
      return
    if np.linalg.norm(quat) < 1e-6:
      self.rejected += 1
      return
    with self._lock:
      self._pos = pos
      self._quat = quat / np.linalg.norm(quat)
      self._recv_t = time.monotonic()
      self.seen = True

  def latest(self) -> tuple[np.ndarray, np.ndarray, float | None]:
    """(pos, quat_wxyz, age_s). age is None until the first valid frame."""
    with self._lock:
      age = None if self._recv_t is None else time.monotonic() - self._recv_t
      return self._pos.copy(), self._quat.copy(), age

  def close(self) -> None:
    self._stop = True
    self._t.join(timeout=1.0)
    self._sock.close(linger=0)
