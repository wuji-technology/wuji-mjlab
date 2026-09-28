# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Read-only Wuji Hand 2 joint feedback for visualization, independent of policy.

Do not use Hand2Compat here: its close() disables motors. This connection only
subscribes to joint_states and releases its own subscription/connection.
"""
from __future__ import annotations

import threading
import time

import numpy as np


class HandFeedback:
    def __init__(self, serial_number=None, address=None):
        import wuji_sdk

        self._lock = threading.Lock()
        self._positions = np.zeros(20)
        self._times = np.full(20, np.nan)
        self._sub = None
        self._connected = False
        self._name = "pen_viewer_feedback"
        self._mgr = wuji_sdk.SdkManager.instance()
        selector = ({"address": address} if address else {"sn": serial_number} if serial_number else
                    {"handedness": wuji_sdk.Handedness.Right})
        try:
            hand = self._mgr.connect(
                **selector, device_name=self._name,
                options=wuji_sdk.ConnectOptions(enable_bridge=False))
            self._connected = True
            self.serial_number = hand.serial_number
            self._sub = hand.joint_states().subscribe_with_callback(self._on_state)
        except BaseException:
            self.close()
            raise

    def _on_state(self, frame):
        now = time.monotonic()
        with self._lock:
            for entry in frame.joints:
                # SDK nid: 1..4, 6..9, 11..14, 16..19, 21..24.
                # Angles already use radians and the MJCF joint directions,
                # identical to Hand2Compat's actual-position mapping.
                finger, joint = divmod(entry.nid - 1, 5)
                if 0 <= finger < 5 and joint < 4 and np.isfinite(entry.position):
                    index = finger * 4 + joint
                    self._positions[index] = entry.position
                    self._times[index] = now

    def latest(self, stale_s=0.25):
        """(qpos20 or None, oldest age or None, number of fresh joints).

        Per-joint timestamps prevent a stream with one offline joint from
        making that joint's old angle appear live. Partial callbacks accumulate
        only while every joint is still within the freshness window.
        """
        now = time.monotonic()
        with self._lock:
            ages = now - self._times
            fresh = np.isfinite(ages) & (ages <= stale_s)
            count = int(fresh.sum())
            age = float(ages.max()) if np.isfinite(ages).all() else None
            return (self._positions.copy() if count == 20 else None, age, count)

    def close(self):
        try:
            if self._sub is not None:
                sub, self._sub = self._sub, None
                sub.close()
        finally:
            if self._connected:
                self._connected = False
                self._mgr.disconnect(self._name)
