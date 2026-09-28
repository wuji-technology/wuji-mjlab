# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji Hand 1 real-hardware driver (wujihandpy)."""
from __future__ import annotations

import numpy as np

from ..constants import Constants
from .base import HandDriverBase


class WujiHand1Driver(HandDriverBase):
  """Real Wuji Hand 1 via wujihandpy's realtime joint-position controller."""

  def __init__(
    self,
    constants: Constants,
    hand_side: str = "right",
    serial_number: str | None = None,
    effort_limit_a: float = 0.5,
    lowpass_cutoff: float = 3.0,
    kp: float | None = None,  # gen2/MIT-impedance params; wujihandpy uses a
    kd: float | None = None,  # low-pass position controller, so these are
    **_ignored,               # accepted for a uniform make_driver() interface and ignored.
  ):
    super().__init__(constants, hand_side)
    del serial_number, kp, kd, _ignored
    import wujihandpy  # lazy: mock/gen2 paths work without wujihandpy installed

    self._wujihandpy = wujihandpy
    self._hand = wujihandpy.Hand()
    self._hand.write_joint_effort_limit(float(effort_limit_a))

    # write_joint_enabled(True) energizes the motors. From here on, ANY failure
    # (opening the realtime_controller) must de-energize the hand before
    # propagating — otherwise the exception escapes __init__ before the caller
    # ever holds the driver, so its try/finally: close() never runs and the
    # physical hand is left live with no way to disable it.
    self._ctrl = None
    self._ctrl_cm = None
    try:
      self._hand.write_joint_enabled(True)
      self._ctrl_cm = self._hand.realtime_controller(
        enable_upstream=True,
        filter=wujihandpy.filter.LowPass(cutoff_freq=float(lowpass_cutoff)),
      )
      self._ctrl = self._ctrl_cm.__enter__()
    except BaseException:
      self.close()
      raise

  def write_target(self, qpos: np.ndarray) -> None:
    qpos = np.asarray(qpos, dtype=np.float64)
    n = self.constants.num_joints
    if qpos.shape != (n,):
      raise ValueError(f"expected ({n},), got {qpos.shape}")
    if not np.isfinite(qpos).all():
      raise ValueError("WujiHand1Driver refuses NaN/Inf")
    self._ctrl.set_joint_target_position(qpos.reshape(5, 4))

  def read_encoders(self) -> np.ndarray:
    return self._hand.read_joint_actual_position().flatten().astype(np.float64)

  def close(self) -> None:
    try:
      if self._ctrl_cm is not None:
        self._ctrl_cm.__exit__(None, None, None)
    except BaseException as e:
      print(f"[hand1_driver] realtime_controller close failed: {e}")
    finally:
      self._ctrl = None
      self._ctrl_cm = None
    try:
      self._hand.write_joint_enabled(False)
    except BaseException as e:
      print(f"[hand1_driver] disable failed: {e}")
