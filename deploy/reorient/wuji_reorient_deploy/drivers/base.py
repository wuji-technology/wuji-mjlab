# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Hand driver interface (``HandDriverBase``) + hardware-free ``MockHandDriver``.

All joint vectors are 20 angles in radians, finger-major == the ONNX action
order. ``close()`` is idempotent.
"""
from __future__ import annotations

import abc
import time

import numpy as np

from ..constants import Constants


class HandDriverBase(abc.ABC):
  """Common driver contract shared by every generation's real/mock driver."""

  def __init__(self, constants: Constants, hand_side: str = "right"):
    self.constants = constants
    self.hand_side = hand_side

  @abc.abstractmethod
  def write_target(self, qpos: np.ndarray) -> None: ...

  @abc.abstractmethod
  def read_encoders(self) -> np.ndarray: ...

  def fault(self) -> str | None:
    """Human-readable description of a latched hardware fault, or None."""
    return None

  def temp_summary(self) -> str | None:
    """Human-readable temperature summary, or None if unavailable."""
    return None

  def home(self, duration_s: float = 2.0, rate_hz: float = 100.0) -> None:
    """Cosine-ease from the current pose to the cage home pose, then hold."""
    start = self.read_encoders()
    goal = self.constants.default_joint_pos.copy()
    n = max(1, int(duration_s * rate_hz))
    for i in range(n + 1):
      s = 0.5 * (1.0 - np.cos(np.pi * i / n))
      self.write_target(start + s * (goal - start))
      time.sleep(1.0 / rate_hz)

  def joint_names_in_encoder_order(self) -> tuple[str, ...]:
    from ..constants import joint_names
    return tuple(joint_names(self.hand_side))

  def close(self) -> None:
    pass

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
    self.close()
    return False


class MockHandDriver(HandDriverBase):
  """Instantly-settling mock — runs the whole pipeline without hardware."""

  def __init__(self, constants: Constants, hand_side: str = "right"):
    super().__init__(constants, hand_side)
    self._q = constants.default_joint_pos.copy()

  def write_target(self, qpos: np.ndarray) -> None:
    self._q = np.asarray(qpos, dtype=np.float64).copy()

  def read_encoders(self) -> np.ndarray:
    return self._q.copy()

  def home(self, duration_s: float = 2.0, rate_hz: float = 100.0) -> None:
    """Instant-settle: no hardware to ramp, so jump straight to the home pose."""
    self._q = self.constants.default_joint_pos.copy()
