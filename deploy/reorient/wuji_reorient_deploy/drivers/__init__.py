# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Generation-dispatched hand driver factory."""
from __future__ import annotations

from ..constants import Constants, load_constants
from .base import HandDriverBase, MockHandDriver
from .hand1 import WujiHand1Driver
from .hand2 import WujiHand2Driver

__all__ = ["HandDriverBase", "MockHandDriver", "make_driver"]


def make_driver(
  gen: int,
  hand_side: str,
  serial_number: str | None = None,
  mock: bool = False,
  constants: Constants | None = None,
  **kw,
) -> HandDriverBase:
  """Build the driver for (gen, hand_side)."""
  if constants is None:
    constants = load_constants(gen, hand_side)
  if mock:
    return MockHandDriver(constants, hand_side)
  if gen == 1:
    return WujiHand1Driver(constants, hand_side, serial_number, **kw)
  if gen == 2:
    return WujiHand2Driver(constants, hand_side, serial_number, **kw)
  raise ValueError(f"gen must be 1 or 2, got {gen!r}")
