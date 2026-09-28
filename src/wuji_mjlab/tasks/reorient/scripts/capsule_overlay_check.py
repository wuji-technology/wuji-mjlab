#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Launch the interactive Wuji Hand 2 collision-capsule alignment check."""

from __future__ import annotations

import tyro
from wuji_mjlab.tasks.reorient.tooling.capsule_overlay_check import (
  CapsuleOverlayCheckConfig,
  launch_capsule_overlay_check,
)

if __name__ == "__main__":
  launch_capsule_overlay_check(tyro.cli(CapsuleOverlayCheckConfig))
