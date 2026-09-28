#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Launch the interactive Wuji Hand 2 cage-boundary check."""

from __future__ import annotations

import tyro
from wuji_mjlab.tasks.reorient.tooling.cage_bounds_check import (
  CageBoundsCheckConfig,
  launch_cage_bounds_check,
)

if __name__ == "__main__":
  launch_cage_bounds_check(tyro.cli(CageBoundsCheckConfig))
