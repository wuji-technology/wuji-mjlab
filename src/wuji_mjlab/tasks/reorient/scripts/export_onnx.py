#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Export a trained reorient policy checkpoint to ONNX format."""

from __future__ import annotations

from wuji_mjlab.tasks.reorient.tooling.onnx_export_core import main

if __name__ == "__main__":
  main()
