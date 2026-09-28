# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import ctypes

import pytest
from wuji_reorient_deploy.camera_config import setup_camera_capture
from wuji_reorient_deploy.mvs_sdk import ensure_mvs_importable, find_mvs_python_path


def _capture_config(frame_rate: float) -> dict:
  return {
    "capture": {
      "exposure_time": 5000.0,
      "gain": 1.0,
      "frame_rate": frame_rate,
    }
  }


class _FakeCamera:
  def MV_CC_SetFloatValue(self, key, value):
    return 0

  def MV_CC_SetBoolValue(self, key, value):
    return 0


def test_resulting_frame_rate_uses_vendor_float_structure(capsys):
  if find_mvs_python_path() is None:
    pytest.skip("MVS SDK ABI test requires a vision deployment machine or SDK runner")
  ensure_mvs_importable()
  from MvImport.CameraParams_header import MVCC_FLOATVALUE

  class VendorCamera(_FakeCamera):
    def MV_CC_GetFloatValue(self, key, buf):
      assert key == "ResultingFrameRate"
      assert isinstance(buf, MVCC_FLOATVALUE)
      assert ctypes.sizeof(buf) == ctypes.sizeof(MVCC_FLOATVALUE)
      buf.fCurValue = 47.25
      return 0

  setup_camera_capture(VendorCamera(), _capture_config(50.0))
  assert "actual=" in capsys.readouterr().out


def test_zero_frame_rate_does_not_configure_or_query_rate():
  rate_calls = []

  class RecordingCamera(_FakeCamera):
    def MV_CC_SetBoolValue(self, key, value):
      rate_calls.append(("enable", key, value))
      return 0

    def MV_CC_SetFloatValue(self, key, value):
      if key == "AcquisitionFrameRate":
        rate_calls.append(("set", key, value))
      return 0

    def MV_CC_GetFloatValue(self, key, buf):
      rate_calls.append(("query", key))
      return 0

  setup_camera_capture(RecordingCamera(), _capture_config(0))
  assert rate_calls == []
