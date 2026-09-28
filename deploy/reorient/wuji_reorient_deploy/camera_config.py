#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Camera intrinsics, distortion and ROI/capture settings from config/camera.yaml."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import yaml

from wuji_reorient_deploy.mvs_sdk import ensure_mvs_importable

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR: str = os.path.dirname(SCRIPT_DIR)

CONFIG_FILE: str = os.path.join(ROOT_DIR, "config", "camera.yaml")

def load_camera_config(config_file: str | None = None) -> dict[str, Any]:
    """Load camera configuration from YAML file."""
    if config_file is None:
        config_file = CONFIG_FILE

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Camera config not found: {config_file}")

    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    return cfg


def get_camera_matrix(cfg: dict[str, Any] | None = None) -> np.ndarray:
    """Get camera intrinsic matrix K, adjusted for ROI offset."""
    if cfg is None:
        cfg = load_camera_config()

    intr = cfg['intrinsics']
    roi = cfg['roi']
    K = np.array([
        [intr['fx'], 0, intr['cx'] - roi['offset_x']],
        [0, intr['fy'], intr['cy'] - roi['offset_y']],
        [0, 0, 1]
    ], dtype=np.float64)
    return K


def get_dist_coeffs(cfg: dict[str, Any] | None = None) -> np.ndarray:
    """Get distortion coefficients."""
    if cfg is None:
        cfg = load_camera_config()

    dist = cfg['distortion']
    return np.array([
        dist['k1'], dist['k2'], dist['p1'], dist['p2'], dist['k3']
    ], dtype=np.float64)


def get_roi(cfg: dict[str, Any] | None = None) -> tuple[int, int, int, int]:
    """Get ROI parameters."""
    if cfg is None:
        cfg = load_camera_config()

    roi = cfg['roi']
    return roi['offset_x'], roi['offset_y'], roi['width'], roi['height']


def get_capture_settings(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get camera capture settings."""
    if cfg is None:
        cfg = load_camera_config()

    cap = cfg['capture']
    return {
        'exposure_time': cap['exposure_time'],
        'gain': cap['gain'],
        'frame_rate': cap.get('frame_rate', 0),
    }


def setup_camera_roi(cam: Any, cfg: dict[str, Any] | None = None) -> tuple[int, int]:
    """Setup camera ROI from config."""
    offset_x, offset_y, width, height = get_roi(cfg)
    cam.MV_CC_SetIntValueEx("OffsetX", offset_x)
    cam.MV_CC_SetIntValueEx("OffsetY", offset_y)
    cam.MV_CC_SetIntValueEx("Width", width)
    cam.MV_CC_SetIntValueEx("Height", height)
    print(f"Camera ROI: {width}x{height} @ ({offset_x}, {offset_y})")
    return width, height


def setup_camera_capture(cam: Any, cfg: dict[str, Any] | None = None) -> None:
    """Setup camera capture settings from config."""
    settings = get_capture_settings(cfg)
    cam.MV_CC_SetFloatValue("ExposureTime", settings['exposure_time'])
    cam.MV_CC_SetFloatValue("Gain", settings['gain'])
    frame_rate = settings.get('frame_rate', 0)
    if frame_rate and frame_rate > 0:
        ret1 = cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        ret2 = cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(frame_rate))
        ensure_mvs_importable()
        from MvImport.CameraParams_header import MVCC_FLOATVALUE

        actual = MVCC_FLOATVALUE()
        ret3 = cam.MV_CC_GetFloatValue("ResultingFrameRate", actual)
        actual_str = f", actual={actual.fCurValue:.1f}Hz" if ret3 == 0 else ", actual=unknown"
        print(f"Camera capture: exposure={settings['exposure_time']}us, gain={settings['gain']}, "
              f"frame_rate={frame_rate}Hz (enable_ret=0x{ret1:X}, set_ret=0x{ret2:X}{actual_str})")
    else:
        print(f"Camera capture: exposure={settings['exposure_time']}us, gain={settings['gain']}, frame_rate=default")


if __name__ == "__main__":
    print("=" * 50)
    print("Camera Config Test")
    print("=" * 50)

    cfg = load_camera_config()
    print("\nCamera Config loaded:")
    print(f"  ROI: {get_roi(cfg)}")
    print(f"  K:\n{get_camera_matrix(cfg)}")
    print(f"  Dist: {get_dist_coeffs(cfg)}")
    print(f"  Capture: {get_capture_settings(cfg)}")
