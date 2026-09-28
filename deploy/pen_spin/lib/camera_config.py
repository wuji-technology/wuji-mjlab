#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Camera Configuration Loader.

Centralized camera parameters from config/camera.yaml.
Provides functions for loading camera intrinsics, distortion, and ROI settings.

Example:
    >>> from camera_config import get_camera_matrix, get_dist_coeffs
    >>> K = get_camera_matrix()
    >>> dist = get_dist_coeffs()
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import yaml

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
"""Directory containing this script."""

ROOT_DIR: str = os.path.dirname(SCRIPT_DIR)
"""Root directory of deploy/reorient (one level up from lib/)."""

CONFIG_FILE: str = os.path.join(ROOT_DIR, "config", "camera.yaml")
"""Path to camera.yaml configuration file."""

def load_camera_config(config_file: str | None = None) -> dict[str, Any]:
    """Load camera configuration from YAML file.

    Args:
        config_file: Path to configuration file. If None, uses default.

    Returns:
        Camera configuration dictionary.

    Raises:
        FileNotFoundError: If configuration file doesn't exist.
    """
    if config_file is None:
        config_file = CONFIG_FILE

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Camera config not found: {config_file}")

    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    return cfg


def get_camera_matrix(cfg: dict[str, Any] | None = None) -> np.ndarray:
    """Get camera intrinsic matrix K, adjusted for ROI offset.

    When ROI is set, cx and cy are shifted by the ROI offset so that the
    intrinsics remain valid for the cropped image.

    Args:
        cfg: Camera configuration dict. If None, loads from file.

    Returns:
        3x3 camera intrinsic matrix.
    """
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
    """Get distortion coefficients.

    Args:
        cfg: Camera configuration dict. If None, loads from file.

    Returns:
        Distortion coefficients array [k1, k2, p1, p2, k3].
    """
    if cfg is None:
        cfg = load_camera_config()

    dist = cfg['distortion']
    return np.array([
        dist['k1'], dist['k2'], dist['p1'], dist['p2'], dist['k3']
    ], dtype=np.float64)


def get_roi(cfg: dict[str, Any] | None = None) -> tuple[int, int, int, int]:
    """Get ROI parameters.

    Args:
        cfg: Camera configuration dict. If None, loads from file.

    Returns:
        Tuple of (offset_x, offset_y, width, height).
    """
    if cfg is None:
        cfg = load_camera_config()

    roi = cfg['roi']
    return roi['offset_x'], roi['offset_y'], roi['width'], roi['height']


def get_capture_settings(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get camera capture settings.

    Args:
        cfg: Camera configuration dict. If None, loads from file.

    Returns:
        Dictionary with exposure_time, gain, and frame_rate.
    """
    if cfg is None:
        cfg = load_camera_config()

    cap = cfg['capture']
    return {
        'exposure_time': cap['exposure_time'],
        'gain': cap['gain'],
        'frame_rate': cap.get('frame_rate', 0),
    }


def setup_camera_roi(cam: Any, cfg: dict[str, Any] | None = None) -> tuple[int, int]:
    """Setup camera ROI from config.

    Args:
        cam: MvCamera instance.
        cfg: Camera configuration dict. If None, loads from file.

    Returns:
        Tuple of (width, height) of the configured ROI.
    """
    offset_x, offset_y, width, height = get_roi(cfg)
    cam.MV_CC_SetIntValueEx("OffsetX", offset_x)
    cam.MV_CC_SetIntValueEx("OffsetY", offset_y)
    cam.MV_CC_SetIntValueEx("Width", width)
    cam.MV_CC_SetIntValueEx("Height", height)
    print(f"Camera ROI: {width}x{height} @ ({offset_x}, {offset_y})")
    return width, height


def setup_camera_capture(cam: Any, cfg: dict[str, Any] | None = None) -> None:
    """Setup camera capture settings from config.

    Args:
        cam: MvCamera instance.
        cfg: Camera configuration dict. If None, loads from file.
    """
    settings = get_capture_settings(cfg)
    # Disable auto exposure so the camera does not override the requested value.
    ret = cam.MV_CC_SetEnumValue("ExposureAuto", 0)
    if ret != 0:
        raise RuntimeError(f"Failed to disable camera auto exposure: 0x{ret & 0xffffffff:X}")
    ret = cam.MV_CC_SetFloatValue("ExposureTime", settings['exposure_time'])
    if ret != 0:
        raise RuntimeError(f"Failed to set exposure to {settings['exposure_time']} us: 0x{ret & 0xffffffff:X}")
    cam.MV_CC_SetFloatValue("Gain", settings['gain'])
    # Frame rate: enable explicit control and set target
    frame_rate = settings.get('frame_rate', 0)
    if frame_rate and frame_rate > 0:
        ret1 = cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        ret2 = cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(frame_rate))
        # Read back actual resulting frame rate
        # The SDK writes the entire structure, so a four-byte c_float is unsafe.
        from MvImport.CameraParams_header import MVCC_FLOATVALUE
        actual_fps = MVCC_FLOATVALUE()
        ret3 = cam.MV_CC_GetFloatValue("ResultingFrameRate", actual_fps)
        if ret3 == 0:
            actual_str = f", actual={actual_fps.fCurValue:.1f}Hz"
        else:
            actual_str = ", actual=unknown"
        print(f"Camera capture: exposure={settings['exposure_time']}us, gain={settings['gain']}, "
              f"frame_rate={frame_rate}Hz (enable_ret=0x{ret1:X}, set_ret=0x{ret2:X}{actual_str})")
    else:
        print(f"Camera capture: exposure={settings['exposure_time']}us, gain={settings['gain']}, frame_rate=default")
