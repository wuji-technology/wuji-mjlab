# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Load the public pen-spin control configuration."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import yaml

# ==============================================================================
# Constants
# ==============================================================================

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
"""Directory containing this script."""

ROOT_DIR: str = os.path.dirname(SCRIPT_DIR)
"""Root directory of wujihand_deploy."""

CONFIG_DIR: str = os.path.join(ROOT_DIR, "config")
"""Directory containing configuration files."""

CONTROL_CONFIG_FILE: str = os.path.join(CONFIG_DIR, "control.yaml")
"""Path to control.yaml."""

# ==============================================================================
# Public Functions
# ==============================================================================


def load_yaml(file_path: str) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If configuration file doesn't exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config not found: {file_path}")
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def get_control_config() -> dict[str, Any]:
    """Load control.yaml (deploy-side timing / smoothing / joint limits)."""
    return load_yaml(CONTROL_CONFIG_FILE)


def get_joint_limits() -> np.ndarray:
    """Get joint limits as numpy array.

    Returns:
        Joint limits array with shape (20, 2) containing [lower, upper]
        bounds for each joint in radians.
    """
    cfg = get_control_config()
    return np.array(cfg['joint_limits'], dtype=np.float64)


def get_timing() -> dict[str, float]:
    """Get timing parameters.

    Returns:
        Dictionary with ctrl_dt, servo_hz, idle_viewer_hz and
        jv_lowpass_cutoff_hz.
    """
    cfg = get_control_config()
    return cfg['timing']


def get_gains() -> dict[str, Any]:
    """Hand drive gains: kp, kd, effort, kp_joint, effort_joint.

    ``--gains k=v`` is a deviation from these, so this is the only place a
    forgotten flag can fall back to. Missing per-joint blocks come back as empty
    dicts rather than None, because every caller merges them.
    """
    cfg = get_control_config()['gains']
    result = {
        'kp': float(cfg['kp']),
        'kd': float(cfg['kd']),
        'effort': float(cfg['effort']),
        'kp_joint': dict(cfg.get('kp_joint') or {}),
        'effort_joint': dict(cfg.get('effort_joint') or {}),
    }
    for key in ('kp', 'kd', 'effort'):
        if not np.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f"gains.{key} must be finite and nonnegative")
    return result
