# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Hand output: the driver facade plus the joint-limit clamp.

Two implementations behind one surface — the real Wuji Hand 2 over wuji_sdk, and a
perfect-actuator stub for ``--no-hardware`` whose reported position echoes the
last commanded target (so a sim run's finite-diff velocity reflects exactly what
the policy asked for). The stub lives here, next to the thing it stands in for,
rather than in the deploy entry script.

The clamp is the last thing between a policy and a physical finger, so it is a
plain function: ``clamp_to_limits`` is called by the loop and by the equivalence
check, and there is no path to the SDK that skips it.
"""

from __future__ import annotations

import numpy as np
from lib.config_loader import get_joint_limits
from lib.motion_clip import RIGHT_FINGER_JOINT_NAMES

# Name -> flat finger-major index, the addressing the adapter and the device's
# 20-element resources use. Built from the canonical list rather than the
# `finger*4 + joint` formula so a typo fails instead of silently landing on a
# valid-but-wrong joint.
_JOINT_INDEX = {n: i for i, n in enumerate(RIGHT_FINGER_JOINT_NAMES)}

ACTION_DIM = 20


def joint_limits() -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) as (5,4), from config/control.yaml.

    Same source the real driver reports, so the stub and the hardware agree and
    an SDK<->MuJoCo identity check stays meaningful.
    """
    lims = get_joint_limits()
    return lims[:, 0].reshape(5, 4), lims[:, 1].reshape(5, 4)


def clamp_to_limits(target20: np.ndarray, lower, upper) -> np.ndarray:
    """(20,) command -> (5,4) clamped into the mechanical range."""
    return np.clip(np.asarray(target20, dtype=np.float64).reshape(5, 4), lower, upper)


class StubController:
    """Perfect actuator: reports back whatever was last commanded."""

    def __init__(self, seed_5x4: np.ndarray):
        self._target = np.asarray(seed_5x4, dtype=np.float64).reshape(5, 4).copy()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_joint_actual_position(self) -> np.ndarray:
        return self._target.copy()

    def set_joint_target_position(self, value_array) -> None:
        self._target = np.asarray(value_array, dtype=np.float64).reshape(5, 4).copy()

    def close(self) -> None:
        pass


class StubHand:
    """--no-hardware stand-in.

    All controller call sites share ONE StubController so the target latch is
    continuous across phases (no jump back to the seed between ease and loop).
    """

    is_real = False

    def __init__(self, lower, upper, seed_5x4):
        self._lower = np.asarray(lower, dtype=np.float64).reshape(5, 4)
        self._upper = np.asarray(upper, dtype=np.float64).reshape(5, 4)
        self._ctrl = StubController(seed_5x4)

    def realtime_controller(self, *_, **__) -> StubController:
        return self._ctrl

    def read_joint_actual_position(self) -> np.ndarray:
        return self._ctrl.get_joint_actual_position()

    def read_joint_upper_limit(self) -> np.ndarray:
        return self._upper.copy()

    def read_joint_lower_limit(self) -> np.ndarray:
        return self._lower.copy()

    def write_joint_effort_limit(self, *_, **__) -> None:
        pass

    def write_joint_enabled(self, *_, **__) -> None:
        pass

    def close(self) -> None:
        pass


def _resolve_gain_keys(overrides: dict | None, what: str) -> dict[int, float]:
    """Per-joint gain override keys -> the adapter's flat 0..19 joint index.

    Operators address joints by NAME everywhere they can reach: `gains.kp_joint`
    in config/control.yaml, `--gains kp.finger5_joint2=10` on the command line.
    The adapter — and under it the device's `mit_params` / `effort_limit`
    resources — is a flat 20-element finger-major array. `connect_real` is the
    single funnel both sources pass through, so the translation belongs here.

    Accepts a name with or without the `right_` prefix (the config files omit it,
    lib.motion_clip's canonical list carries it).
    """
    out: dict[int, float] = {}
    for key, val in (overrides or {}).items():
        name = str(key)
        full = name if name.startswith("right_") else f"right_{name}"
        if full not in _JOINT_INDEX:
            raise ValueError(
                f"gains.{what}: unknown joint {name!r}. Expected one of "
                f"{', '.join(n.removeprefix('right_') for n in RIGHT_FINGER_JOINT_NAMES)}")
        idx = _JOINT_INDEX[full]
        value = float(val)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"gains.{what}: {key!r} must be finite and nonnegative")
        out[idx] = value
    return out


def connect_real(*, serial_number: str | None,
                 kp: float, kd: float, servo_hz: float, interp_s: float,
                 kp_overrides: dict | None = None,
                 effort_overrides: dict | None = None):
    """Open the Wuji Hand 2. Raises on any failure — there is no fallback driver.

    ``interp_s`` is passed explicitly rather than left to the adapter's default:
    it is the servo ramp window and must be exactly one control period, so the
    caller — who is the only one holding the checkpoint's ctrl_dt — is the only
    one who can state it.

    The two override dicts are keyed by joint name; see
    ``_resolve_gain_keys``. Resolving before the adapter is constructed means a
    typo fails here, with the name in the message, instead of after the hand is
    already connected and enabled.

    Import is local because lib.hand2_adapter pulls in wuji_sdk, which a
    --no-hardware or sim-only machine need not have installed.
    """
    kp_idx = _resolve_gain_keys(kp_overrides, "kp_joint")
    effort_idx = _resolve_gain_keys(effort_overrides, "effort_joint")

    from lib.hand2_adapter import Hand2Compat

    hand = Hand2Compat(
        serial_number=serial_number,
        kp=kp, kd=kd, servo_hz=servo_hz, interp_s=interp_s,
        kp_overrides=kp_idx,
        effort_overrides=effort_idx,
    )
    hand.is_real = True
    return hand
