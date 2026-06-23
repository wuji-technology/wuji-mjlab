# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Revo3 hand drivers for dry-run, shadow, and guarded active control."""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .hand_driver import HandDriverBase
from .revo3_profile import JOINT_DIM, Revo3Profile

DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi
RPM_TO_RAD_S = 2.0 * math.pi / 60.0
RAD_S_TO_RPM = 60.0 / (2.0 * math.pi)


def _vector(value, name: str) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float32).reshape(-1)
    if vec.shape != (JOINT_DIM,):
        raise ValueError(f"{name} must have {JOINT_DIM} values, got {vec.shape}.")
    if not np.isfinite(vec).all():
        raise ValueError(f"{name} contains NaN/Inf.")
    return vec


def _load_bc_stark_sdk():
    try:
        from bc_stark_sdk import main_mod as sdk
    except ImportError:
        try:
            from bc_stark_sdk import bc_stark_sdk as sdk
        except ImportError as exc:
            raise RuntimeError(
                "bc-stark-sdk is not installed. Shadow/active Revo3 hardware "
                "modes require the SDK wheel provided for your platform. "
                "Use --mode dry-run for software-only validation."
            ) from exc
    return sdk


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Revo3SdkHandDriver cannot run inside an existing asyncio loop.")


def _run_async_call(func, *args, **kwargs):
    async def call():
        result = func(*args, **kwargs)
        if hasattr(result, "__await__"):
            return await result
        return result

    return _run_async(call())


@dataclass(frozen=True)
class Revo3SdkConfig:
    port: str | None = None
    baudrate: int = 5000000
    slave_id: int = 126
    auto_detect: bool = True


class MockRevo3HandDriver(HandDriverBase):
    """In-memory Revo3 driver.

    ``track_targets=False`` gives shadow-like semantics: writes are recorded but
    encoder reads stay at the current pose.
    """

    def __init__(self, profile: Revo3Profile, *, track_targets: bool = True) -> None:
        self.profile = profile
        self.track_targets = bool(track_targets)
        self._encoders = profile.home_joint_pos_policy.astype(np.float32).copy()
        self.last_target_policy: np.ndarray | None = None
        self.write_count = 0

    def home(self) -> None:
        self._encoders = self.profile.home_joint_pos_policy.astype(np.float32).copy()

    def write_target(self, joint_targets: np.ndarray) -> None:
        target = _vector(joint_targets, "joint_targets")
        self.last_target_policy = target.copy()
        self.write_count += 1
        if self.track_targets:
            self._encoders = target.copy()

    def read_encoders(self) -> np.ndarray:
        return self._encoders.copy()

    def joint_names_in_encoder_order(self) -> tuple[str, ...]:
        return self.profile.policy_joint_order


class Revo3SdkHandDriver(HandDriverBase):
    """Synchronous HandDriver adapter around the async Revo3 SDK.

    In ``shadow`` mode, reads are real but writes are recorded-only.  ``active``
    mode is guarded by both the CLI and this class constructor.
    """

    def __init__(
        self,
        profile: Revo3Profile,
        config: Revo3SdkConfig | None = None,
        *,
        command_mode: str = "shadow",
        allow_active: bool = False,
        kp: float | None = None,
        kd: float | None = None,
        effort_ma: float | None = None,
        home_duration_s: float = 3.0,
    ) -> None:
        if command_mode not in {"shadow", "active"}:
            raise ValueError("command_mode must be 'shadow' or 'active'.")
        if command_mode == "active" and not allow_active:
            raise ValueError("active Revo3 control requires allow_active=True.")
        self.profile = profile
        self.config = config or Revo3SdkConfig()
        self.command_mode = command_mode
        self.allow_active = bool(allow_active)
        self.home_duration_s = float(home_duration_s)

        mit = profile.mit
        self.kp = float(kp if kp is not None else mit.get("kp", 1.0))
        self.kd = float(kd if kd is not None else mit.get("kd", 0.1))
        self.effort_ma = float(
            effort_ma if effort_ma is not None else mit.get("effort_ma", 0.0)
        )

        self.sdk = None
        self.ctx: Any | None = None
        self.slave_id = int(self.config.slave_id)
        self.last_target_policy: np.ndarray | None = None
        self.write_count = 0

    def __enter__(self):
        self.sdk = _load_bc_stark_sdk()
        if hasattr(self.sdk, "init_logging"):
            self.sdk.init_logging()
        port = self.config.port
        baudrate = int(self.config.baudrate)
        slave_id = int(self.config.slave_id)
        if self.config.auto_detect and port is None:
            _kind, port, baudrate, slave_id = _run_async_call(
                self.sdk.auto_detect_modbus_revo3
            )
        self.slave_id = int(slave_id)
        self.ctx = _run_async_call(
            self.sdk.modbus_open,
            port,
            self._baudrate_enum(baudrate),
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ctx is not None and self.sdk is not None:
            _run_async_call(self.sdk.modbus_close, self.ctx)
            self.ctx = None
        return False

    def home(self) -> None:
        if self.command_mode != "active":
            return
        current = self.read_encoders()
        target = self.profile.home_joint_pos_policy.astype(np.float32)
        steps = max(1, int(self.home_duration_s * 50.0))
        dt = self.home_duration_s / steps
        for i in range(steps):
            t = (i + 1) / steps
            smooth = t * t * (3.0 - 2.0 * t)
            self.write_target(current + smooth * (target - current))
            time.sleep(dt)

    def write_target(self, joint_targets: np.ndarray) -> None:
        target_policy = _vector(joint_targets, "joint_targets")
        self.last_target_policy = target_policy.copy()
        self.write_count += 1
        if self.command_mode != "active":
            return
        sdk_target = self.profile.target_policy_to_sdk(target_policy)
        _run_async(self._send_mit_command_sdk_order(sdk_target))

    def read_encoders(self) -> np.ndarray:
        status = _run_async_call(self._ctx.v3_get_motor_status_data, self.slave_id)
        sdk_pos = np.asarray(status.positions[:JOINT_DIM], dtype=np.float32) * DEG_TO_RAD
        return self.profile.measured_sdk_to_policy(sdk_pos)

    def joint_names_in_encoder_order(self) -> tuple[str, ...]:
        return self.profile.policy_joint_order

    @property
    def _ctx(self):
        if self.ctx is None:
            raise RuntimeError("Revo3SdkHandDriver is not open.")
        return self.ctx

    async def _send_mit_command_sdk_order(self, sdk_position_rad: np.ndarray) -> None:
        pos_deg = _vector(sdk_position_rad, "sdk_position_rad") * RAD_TO_DEG
        zeros_rpm = np.zeros(JOINT_DIM, dtype=np.float32) * RAD_S_TO_RPM
        await self._ctx.revo3_multi_mit_set_all(
            self.slave_id,
            self._command_values(self.kp),
            self._command_values(self.kd),
            pos_deg.tolist(),
            zeros_rpm.tolist(),
            self._command_values(self.effort_ma),
        )

    def _baudrate_enum(self, value: int):
        baudrate_type = self.sdk.Baudrate
        if isinstance(value, baudrate_type):
            return value
        if hasattr(baudrate_type, "from_int"):
            return baudrate_type.from_int(value)
        mapping = {
            115200: baudrate_type.Baud115200,
            57600: baudrate_type.Baud57600,
            19200: baudrate_type.Baud19200,
            460800: baudrate_type.Baud460800,
            1000000: baudrate_type.Baud1Mbps,
            2000000: baudrate_type.Baud2Mbps,
            5000000: baudrate_type.Baud5Mbps,
        }
        return mapping[value]

    @staticmethod
    def _command_values(value: float | list[float] | np.ndarray) -> list[float]:
        vec = np.asarray(value, dtype=np.float32).reshape(-1)
        if vec.shape == (1,):
            return [float(vec[0])] * JOINT_DIM
        if vec.shape != (JOINT_DIM,):
            raise ValueError(f"command value must be scalar or {JOINT_DIM} values.")
        return [float(v) for v in vec]
