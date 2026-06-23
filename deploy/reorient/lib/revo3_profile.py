# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Revo3 deploy profile, joint-order mapping, and vision-frame transforms."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

JOINT_DIM = 21


def _as_vector(value: Any, name: str, dim: int = JOINT_DIM) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float32).reshape(-1)
    if vec.shape != (dim,):
        raise ValueError(f"{name} must have shape ({dim},), got {vec.shape}.")
    if not np.isfinite(vec).all():
        raise ValueError(f"{name} contains NaN/Inf.")
    return vec


def _as_quat_wxyz(value: Any, name: str) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64).reshape(-1)
    if quat.shape != (4,):
        raise ValueError(f"{name} must have shape (4,), got {quat.shape}.")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        raise ValueError(f"{name} has zero norm.")
    quat = quat / norm
    if quat[0] < 0.0:
        quat = -quat
    return quat


def quat_mul_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = lhs
    w2, x2, y2, z2 = rhs
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_apply_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    vx, vy, vz = vec
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.array(
        [
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class Revo3Profile:
    path: Path
    hand: str
    policy_joint_order: tuple[str, ...]
    sdk_joint_order: tuple[str, ...]
    joint_lower_policy: np.ndarray
    joint_upper_policy: np.ndarray
    joint_lower_sdk: np.ndarray
    joint_upper_sdk: np.ndarray
    policy_to_sdk_perm: np.ndarray
    sdk_to_policy_perm: np.ndarray
    sdk_offset_rad: np.ndarray
    policy_offset_rad: np.ndarray
    home_joint_pos_policy: np.ndarray
    default_rate_hz: float
    sdk: dict[str, Any]
    mit: dict[str, Any]
    vision_to_policy_pos: np.ndarray
    vision_to_policy_quat: np.ndarray
    vision_transform_calibrated: bool

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_policy_joint_order: tuple[str, ...] | list[str] | None = None,
    ) -> "Revo3Profile":
        profile_path = Path(path)
        with profile_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"{profile_path} must contain a YAML mapping.")

        policy_order = tuple(str(v) for v in (cfg.get("policy_joint_order") or []))
        sdk_order = tuple(
            str(v)
            for v in (cfg.get("sdk_joint_order") or cfg.get("controller_joint_order") or [])
        )
        _validate_joint_order(policy_order, "policy_joint_order")
        _validate_joint_order(sdk_order, "sdk_joint_order")
        if set(policy_order) != set(sdk_order):
            raise ValueError("policy_joint_order and sdk_joint_order must contain the same joints.")
        if expected_policy_joint_order is not None and policy_order != tuple(expected_policy_joint_order):
            raise ValueError("profile policy_joint_order differs from artifact policy_joint_order.")

        limits = cfg.get("joint_limits") or {}
        lower = []
        upper = []
        for joint in policy_order:
            item = limits.get(joint)
            if not isinstance(item, dict):
                raise ValueError(f"joint_limits missing {joint!r}.")
            lower.append(float(item["lower"]))
            upper.append(float(item["upper"]))
        joint_lower_policy = _as_vector(lower, "joint_lower_policy")
        joint_upper_policy = _as_vector(upper, "joint_upper_policy")
        if np.any(joint_upper_policy <= joint_lower_policy):
            raise ValueError("Every joint upper limit must be greater than lower.")

        policy_to_sdk = np.asarray([policy_order.index(name) for name in sdk_order], dtype=np.int64)
        sdk_to_policy = np.asarray([sdk_order.index(name) for name in policy_order], dtype=np.int64)
        joint_lower_sdk = joint_lower_policy[policy_to_sdk]
        joint_upper_sdk = joint_upper_policy[policy_to_sdk]

        sdk_offset = _load_sdk_offset(cfg)
        policy_offset = np.zeros(JOINT_DIM, dtype=np.float32)
        for sdk_idx, policy_idx in enumerate(policy_to_sdk):
            policy_offset[policy_idx] = sdk_offset[sdk_idx]

        home_map = cfg.get("home_joint_pos") or {}
        home = np.asarray(
            [float(home_map.get(joint, 0.0)) for joint in policy_order],
            dtype=np.float32,
        )

        frame = cfg.get("vision_to_policy_frame") or {}
        frame_pos = np.asarray(frame.get("position_xyz", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        frame_quat = _as_quat_wxyz(frame.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0]), "vision_to_policy_frame.quat_wxyz")

        return cls(
            path=profile_path,
            hand=str(cfg.get("hand", "right")),
            policy_joint_order=policy_order,
            sdk_joint_order=sdk_order,
            joint_lower_policy=joint_lower_policy,
            joint_upper_policy=joint_upper_policy,
            joint_lower_sdk=joint_lower_sdk,
            joint_upper_sdk=joint_upper_sdk,
            policy_to_sdk_perm=policy_to_sdk,
            sdk_to_policy_perm=sdk_to_policy,
            sdk_offset_rad=sdk_offset,
            policy_offset_rad=policy_offset,
            home_joint_pos_policy=home,
            default_rate_hz=float(cfg.get("default_rate_hz", 20.0)),
            sdk=dict(cfg.get("sdk") or {}),
            mit=dict(cfg.get("mit") or {}),
            vision_to_policy_pos=frame_pos,
            vision_to_policy_quat=frame_quat,
            vision_transform_calibrated=bool(frame.get("calibrated", False)),
        )

    def measured_sdk_to_policy(self, sdk_pos_rad: np.ndarray) -> np.ndarray:
        sdk_pos = _as_vector(sdk_pos_rad, "sdk_pos_rad")
        return (sdk_pos[self.sdk_to_policy_perm] - self.policy_offset_rad).astype(np.float32)

    def target_policy_to_sdk(self, policy_target_rad: np.ndarray) -> np.ndarray:
        policy_target = _as_vector(policy_target_rad, "policy_target_rad")
        sdk_target = policy_target[self.policy_to_sdk_perm] + self.sdk_offset_rad
        return np.clip(sdk_target, self.joint_lower_sdk, self.joint_upper_sdk).astype(np.float32)

    def transform_vision_pose_to_policy(
        self,
        pos_xyz: np.ndarray,
        quat_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform a pose from physical vision tag frame to policy tag frame."""
        pos = np.asarray(pos_xyz, dtype=np.float64).reshape(3)
        quat = _as_quat_wxyz(quat_wxyz, "quat_wxyz")
        out_pos = self.vision_to_policy_pos + quat_apply_wxyz(self.vision_to_policy_quat, pos)
        out_quat = _as_quat_wxyz(quat_mul_wxyz(self.vision_to_policy_quat, quat), "policy_quat_wxyz")
        return out_pos.astype(np.float64), out_quat.astype(np.float64)

    def active_control_allowed(self) -> bool:
        return self.vision_transform_calibrated


class Revo3CubePoseAdapter:
    """Wrap a CubeReceiver-like object and expose policy-frame poses."""

    def __init__(self, source, profile: Revo3Profile) -> None:
        self.source = source
        self.profile = profile

    @property
    def cube_size(self) -> float | None:
        return getattr(self.source, "cube_size", None)

    def latest(self) -> tuple[np.ndarray, np.ndarray]:
        pos, quat = self.source.latest()
        return self.profile.transform_vision_pose_to_policy(pos, quat)


class Revo3GoalPoseAdapter:
    """Wrap a GoalReceiver-like object and expose policy-frame quaternions."""

    def __init__(self, source, profile: Revo3Profile) -> None:
        self.source = source
        self.profile = profile

    def latest(self) -> np.ndarray:
        _pos, quat = self.profile.transform_vision_pose_to_policy(
            np.zeros(3, dtype=np.float64),
            self.source.latest(),
        )
        return quat


def _validate_joint_order(order: tuple[str, ...], name: str) -> None:
    if len(order) != JOINT_DIM:
        raise ValueError(f"{name} must contain {JOINT_DIM} joints, got {len(order)}.")
    if len(set(order)) != JOINT_DIM:
        raise ValueError(f"{name} contains duplicate joints.")


def _load_sdk_offset(cfg: dict[str, Any]) -> np.ndarray:
    offset_cfg = cfg.get("sim2real_joint_offset") or {}
    if not offset_cfg:
        return np.zeros(JOINT_DIM, dtype=np.float32)
    if offset_cfg.get("order") not in {"sdk_joint_order", "controller_joint_order"}:
        raise ValueError("sim2real_joint_offset.order must be sdk_joint_order.")
    return _as_vector(offset_cfg.get("values") or [], "sim2real_joint_offset.values")
