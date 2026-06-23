# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Replay-dataset trajectory helpers for Revo3 policy artifacts.

This module is intentionally ROS-free and hardware-free. It only loads saved
artifact targets, validates their contract, and converts them through the
Revo3 profile into SDK-order command previews.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .revo3_policy_artifact import ACTION_DIM, Revo3PolicyArtifact
from .revo3_profile import Revo3Profile

TARGET_SOURCE = "target_post_clip_policy_order"
FALLBACK_TARGET_SOURCE = "joint_pos_policy_order"


@dataclass(frozen=True)
class ArtifactTrajectory:
    artifact_dir: Path
    replay_path: Path
    target_source: str
    warning: str | None
    policy_order: tuple[str, ...]
    frame_index: np.ndarray
    timestamp_sec: np.ndarray
    targets_policy_order: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.targets_policy_order.shape[0])


@dataclass(frozen=True)
class LimitIssue:
    sdk_index: int
    policy_index: int
    joint_name: str
    value_rad: float
    lower_rad: float
    upper_rad: float
    clamped_rad: float
    reason: str


@dataclass(frozen=True)
class ReplayCommandFrame:
    dataset_index: int
    frame_index: int
    timestamp_sec: float
    target_policy_order: np.ndarray
    raw_final_sdk_order: np.ndarray
    command_sdk_order: np.ndarray
    command_policy_order: np.ndarray
    clipped: tuple[LimitIssue, ...]


@dataclass(frozen=True)
class ReplayPlan:
    trajectory: ArtifactTrajectory
    profile_path: Path
    start_frame: int
    frames: tuple[ReplayCommandFrame, ...]

    @property
    def clipped_count(self) -> int:
        return sum(len(frame.clipped) for frame in self.frames)


def load_artifact_trajectory(
    artifact_dir: str | Path,
    replay_data: str | Path | None = None,
) -> ArtifactTrajectory:
    """Load saved absolute replay targets from a Revo3 policy artifact."""
    artifact = Revo3PolicyArtifact.load(artifact_dir)
    root = artifact.artifact_dir
    replay_path = Path(replay_data) if replay_data is not None else root / "replay_dataset.npz"
    if not replay_path.exists():
        raise FileNotFoundError(f"replay_dataset.npz not found: {replay_path}")

    with np.load(replay_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    policy_order = artifact.policy_joint_order
    replay_order = _require_joint_order(arrays, policy_order)
    if replay_order != policy_order:
        raise ValueError(
            "replay_dataset.npz joint_names_policy_order must match "
            "policy_manifest.yaml policy_joint_order."
        )

    warning: str | None = None
    if TARGET_SOURCE in arrays:
        target_source = TARGET_SOURCE
    elif FALLBACK_TARGET_SOURCE in arrays:
        target_source = FALLBACK_TARGET_SOURCE
        warning = (
            "target_post_clip_policy_order is missing; using "
            "joint_pos_policy_order fallback for preview/replay targets."
        )
    else:
        raise ValueError(
            "replay_dataset.npz does not contain target_post_clip_policy_order "
            "or joint_pos_policy_order; refusing to derive trajectory targets "
            "from action_raw."
        )

    frame_index = _require_array(arrays, "frame_index", ndim=1).astype(np.int64)
    timestamp_sec = _require_array(arrays, "timestamp_sec", ndim=1).astype(np.float64)
    if frame_index.shape != timestamp_sec.shape:
        raise ValueError("frame_index and timestamp_sec must have matching shape.")
    if frame_index.shape[0] == 0:
        raise ValueError("replay_dataset.npz contains no trajectory frames.")
    if not np.isfinite(timestamp_sec).all():
        raise ValueError("timestamp_sec contains NaN/Inf.")
    if timestamp_sec.shape[0] > 1 and np.any(np.diff(timestamp_sec) <= 0.0):
        raise ValueError("timestamp_sec must be strictly increasing.")

    targets = _require_array(
        arrays,
        target_source,
        ndim=2,
        trailing_dim=ACTION_DIM,
    ).astype(np.float32)
    if targets.shape[0] != frame_index.shape[0]:
        raise ValueError(
            f"{target_source} frame count {targets.shape[0]} does not match "
            f"frame_index count {frame_index.shape[0]}."
        )
    if not np.isfinite(targets).all():
        raise ValueError(f"{target_source} contains NaN/Inf.")

    return ArtifactTrajectory(
        artifact_dir=root,
        replay_path=replay_path,
        target_source=target_source,
        warning=warning,
        policy_order=policy_order,
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        targets_policy_order=targets,
    )


def load_sdk_offset_file(
    offset_path: str | Path,
    sdk_joint_order: tuple[str, ...] | list[str],
) -> np.ndarray:
    """Load tuned SDK-order offsets from a YAML file."""
    path = Path(offset_path)
    with path.open("r", encoding="utf-8") as f:
        data: Any = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"offset file must contain a YAML mapping: {path}")
    unit = str(data.get("unit", "rad")).lower()
    if unit not in {"rad", "radian", "radians"}:
        raise ValueError(f"offset file unit must be rad, got {unit!r}.")
    names = tuple(str(name) for name in data.get("joint_names") or [])
    expected = tuple(str(name) for name in sdk_joint_order)
    if names != expected:
        raise ValueError(
            "offset file joint_names must exactly match profile sdk_joint_order."
        )
    offsets = _as_vector(np.asarray(data.get("offsets"), dtype=np.float32), "offsets")
    return offsets.astype(np.float32)


def profile_with_sdk_offsets(
    profile: Revo3Profile,
    sdk_offsets: np.ndarray,
) -> Revo3Profile:
    """Return a copy of a Revo3Profile using tuned SDK-order offsets."""
    offsets = _as_vector(sdk_offsets, "sdk_offsets")
    policy_offsets = np.zeros(ACTION_DIM, dtype=np.float32)
    for sdk_index, policy_index in enumerate(profile.policy_to_sdk_perm):
        policy_offsets[int(policy_index)] = offsets[sdk_index]
    return replace(
        profile,
        sdk_offset_rad=offsets.copy(),
        policy_offset_rad=policy_offsets,
    )


def build_replay_plan(
    trajectory: ArtifactTrajectory,
    profile: Revo3Profile,
    *,
    start_frame: int = 0,
    max_replay_frames: int | None = None,
) -> ReplayPlan:
    if trajectory.policy_order != profile.policy_joint_order:
        raise ValueError("trajectory policy joint order differs from Revo3 profile.")
    if start_frame < 0 or start_frame >= trajectory.frame_count:
        raise ValueError(
            f"start_frame must be between 0 and {trajectory.frame_count - 1}."
        )
    if max_replay_frames is not None and int(max_replay_frames) <= 0:
        raise ValueError("max_replay_frames must be positive when provided.")

    end_frame = trajectory.frame_count
    if max_replay_frames is not None:
        end_frame = min(end_frame, start_frame + int(max_replay_frames))

    frames = tuple(
        _build_command_frame(
            trajectory,
            profile,
            dataset_index=index,
        )
        for index in range(start_frame, end_frame)
    )
    return ReplayPlan(
        trajectory=trajectory,
        profile_path=profile.path,
        start_frame=int(start_frame),
        frames=frames,
    )


def format_preview_table(
    frame: ReplayCommandFrame,
    profile: Revo3Profile,
    current_policy_order: np.ndarray | None = None,
) -> str:
    current_sdk = None
    if current_policy_order is not None:
        current = _as_vector(current_policy_order, "current_policy_order")
        current_sdk = current[profile.policy_to_sdk_perm] + profile.sdk_offset_rad

    headers = (
        "sdk_idx",
        "policy_idx",
        "joint",
        "target_policy",
        "offset",
        "final_sdk",
        "current_sdk",
        "delta",
        "limits",
        "status",
    )
    rows: list[tuple[str, ...]] = []
    clipped_by_index = {issue.sdk_index: issue for issue in frame.clipped}
    for sdk_index, joint_name in enumerate(profile.sdk_joint_order):
        policy_index = int(profile.policy_to_sdk_perm[sdk_index])
        current_value = None if current_sdk is None else float(current_sdk[sdk_index])
        delta = None if current_value is None else frame.command_sdk_order[sdk_index] - current_value
        if sdk_index in clipped_by_index:
            status = "CLIPPED"
        else:
            status = "OK"
        rows.append(
            (
                str(sdk_index),
                str(policy_index),
                joint_name,
                _fmt(frame.target_policy_order[policy_index]),
                _fmt(profile.sdk_offset_rad[sdk_index]),
                _fmt(frame.command_sdk_order[sdk_index]),
                _fmt_optional(current_value),
                _fmt_optional(delta),
                f"[{_fmt(profile.joint_lower_sdk[sdk_index])},"
                f"{_fmt(profile.joint_upper_sdk[sdk_index])}]",
                status,
            )
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render(items: tuple[str, ...]) -> str:
        return "  ".join(item.rjust(widths[index]) for index, item in enumerate(items))

    lines = [render(headers), render(tuple("-" * width for width in widths))]
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)


def replay_plan_to_arrays(plan: ReplayPlan) -> dict[str, np.ndarray]:
    frames = plan.frames
    return {
        "dataset_index": np.asarray([frame.dataset_index for frame in frames], dtype=np.int64),
        "frame_index": np.asarray([frame.frame_index for frame in frames], dtype=np.int64),
        "timestamp_sec": np.asarray([frame.timestamp_sec for frame in frames], dtype=np.float64),
        "target_policy_order": np.asarray(
            [frame.target_policy_order for frame in frames],
            dtype=np.float32,
        ),
        "raw_final_sdk_order": np.asarray(
            [frame.raw_final_sdk_order for frame in frames],
            dtype=np.float32,
        ),
        "command_sdk_order": np.asarray(
            [frame.command_sdk_order for frame in frames],
            dtype=np.float32,
        ),
        "command_policy_order": np.asarray(
            [frame.command_policy_order for frame in frames],
            dtype=np.float32,
        ),
        "clipped_count": np.asarray(
            [len(frame.clipped) for frame in frames],
            dtype=np.int64,
        ),
    }


def _build_command_frame(
    trajectory: ArtifactTrajectory,
    profile: Revo3Profile,
    *,
    dataset_index: int,
) -> ReplayCommandFrame:
    target_policy = _as_vector(
        trajectory.targets_policy_order[dataset_index],
        "target_policy_order",
    )
    target_sdk = target_policy[profile.policy_to_sdk_perm]
    raw_final_sdk = target_sdk + profile.sdk_offset_rad
    command_sdk = np.clip(
        raw_final_sdk,
        profile.joint_lower_sdk,
        profile.joint_upper_sdk,
    )
    clipped: list[LimitIssue] = []

    for sdk_index, (value, lower, upper) in enumerate(
        zip(raw_final_sdk, profile.joint_lower_sdk, profile.joint_upper_sdk, strict=True)
    ):
        if lower <= value <= upper:
            continue
        clamped_value = float(min(max(value, lower), upper))
        policy_index = int(profile.policy_to_sdk_perm[sdk_index])
        clipped.append(
            LimitIssue(
                sdk_index=sdk_index,
                policy_index=policy_index,
                joint_name=profile.sdk_joint_order[sdk_index],
                value_rad=float(value),
                lower_rad=float(lower),
                upper_rad=float(upper),
                clamped_rad=clamped_value,
                reason="final_output_clip",
            )
        )

    command_policy = _sdk_command_to_policy_target(command_sdk, profile)
    return ReplayCommandFrame(
        dataset_index=int(dataset_index),
        frame_index=int(trajectory.frame_index[dataset_index]),
        timestamp_sec=float(trajectory.timestamp_sec[dataset_index]),
        target_policy_order=target_policy.astype(np.float32),
        raw_final_sdk_order=raw_final_sdk.astype(np.float32),
        command_sdk_order=command_sdk.astype(np.float32),
        command_policy_order=command_policy.astype(np.float32),
        clipped=tuple(clipped),
    )


def _sdk_command_to_policy_target(
    command_sdk_order: np.ndarray,
    profile: Revo3Profile,
) -> np.ndarray:
    command_sdk = _as_vector(command_sdk_order, "command_sdk_order")
    target_policy = np.zeros(ACTION_DIM, dtype=np.float32)
    for sdk_index, policy_index in enumerate(profile.policy_to_sdk_perm):
        target_policy[int(policy_index)] = (
            command_sdk[sdk_index] - profile.sdk_offset_rad[sdk_index]
        )
    return target_policy


def _require_array(
    arrays: dict[str, np.ndarray],
    key: str,
    *,
    ndim: int,
    trailing_dim: int | None = None,
) -> np.ndarray:
    if key not in arrays:
        raise ValueError(f"replay_dataset.npz missing required array {key!r}.")
    array = arrays[key]
    if array.ndim != ndim:
        raise ValueError(f"{key} has shape {array.shape}; expected {ndim} dims.")
    if trailing_dim is not None and array.shape[-1] != trailing_dim:
        raise ValueError(
            f"{key} has shape {array.shape}; expected trailing dim {trailing_dim}."
        )
    return array


def _require_joint_order(
    arrays: dict[str, np.ndarray],
    expected_order: tuple[str, ...],
) -> tuple[str, ...]:
    if "joint_names_policy_order" not in arrays:
        raise ValueError("replay_dataset.npz missing joint_names_policy_order.")
    order = tuple(str(item) for item in arrays["joint_names_policy_order"].tolist())
    if len(order) != len(expected_order) or len(set(order)) != len(order):
        raise ValueError("joint_names_policy_order must contain 21 unique joints.")
    return order


def _as_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (ACTION_DIM,):
        raise ValueError(f"{name} must have shape ({ACTION_DIM},), got {vector.shape}.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains NaN/Inf.")
    return vector


def _fmt(value: float | np.floating) -> str:
    return f"{float(value):.6f}"


def _fmt_optional(value: float | np.floating | None) -> str:
    if value is None:
        return "N/A"
    return _fmt(value)
