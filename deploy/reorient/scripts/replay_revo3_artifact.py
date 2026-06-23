#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Replay saved Revo3 artifact trajectory targets.

Default mode is dry-run: load replay_dataset.npz and preview the clipped
SDK-order commands without opening the hardware SDK.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import yaml
from deploy.reorient.lib.revo3_artifact_trajectory import (
    ReplayPlan,
    build_replay_plan,
    format_preview_table,
    load_artifact_trajectory,
    load_sdk_offset_file,
    profile_with_sdk_offsets,
    replay_plan_to_arrays,
)
from deploy.reorient.lib.revo3_hand_driver import Revo3SdkConfig, Revo3SdkHandDriver
from deploy.reorient.lib.revo3_profile import Revo3Profile

DEFAULT_PROFILE = Path(__file__).resolve().parents[1] / "config" / "revo3_right.yaml"
DEFAULT_OFFSET = (
    Path(__file__).resolve().parents[1] / "config" / "revo3_right_offset_tuned.yaml"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE), type=Path)
    parser.add_argument("--replay-data", default=None, type=Path)
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Optional debug output directory. Omit to avoid writing replay logs.",
    )
    parser.add_argument(
        "--offset-file",
        default=DEFAULT_OFFSET,
        type=Path,
        help="Tuned SDK-order offset YAML. Defaults to deploy/reorient/config/revo3_right_offset_tuned.yaml.",
    )
    parser.add_argument(
        "--no-offset-file",
        dest="offset_file",
        action="store_const",
        const=None,
        help="Use sim2real_joint_offset from --profile instead of a tuned offset YAML.",
    )
    parser.add_argument(
        "--mode",
        choices=("dry-run", "shadow", "active"),
        default="dry-run",
        help="dry-run never opens SDK; shadow reads encoders only; active writes targets.",
    )
    parser.add_argument("--replay-start-frame", type=int, default=0)
    parser.add_argument("--max-replay-frames", type=int, default=None)
    parser.add_argument("--rate-scale", type=float, default=1.0)
    parser.add_argument("--preposition-only", action="store_true")
    parser.add_argument("--preposition-duration", type=float, default=5.0)
    parser.add_argument("--hold-before-replay", type=float, default=3.0)
    parser.add_argument("--preposition-rate", type=float, default=50.0)
    parser.add_argument("--port", default=None)
    parser.add_argument("--baudrate", type=int, default=None)
    parser.add_argument("--slave-id", type=int, default=None)
    parser.add_argument("--kp", type=float, default=None)
    parser.add_argument("--kd", type=float, default=None)
    parser.add_argument("--effort-ma", type=float, default=None)
    parser.add_argument(
        "--i-understand-active-revo3-control",
        action="store_true",
        help="Required with --mode active. Active mode can move real hardware.",
    )
    return parser


def _build_driver(args: argparse.Namespace, profile: Revo3Profile) -> Revo3SdkHandDriver:
    sdk_cfg = profile.sdk
    return Revo3SdkHandDriver(
        profile,
        Revo3SdkConfig(
            port=args.port,
            baudrate=int(args.baudrate or sdk_cfg.get("baudrate", 5000000)),
            slave_id=int(args.slave_id or sdk_cfg.get("slave_id", 126)),
            auto_detect=args.port is None and bool(sdk_cfg.get("auto_detect", True)),
        ),
        command_mode="active" if args.mode == "active" else "shadow",
        allow_active=args.mode == "active" and args.i_understand_active_revo3_control,
        kp=args.kp,
        kd=args.kd,
        effort_ma=args.effort_ma,
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.mode == "active" and not args.i_understand_active_revo3_control:
        raise ValueError("--mode active requires --i-understand-active-revo3-control.")
    if args.rate_scale <= 0.0:
        raise ValueError("--rate-scale must be positive.")
    if args.preposition_duration <= 0.0:
        raise ValueError("--preposition-duration must be positive.")
    if args.hold_before_replay < 0.0:
        raise ValueError("--hold-before-replay must be >= 0.")
    if args.preposition_rate <= 0.0:
        raise ValueError("--preposition-rate must be positive.")
    if args.preposition_only and args.mode != "active":
        raise ValueError("--preposition-only is only meaningful with --mode active.")


def _print_preview(
    plan: ReplayPlan,
    profile: Revo3Profile,
    current_policy_order: np.ndarray | None,
    *,
    offset_source: str,
) -> None:
    first = plan.frames[0]
    print("\nReplay-start-frame preview")
    print(f"  artifact_dir: {plan.trajectory.artifact_dir}")
    print(f"  replay_data: {plan.trajectory.replay_path}")
    print(f"  target_source: {plan.trajectory.target_source}")
    print(f"  replay_start_frame: {plan.start_frame}")
    print(f"  source_frame_index: {first.frame_index}")
    print(f"  timestamp_sec: {first.timestamp_sec:.6f}")
    print(f"  selected_frames: {len(plan.frames)}")
    print(
        "  selected_duration_sec: "
        f"{plan.frames[-1].timestamp_sec - first.timestamp_sec:.6f}"
    )
    print(f"  offset_source: {offset_source}")
    print(f"  clipped_values_in_selected_frames: {plan.clipped_count}")
    print("  conversion: policy order -> sdk_joint_order -> + sim2real offset -> clip to joint limits")
    if plan.trajectory.warning:
        print(f"  warning: {plan.trajectory.warning}")
    if current_policy_order is None:
        print("  current_encoder_state: unavailable")
    print(format_preview_table(first, profile, current_policy_order))
    if first.clipped:
        print("  first_frame_clipped:")
        for issue in first.clipped:
            print(
                f"    {issue.joint_name}: {issue.value_rad:.6f} -> "
                f"{issue.clamped_rad:.6f} ({issue.reason})"
            )


def _write_outputs(
    output_dir: Path,
    plan: ReplayPlan,
    profile: Revo3Profile,
    *,
    mode: str,
    offset_source: str,
    current_policy_order: np.ndarray | None,
    hardware_writes: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = replay_plan_to_arrays(plan)
    arrays["policy_joint_order"] = np.asarray(profile.policy_joint_order, dtype=str)
    arrays["sdk_joint_order"] = np.asarray(profile.sdk_joint_order, dtype=str)
    arrays["sdk_offset_rad"] = profile.sdk_offset_rad.astype(np.float32)
    if current_policy_order is not None:
        arrays["current_policy_order"] = np.asarray(current_policy_order, dtype=np.float32)
        arrays["current_sdk_order"] = (
            np.asarray(current_policy_order, dtype=np.float32)[profile.policy_to_sdk_perm]
            + profile.sdk_offset_rad
        ).astype(np.float32)

    log_path = output_dir / "revo3_artifact_replay_log.npz"
    summary_path = output_dir / "revo3_artifact_replay_summary.yaml"
    np.savez(log_path, **arrays)

    summary: dict[str, object] = {
        "artifact_dir": str(plan.trajectory.artifact_dir),
        "replay_data": str(plan.trajectory.replay_path),
        "profile": str(profile.path),
        "offset_source": offset_source,
        "mode": mode,
        "target_source": plan.trajectory.target_source,
        "target_source_warning": plan.trajectory.warning,
        "replay_start_frame": plan.start_frame,
        "selected_frames": len(plan.frames),
        "first_frame_index": plan.frames[0].frame_index,
        "first_timestamp_sec": plan.frames[0].timestamp_sec,
        "last_frame_index": plan.frames[-1].frame_index,
        "last_timestamp_sec": plan.frames[-1].timestamp_sec,
        "clipped_count": plan.clipped_count,
        "hardware_writes": hardware_writes,
        "output_log": str(log_path),
        "output_summary": str(summary_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    return log_path, summary_path


def _quintic(tau: float) -> float:
    tau = min(1.0, max(0.0, tau))
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def _sleep_until(target_time: float) -> None:
    while True:
        remaining = target_time - time.monotonic()
        if remaining <= 0.0:
            return
        time.sleep(min(remaining, 0.01))


def _preposition(
    driver: Revo3SdkHandDriver,
    current_policy: np.ndarray,
    target_policy: np.ndarray,
    *,
    duration: float,
    rate_hz: float,
) -> int:
    steps = max(1, int(round(duration * rate_hz)))
    period = duration / steps
    writes = 0
    start_time = time.monotonic()
    for step in range(steps):
        tau = (step + 1) / steps
        blend = _quintic(tau)
        command = current_policy + blend * (target_policy - current_policy)
        driver.write_target(command.astype(np.float32))
        writes += 1
        _sleep_until(start_time + (step + 1) * period)
    return writes


def _hold(
    driver: Revo3SdkHandDriver,
    target_policy: np.ndarray,
    *,
    duration: float,
    rate_hz: float,
) -> int:
    if duration <= 0.0:
        return 0
    period = 1.0 / rate_hz
    steps = max(1, int(round(duration * rate_hz)))
    start_time = time.monotonic()
    writes = 0
    for step in range(steps):
        driver.write_target(target_policy.astype(np.float32))
        writes += 1
        _sleep_until(start_time + (step + 1) * period)
    return writes


def _replay_active(
    driver: Revo3SdkHandDriver,
    plan: ReplayPlan,
    current_policy: np.ndarray,
    args: argparse.Namespace,
) -> int:
    first_target = plan.frames[0].command_policy_order
    print(
        "[active] prepositioning to replay-start frame "
        f"over {args.preposition_duration:.3f}s...",
        flush=True,
    )
    writes = _preposition(
        driver,
        current_policy,
        first_target,
        duration=args.preposition_duration,
        rate_hz=args.preposition_rate,
    )
    print(f"[active] preposition complete; writes={writes}.", flush=True)
    writes += _hold(
        driver,
        first_target,
        duration=args.hold_before_replay,
        rate_hz=args.preposition_rate,
    )
    if args.hold_before_replay > 0.0:
        print(
            "[active] start-frame hold complete; "
            f"duration={args.hold_before_replay:.3f}s.",
            flush=True,
        )
    if args.preposition_only:
        print("[active] preposition-only complete; trajectory frames were not replayed.")
        return writes

    origin_timestamp = plan.frames[0].timestamp_sec
    duration = plan.frames[-1].timestamp_sec - origin_timestamp
    print(
        "[active] replaying "
        f"{len(plan.frames)} frame(s) over {duration / args.rate_scale:.3f}s "
        f"(artifact duration {duration:.3f}s, rate_scale={args.rate_scale:g}).",
        flush=True,
    )
    start_time = time.monotonic()
    for frame in plan.frames:
        target_time = start_time + (frame.timestamp_sec - origin_timestamp) / args.rate_scale
        _sleep_until(target_time)
        driver.write_target(frame.command_policy_order.astype(np.float32))
        writes += 1
    print(
        f"[active] replay complete; replay_frames={len(plan.frames)}, "
        f"total_hardware_writes={writes}.",
        flush=True,
    )
    return writes


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        _validate_args(args)
        trajectory = load_artifact_trajectory(args.artifact_dir, args.replay_data)
        profile = Revo3Profile.load(
            args.profile,
            expected_policy_joint_order=trajectory.policy_order,
        )
        offset_source = "profile sim2real_joint_offset"
        if args.offset_file is not None:
            offsets = load_sdk_offset_file(args.offset_file, profile.sdk_joint_order)
            profile = profile_with_sdk_offsets(profile, offsets)
            offset_source = str(args.offset_file)
        plan = build_replay_plan(
            trajectory,
            profile,
            start_frame=args.replay_start_frame,
            max_replay_frames=args.max_replay_frames,
        )

        current_policy_order: np.ndarray | None = None
        hardware_writes = 0
        if args.mode == "dry-run":
            _print_preview(plan, profile, None, offset_source=offset_source)
            print("[dry-run] SDK was not opened; no hardware commands were sent.")
        else:
            with _build_driver(args, profile) as driver:
                current_policy_order = driver.read_encoders()
                _print_preview(
                    plan,
                    profile,
                    current_policy_order,
                    offset_source=offset_source,
                )
                if args.mode == "shadow":
                    print("[shadow] current encoders were read; no hardware commands were sent.")
                else:
                    if plan.clipped_count:
                        print(
                            "[active] selected artifact target(s) exceed configured "
                            "joint limits; final SDK commands will be clipped."
                        )
                    print("[active] hardware command mode enabled; replaying selected frames.")
                    hardware_writes = _replay_active(
                        driver,
                        plan,
                        current_policy_order,
                        args,
                    )

        if args.output_dir is not None:
            log_path, summary_path = _write_outputs(
                args.output_dir,
                plan,
                profile,
                mode=args.mode,
                offset_source=offset_source,
                current_policy_order=current_policy_order,
                hardware_writes=hardware_writes,
            )
            print(f"[log] {log_path}")
            print(f"[summary] {summary_path}")
        else:
            print("[log] no output files requested.")
        return 0
    except Exception as exc:
        print(f"revo3 artifact replay: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
