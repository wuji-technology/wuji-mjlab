#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Shadow-ready Revo3 real-hand policy loop.

Default mode is shadow: read hardware + vision, run the real 216-D policy, and
record the processed targets without sending motor commands.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from deploy.reorient.lib.real_hand_env import RealHandEnv
from deploy.reorient.lib.real_hand_env_cfg import make_real_hand_env_cfg
from deploy.reorient.lib.revo3_hand_driver import (
    MockRevo3HandDriver,
    Revo3SdkConfig,
    Revo3SdkHandDriver,
)
from deploy.reorient.lib.revo3_policy_artifact import Revo3PolicyArtifact
from deploy.reorient.lib.revo3_profile import (
    Revo3CubePoseAdapter,
    Revo3GoalPoseAdapter,
    Revo3Profile,
)
from deploy.reorient.lib.zmq_bridge import CubeReceiver, GoalReceiver

DEFAULT_PROFILE = Path(__file__).resolve().parents[1] / "config" / "revo3_right.yaml"


class _GoalStub:
    def __init__(self, quat_wxyz: np.ndarray) -> None:
        self._quat = _normalize_quat_wxyz(quat_wxyz)

    def latest(self) -> np.ndarray:
        return self._quat.copy()


def _normalize_quat_wxyz(quat) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise argparse.ArgumentTypeError("quaternion has zero norm")
    q = q / norm
    if q[0] < 0.0:
        q = -q
    return q


def _parse_quat_wxyz(text: str) -> np.ndarray:
    parts = [float(item) for item in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--goal-quat expects W,X,Y,Z")
    return _normalize_quat_wxyz(parts)


def _fixed_goal_from_manifest(artifact: Revo3PolicyArtifact) -> np.ndarray:
    fixed = artifact.manifest.get("fixed_goal") or {}
    quat = fixed.get("quat_tag_wxyz")
    if quat is None:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return _normalize_quat_wxyz(quat)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact-dir", required=True, help="Revo3 policy artifact directory.")
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE), help="Revo3 profile YAML.")
    parser.add_argument(
        "--mode",
        choices=["dry-run", "shadow", "active"],
        default="shadow",
        help="dry-run uses a mock hand; shadow reads hardware but writes no commands.",
    )
    parser.add_argument("--duration", type=float, default=1.0, help="Run duration in seconds.")
    parser.add_argument("--cube-port", type=int, default=5555)
    parser.add_argument("--goal-port", type=int, default=5556)
    parser.add_argument("--no-cube-zmq", action="store_true")
    parser.add_argument("--goal-mode", choices=["fixed", "external"], default="fixed")
    parser.add_argument("--goal-quat", type=_parse_quat_wxyz, default=None)
    parser.add_argument("--log-file", default=None, help="Optional .npz shadow log path.")
    parser.add_argument("--use-gpu", action="store_true", help="Use CUDA ONNX Runtime provider.")
    parser.add_argument("--port", default=None, help="Revo3 serial port; omit for SDK auto-detect.")
    parser.add_argument("--baudrate", type=int, default=None)
    parser.add_argument("--slave-id", type=int, default=None)
    parser.add_argument("--kp", type=float, default=None)
    parser.add_argument("--kd", type=float, default=None)
    parser.add_argument("--effort-ma", type=float, default=None)
    parser.add_argument(
        "--i-understand-active-revo3-control",
        action="store_true",
        help="Required with --mode active. Shadow/dry-run do not need this.",
    )
    return parser


def _build_driver(args, profile: Revo3Profile):
    if args.mode == "dry-run":
        return MockRevo3HandDriver(profile, track_targets=True)

    sdk_cfg = profile.sdk
    return Revo3SdkHandDriver(
        profile,
        Revo3SdkConfig(
            port=args.port,
            baudrate=int(args.baudrate or sdk_cfg.get("baudrate", 5000000)),
            slave_id=int(args.slave_id or sdk_cfg.get("slave_id", 126)),
            auto_detect=args.port is None and bool(sdk_cfg.get("auto_detect", True)),
        ),
        command_mode=args.mode,
        allow_active=args.i_understand_active_revo3_control,
        kp=args.kp,
        kd=args.kd,
        effort_ma=args.effort_ma,
    )


def _write_log(path: str | Path, records: list[dict]) -> None:
    if not records:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = records[0].keys()
    arrays = {key: np.asarray([row[key] for row in records]) for key in keys}
    np.savez(out, **arrays)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    artifact = Revo3PolicyArtifact.load(args.artifact_dir, use_gpu=args.use_gpu)
    profile = Revo3Profile.load(
        args.profile,
        expected_policy_joint_order=artifact.policy_joint_order,
    )

    if args.mode == "active":
        if not args.i_understand_active_revo3_control:
            print("ERROR: --mode active requires --i-understand-active-revo3-control")
            return 2
        if not profile.active_control_allowed():
            print(
                "ERROR: profile vision_to_policy_frame is not calibrated; "
                "active Revo3 control is refused."
            )
            return 2

    if not profile.vision_transform_calibrated:
        print(
            "WARNING: profile vision_to_policy_frame.calibrated=false. "
            "Dry-run/shadow may proceed, but active control is disabled."
        )

    golden_path = Path(args.artifact_dir) / "golden_io.npz"
    if golden_path.exists():
        artifact.validate_golden_io()
        print("[artifact] golden_io raw-input ONNX check passed.")
    else:
        print("[artifact] golden_io.npz not found; skipped raw-input ONNX regression check.")

    cfg = make_real_hand_env_cfg(
        robot_variant="revo3_right",
        policy_config=artifact.env_policy_config,
    )
    cfg.observations["policy"].enable_corruption = False

    cube_recv = None
    cube_source = None
    if not args.no_cube_zmq:
        cube_recv = CubeReceiver(port=args.cube_port)
        cube_source = Revo3CubePoseAdapter(cube_recv, profile)

    if args.goal_mode == "external":
        goal_recv = GoalReceiver(port=args.goal_port)
        goal_source = Revo3GoalPoseAdapter(goal_recv, profile)
    else:
        goal_quat = args.goal_quat if args.goal_quat is not None else _fixed_goal_from_manifest(artifact)
        goal_source = _GoalStub(goal_quat)

    print("=" * 72)
    print(f"Revo3 deploy loop mode={args.mode} duration={args.duration:.2f}s")
    print(f"artifact={Path(args.artifact_dir).resolve()}")
    print(f"profile={Path(args.profile).resolve()}")
    print("policy_io=raw obs[216] -> actions[21], no external normalization")
    print("=" * 72)

    records: list[dict] = []
    driver = _build_driver(args, profile)
    with driver as drv:
        env = RealHandEnv(cfg=cfg, hand_driver=drv)
        if cube_source is not None:
            env._cube_zmq = cube_source
        env._goal_zmq = goal_source

        obs, _ = env.reset()
        print(f"[reset] obs['policy'] shape={tuple(obs['policy'].shape)}")

        start = time.perf_counter()
        step_idx = 0
        while time.perf_counter() - start < max(args.duration, 0.0):
            obs_vec = obs["policy"][0].cpu().numpy()
            action_raw = artifact(obs_vec)
            action_clipped = np.clip(action_raw, -1.0, 1.0).astype(np.float32)
            obs, *_ = env.step(torch.from_numpy(action_clipped).float().unsqueeze(0))

            target = getattr(drv, "last_target_policy", None)
            if target is None:
                target = np.full(21, np.nan, dtype=np.float32)
            joint_pos = env.scene["robot"].data.joint_pos[0].detach().cpu().numpy()
            records.append(
                {
                    "timestamp_sec": time.perf_counter() - start,
                    "step": step_idx,
                    "policy_input_raw": obs_vec.astype(np.float32),
                    "action_raw": action_raw.astype(np.float32),
                    "action_clipped": action_clipped,
                    "target_policy_order": np.asarray(target, dtype=np.float32),
                    "joint_pos_policy_order": joint_pos.astype(np.float32),
                }
            )
            step_idx += 1

    if args.log_file:
        _write_log(args.log_file, records)
        print(f"[log] wrote {len(records)} steps to {args.log_file}")

    print(f"[done] steps={len(records)} mode={args.mode} hardware_writes={'yes' if args.mode == 'active' else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
