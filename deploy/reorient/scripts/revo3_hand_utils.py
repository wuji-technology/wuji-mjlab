#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Revo3 hardware utilities.

``check`` is read-only. ``home`` writes commands and requires an explicit
active-control acknowledgement.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from deploy.reorient.lib.revo3_hand_driver import Revo3SdkConfig, Revo3SdkHandDriver
from deploy.reorient.lib.revo3_profile import Revo3Profile

DEFAULT_PROFILE = Path(__file__).resolve().parents[1] / "config" / "revo3_right.yaml"


def _sdk_config_from_args(args, profile: Revo3Profile) -> Revo3SdkConfig:
    sdk = profile.sdk
    return Revo3SdkConfig(
        port=args.port,
        baudrate=int(args.baudrate or sdk.get("baudrate", 5000000)),
        slave_id=int(args.slave_id or sdk.get("slave_id", 126)),
        auto_detect=args.port is None and bool(sdk.get("auto_detect", True)),
    )


def cmd_check(args: argparse.Namespace) -> int:
    profile = Revo3Profile.load(args.profile)
    with Revo3SdkHandDriver(
        profile,
        _sdk_config_from_args(args, profile),
        command_mode="shadow",
    ) as drv:
        qpos = drv.read_encoders()

    print("=" * 72)
    print("Revo3 hardware check (read-only)")
    print(f"profile={Path(args.profile).resolve()}")
    print(f"joints={len(profile.policy_joint_order)}")
    print(f"position range rad=[{qpos.min():+.4f}, {qpos.max():+.4f}]")
    print("policy-order joint positions:")
    for idx, name in enumerate(profile.policy_joint_order):
        print(f"  [{idx:02d}] {name}: {qpos[idx]:+.5f} rad ({np.rad2deg(qpos[idx]):+.2f} deg)")
    print("=" * 72)
    print("OK: read-only Revo3 SDK check completed; no commands were sent.")
    return 0


def cmd_home(args: argparse.Namespace) -> int:
    if not args.i_understand_active_revo3_control:
        print("ERROR: home writes motor commands and requires --i-understand-active-revo3-control")
        return 2

    profile = Revo3Profile.load(args.profile)
    with Revo3SdkHandDriver(
        profile,
        _sdk_config_from_args(args, profile),
        command_mode="active",
        allow_active=True,
        home_duration_s=args.duration,
    ) as drv:
        print(f"[home] moving to configured Revo3 home over {args.duration:.1f}s ...")
        drv.home()
        qpos = drv.read_encoders()
    err = np.abs(qpos - profile.home_joint_pos_policy)
    print(f"[home] max error {float(err.max()):.5f} rad ({np.rad2deg(float(err.max())):.2f} deg)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--port", default=None)
    parser.add_argument("--baudrate", type=int, default=None)
    parser.add_argument("--slave-id", type=int, default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser("check", help="Read-only Revo3 SDK and encoder check.")
    check.set_defaults(func=cmd_check)

    home = sub.add_parser("home", help="Move to configured home pose; writes commands.")
    home.add_argument("--duration", type=float, default=3.0)
    home.add_argument("--i-understand-active-revo3-control", action="store_true")
    home.set_defaults(func=cmd_home)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
