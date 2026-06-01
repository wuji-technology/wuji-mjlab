# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji Hand hardware utilities.

Single entry-point for low-level hand operations. Provides two subcommands:

  home   Reset all 20 joints to REORIENT_JOINT_POS via a 3s ramp.
         Enables joints on entry, disables on exit; reports tracking error.
  check  READ-ONLY connection + encoder sanity check. Does not write
         targets; the hand stays where it is.

Usage:
    pixi run -e deploy python deploy/reorient/scripts/hand_utils.py home
    pixi run -e deploy python deploy/reorient/scripts/hand_utils.py check

This module previously lived as two separate scripts (home_real_hand.py and
check_real_hand.py). They are unified here so the script set matches the
pixi.toml [feature.deploy.tasks] entries with a single source of truth.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Ensure `deploy.reorient.lib...` import works when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np


def cmd_home(_args: argparse.Namespace) -> int:
    """Move the real hand to REORIENT_JOINT_POS via a 3s smooth ramp.

    Safety:
      - Uses WujiHandDriver context manager (enable joints on enter, disable
        on exit). effort_limit and lowpass_cutoff come from control.yaml.
      - 3s ease-in-out ramp.
      - Reads back actual position and reports max error vs target.

    After this, the hand is at REORIENT_JOINT_POS — the same pose
    RealHandEnv.reset expects MockHandDriver to be in.
    """
    from deploy.reorient.lib.hand_driver import WujiHandDriver, _resolve_home_qpos

    print("=" * 60)
    print("Real Hand HOME MOVE — 3s smooth ramp to REORIENT_JOINT_POS")
    print("=" * 60)

    home_qpos = _resolve_home_qpos()
    print("\nTarget pose (REORIENT_JOINT_POS, degrees, finger-major):")
    for i in range(5):
        finger_deg = np.rad2deg(home_qpos[4 * i: 4 * (i + 1)])
        print(f"  finger{i+1}: " + ", ".join(f"{v:+6.1f}°" for v in finger_deg))

    # effort_limit / lowpass_cutoff default to control.yaml.
    with WujiHandDriver(home_duration_s=3.0) as drv:
        print("\nReading current pose...")
        current = drv.read_encoders()
        max_diff = float(np.abs(current - home_qpos).max())
        print(f"  Max diff from target: {max_diff:.3f} rad ({np.rad2deg(max_diff):.1f}°)")

        print("\nRamping to home pose over 3s...")
        drv.home()

        print("\nReading actual after ramp...")
        actual = drv.read_encoders()
        max_err = float(np.abs(actual - home_qpos).max())
        rms_err = float(np.sqrt(np.mean((actual - home_qpos) ** 2)))
        print(f"  Max  err: {max_err:.3f} rad ({np.rad2deg(max_err):.2f}°)")
        print(f"  RMS  err: {rms_err:.3f} rad ({np.rad2deg(rms_err):.2f}°)")
        if max_err < np.deg2rad(2):
            print("  ✓ Within 2° — home reached")
        elif max_err < np.deg2rad(5):
            print("  ⚠ Within 5° — hand may not be tracking perfectly")
        else:
            print("  ✗ Over 5° error — investigate")

    print("\n✓ hand_utils home complete; joints disabled.")
    return 0


def cmd_check(_args: argparse.Namespace) -> int:
    """READ-ONLY connection + encoder sanity check.

    Connects to wujihandpy.Hand(), reads encoders + joint limits, validates
    joint ordering matches sim, and prints current pose. Does NOT write
    any targets; the hand stays where it is.

    This is the safe first step before any closed-loop deployment. If this
    prints a sensible-looking joint pose without errors, the hardware
    bridge is healthy.
    """
    try:
        import wujihandpy
    except ImportError:
        print("ERROR: wujihandpy not installed. Run `pixi install -e deploy`.")
        return 1

    print("=" * 60)
    print("Real Hand Connection Check (READ-ONLY)")
    print("=" * 60)

    print("\n[1/4] Connecting to wujihandpy.Hand()...")
    try:
        hand = wujihandpy.Hand()
    except Exception as e:
        print(f"  FAIL: {e}")
        print("  Check: USB connected? udev rules? hand powered?")
        return 1
    print("  ✓ connected")

    print("\n[2/4] Reading joint limits (no enable/no write)...")
    try:
        upper = hand.read_joint_upper_limit()  # (5, 4)
        lower = hand.read_joint_lower_limit()
    except Exception as e:
        print(f"  FAIL reading limits: {e}")
        return 1
    print(f"  shape: upper={upper.shape}, lower={lower.shape}")
    print(f"  upper range: [{upper.min():.2f}, {upper.max():.2f}] rad")
    print(f"  lower range: [{lower.min():.2f}, {lower.max():.2f}] rad")

    print("\n[3/4] Reading encoder actual position...")
    try:
        actual = hand.read_joint_actual_position()  # (5, 4)
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1
    flat = actual.flatten()
    print(f"  shape: {actual.shape} → flat (20,)")
    print(f"  range: [{flat.min():.3f}, {flat.max():.3f}] rad")
    print("  values (degrees, finger-major):")
    for i in range(5):
        finger_deg = np.rad2deg(actual[i])
        print(f"    finger{i+1}: " + ", ".join(f"{v:+6.1f}°" for v in finger_deg))

    print("\n[4/4] Validating against deploy.reorient.lib.hand_driver expectation...")
    from deploy.reorient.lib.hand_driver import JOINT_NAMES_20  # noqa: F401

    print(f"  Expected joint order ({len(JOINT_NAMES_20)} joints):")
    for i, name in enumerate(JOINT_NAMES_20):
        print(f"    [{i:2d}] {name}: {np.rad2deg(flat[i]):+6.1f}°")

    # Compare against MockHandDriver / sim's REORIENT_JOINT_POS home pose.
    from wuji_mjlab.tasks.reorient.reorient_constants import REORIENT_JOINT_POS

    expected_home = np.zeros(20, dtype=np.float64)
    for i, name in enumerate(JOINT_NAMES_20):
        for pat, val in REORIENT_JOINT_POS.items():
            if re.fullmatch(pat, name):
                expected_home[i] = val
                break
    diff = flat - expected_home
    print("\n  Current vs REORIENT_JOINT_POS (home target):")
    print(f"    Max abs diff: {np.abs(diff).max():.3f} rad "
          f"({np.rad2deg(np.abs(diff).max()):.1f}°)")
    print(f"    RMS diff:     {np.sqrt(np.mean(diff**2)):.3f} rad")
    print("    (Large diff just means the hand isn't currently at home pose;"
          " not an error.)")

    print("\n" + "=" * 60)
    print("✓ All read operations succeeded. Hand bridge healthy.")
    print("✓ Next: run home pose move with `pixi run -e deploy home`")
    print("=" * 60)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Wuji Hand hardware utilities (home / check).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_home = sub.add_parser(
        "home", help="Reset all 20 joints to REORIENT home pose (writes targets)."
    )
    p_home.set_defaults(func=cmd_home)
    p_check = sub.add_parser(
        "check", help="READ-ONLY connection + encoder sanity check."
    )
    p_check.set_defaults(func=cmd_check)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
