#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji Hand deploy preflight — check deps, SDKs, and policy before a run."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from wuji_reorient_deploy.mvs_sdk import find_mvs_python_path, mvs_python_path

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


class Check:
  def __init__(self, name: str, ok: bool, detail: str, required: bool, hint: str = ""):
    self.name, self.ok, self.detail, self.required, self.hint = (
      name, ok, detail, required, hint)


def _dep(module: str, required: bool, hint: str = "") -> Check:
  try:
    m = importlib.import_module(module)
    return Check(module, True, getattr(m, "__version__", "ok"), required)
  except Exception as e:  # noqa: BLE001 — report any import failure
    return Check(module, False, f"{type(e).__name__}", required, hint)


def _check_mvs() -> Check:
  path = find_mvs_python_path()
  if path is None:
    return Check(
      "MvImport (MVS)", False, f"not found at {mvs_python_path()}/MvImport", False,
      "install Hikvision MVS SDK (hikrobotics.com) or set MVS_PYTHON_PATH")
  return Check("MvImport (MVS)", True, path, False)


def _check_policy(ckpt: str, gen: int, hand: str) -> Check:
  onnx = Path(ckpt)
  if not onnx.exists():
    return Check(
      f"policy ({hand})", False, f"missing {onnx}", True,
      "pass an existing --ckpt path; export one with: pixi run python -m "
      "wuji_mjlab.tasks.reorient.scripts.export_onnx <model.pt> --filename "
      f"policy_hand{gen}_{hand}.onnx  (its config sidecar must sit beside it)")
  try:
    from wuji_reorient_deploy.constants import load_constants
    from wuji_reorient_deploy.onnx_policy import ONNXPolicy
    const = load_constants(gen, hand)
    p = ONNXPolicy(str(onnx), const)
    cfg = p.config
    return Check(
      f"policy ({hand})", True,
      f"IO {const.obs_dim}->{const.action_dim}, history_len={cfg.get('history_len')}, "
      f"control_mode={cfg.get('control_mode')}", True)
  except Exception as e:  # noqa: BLE001
    return Check(f"policy ({hand})", False, f"{type(e).__name__}: {e}", True,
                 "obs/action contract mismatch — re-export from this branch")


def _check_connect(
  gen: int, hand: str, sn: str | None, *, allow_energize: bool = False
) -> Check:
  if gen == 1 and not allow_energize:
    return Check(
      f"hand connect ({hand})", False,
      "refusing to connect: generation 1 driver energizes motors", True,
      "pass --allow-energize to confirm motor power before connecting")
  if gen == 1:
    print("WARNING: generation 1 connection will energize motors", file=sys.stderr,
          flush=True)
  try:
    from wuji_reorient_deploy.constants import load_constants
    from wuji_reorient_deploy.drivers import make_driver
    const = load_constants(gen, hand)
    drv = make_driver(
      gen, hand, sn, mock=False, constants=const, energize=gen == 1
    )
    try:
      q = drv.read_encoders()
      fault = drv.fault()
      missing_feedback = getattr(drv, "missing_feedback", None)
      missing = missing_feedback() if missing_feedback is not None else ()
      energized = getattr(drv, "energized", True)
      ok = q.shape == (const.num_joints,) and fault is None and not missing
      detail = (
        f"{const.num_joints - len(missing)}/{const.num_joints} joints reporting fresh "
        f"feedback, fault={fault or 'none'}"
        + (f", MISSING: {', '.join(missing)}" if missing else "")
        + (
          "; ⚠ motors WERE energized — this driver has no read-only "
          "connect mode"
          if energized
          else "; motors NOT energized (read-only connect)"
        )
      )
      mode = "energized" if energized else "read-only"
      return Check(f"hand connect ({mode}, {hand})", ok, detail, True,
                   "" if ok else "hand reports a fault, missing feedback, or wrong joint count")
    finally:
      drv.close()
  except Exception as e:  # noqa: BLE001
    return Check(f"hand connect ({hand})", False, f"{type(e).__name__}: {e}",
                 True, "check hand is powered, on the bus, and --sn is correct")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--gen", type=int, choices=(1, 2), default=2,
                 help="hand generation (default: 2)")
  ap.add_argument("--hand", choices=["right"], default="right")
  ap.add_argument("--ckpt", required=True,
                  help="path to the ONNX policy to validate (no default location; "
                       "its config sidecar must sit beside it)")
  ap.add_argument("--sn", default=None, help="hand serial (default: auto-connect)")
  ap.add_argument("--connect", action="store_true",
                  help="connect and read encoders (gen2 read-only; gen1 requires --allow-energize)")
  ap.add_argument("--allow-energize", action="store_true",
                  help="explicitly permit gen1 --connect to energize motors")
  ap.add_argument("--need-camera", action="store_true",
                  help="treat a missing MVS camera SDK as a failure")
  args = ap.parse_args()

  driver_sdk = "wujihandpy" if args.gen == 1 else "wuji_sdk"
  checks: list[Check] = [
    _dep("numpy", True),
    _dep("onnxruntime", True),
    _dep("cv2", True, "pixi install -e reorient-deploy (opencv-contrib-python)"),
    _dep("pupil_apriltags", True, "pixi install -e reorient-deploy"),
    _dep("zmq", True, "pixi install -e reorient-deploy (pyzmq)"),
    _dep("yaml", True, "pixi install -e reorient-deploy (pyyaml)"),
    _dep("scipy", True),
    _check_policy(args.ckpt, args.gen, args.hand),
    _dep(driver_sdk, args.connect,
         "driver SDK for this --gen; `pixi install -e reorient-deploy` on a "
         "glibc>=2.34 host installs it (not needed for --mock)"),
    _check_mvs() if not args.need_camera else _need(_check_mvs()),
  ]
  if args.connect:
    checks.append(_check_connect(args.gen, args.hand, args.sn,
                                 allow_energize=args.allow_energize))

  print(f"\nWuji Hand deploy preflight  (gen={args.gen}, hand={args.hand}, "
        f"mode={'real' if args.connect else 'offline/mock'})\n")
  width = max(len(c.name) for c in checks)
  failed_required = 0
  for c in checks:
    if c.ok:
      mark = f"{GREEN}PASS{RESET}"
    elif c.required:
      mark = f"{RED}FAIL{RESET}"
      failed_required += 1
    else:
      mark = f"{YELLOW}n/a {RESET}"
    line = f"  [{mark}] {c.name:<{width}}  {c.detail}"
    print(line)
    if not c.ok and c.hint:
      print(f"         {DIM}↳ {c.hint}{RESET}")

  print()
  if failed_required:
    print(f"{RED}✗ {failed_required} required check(s) failed — not ready to "
          f"run.{RESET}\n")
    sys.exit(1)
  note = "" if args.connect else "  (run with --connect to test real hardware)"
  print(f"{GREEN}✓ ready to run{RESET}{note}\n")


def _need(c: Check) -> Check:
  c.required = True
  return c


if __name__ == "__main__":
  main()
