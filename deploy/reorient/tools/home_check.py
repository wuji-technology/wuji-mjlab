#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Enable the hand and hold it at the default (cage home) pose — to eyeball the pose."""
from __future__ import annotations

import argparse
import math
import time

from wuji_reorient_deploy import config_loader
from wuji_reorient_deploy.constants import load_constants
from wuji_reorient_deploy.drivers import make_driver


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--gen", type=int, choices=(1, 2), default=2,
                help="hand generation (default: 2)")
  p.add_argument("--hand", choices=["right"], required=True)
  p.add_argument("--sn", default=None, help="hand serial (default: control.yaml, else auto)")
  p.add_argument("--mock", action="store_true")
  p.add_argument("--kp", type=float, default=None)
  p.add_argument("--kd", type=float, default=None)
  p.add_argument("--effort-limit", type=float, default=None)
  p.add_argument("--hz", type=float, default=100.0, help="hold-stream rate")
  args = p.parse_args()
  if not math.isfinite(args.hz) or args.hz <= 0:
    p.error("--hz must be a finite positive number")

  ctl = config_loader.control()
  kp = args.kp if args.kp is not None else 5.0
  kd = args.kd if args.kd is not None else 0.1
  eff = args.effort_limit if args.effort_limit is not None else float(ctl.get("effort_limit_a", 1.5))
  sn = args.sn or config_loader.serial_number(args.hand) or None

  const = load_constants(args.gen, args.hand)
  driver_kwargs = {"kp": kp, "kd": kd}
  if args.gen == 2 or args.effort_limit is not None:
    driver_kwargs["effort_limit_a"] = eff
  driver = make_driver(args.gen, args.hand, sn, mock=args.mock, constants=const, **driver_kwargs)
  try:
    displayed_eff = eff if (args.gen == 2 or args.effort_limit is not None) else 0.5
    print(f"[home_check] gen={args.gen} hand={args.hand} kp={kp} kd={kd} effort={displayed_eff}A — "
          "homing to default pose…")
    driver.home()
    print("[home_check] holding at default pose. Ctrl+C to disable & exit.")
    dt = 1.0 / args.hz
    while True:
      driver.write_target(const.default_joint_pos)
      time.sleep(dt)
  except KeyboardInterrupt:
    print("\n[home_check] disabling…")
  finally:
    driver.close()


if __name__ == "__main__":
  main()
