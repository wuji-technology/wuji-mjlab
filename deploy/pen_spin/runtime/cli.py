# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Command line for the public fixed-wrist pen-spin runtime."""

from __future__ import annotations

import argparse

_SCALAR_GAINS = {"kp": float, "kd": float, "effort": float, "servo": float}


def parse_gains(spec: str) -> dict:
    """Parse scalar and per-joint gain overrides."""
    out: dict = {"kp_joint": {}, "effort_joint": {}}
    for item in (s.strip() for s in spec.split(",")):
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"--gains: expected k=v, got {item!r}")
        key, val = (s.strip() for s in item.split("=", 1))
        try:
            fval = float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--gains: {key}={val!r} is not a number"
            ) from None
        if key in _SCALAR_GAINS:
            out[key] = fval
        elif key.startswith("kp.") or key.startswith("effort."):
            kind, joint = key.split(".", 1)
            out[f"{kind}_joint"][joint] = fval
        else:
            raise argparse.ArgumentTypeError(f"--gains: unknown key {key!r}")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pen-spin-runtime",
        description="Fixed-wrist pen-spin deployment for the Wuji Hand.",
    )
    source = parser.add_argument_group("release artifacts")
    source.add_argument("--policy", required=True, help="exported policy .onnx")
    source.add_argument("--motion", required=True, help="reference clip .npz")

    mode = parser.add_argument_group("mode")
    mode.add_argument("--hold-last-frame", action="store_true",
                      help="keep running against a stationary final reference until manually stopped")
    mode.add_argument("--open-loop", action="store_true")
    mode.add_argument("--obj-from-ref", action="store_true")

    io = parser.add_argument_group("hardware and streams")
    io.add_argument("--no-hardware", action="store_true")
    io.add_argument("--hand-sn", help="serial number; otherwise the only hand on the bus")
    sim = io.add_mutually_exclusive_group()
    sim.add_argument("--sim", dest="sim", action="store_true", default=True,
                     help="open the live MuJoCo mirror alongside policy control (default; requires a desktop display)")
    sim.add_argument("--no-sim", dest="sim", action="store_false",
                     help="disable the mirror for headless deployment; real-hand control remains enabled")
    io.add_argument("--pen-host", default="localhost",
                    help="camera observer host (default: localhost)")
    io.add_argument("--pen-port", type=int, default=5555,
                    help="camera observer ZMQ port (default: 5555)")
    # The shipped tag layout describes the physical pen with a +Z shaft. The
    # policy and reference clips use the tracking frame (+X), so live camera poses
    # must be converted unless a custom observer already publishes +X.
    io.add_argument("--pen-source-frame", choices=("tracking-x", "source-z"),
                    default="source-z",
                    help="observer pen body frame; source-z is the shipped tag layout")
    io.add_argument("--pen-axis-offset-m", type=float, default=0.0,
                    help="optional offset toward the pen A_top axis")
    io.add_argument("--jmode", default="neg90")
    io.add_argument("--gains", type=parse_gains, default=parse_gains(""), metavar="K=V,...")
    io.add_argument("--init-curl-deg", type=float, default=0.0, metavar="DEG")
    return parser


def resolve_sources(args) -> tuple[str, str]:
    """Return the explicit public release artifacts."""
    return args.policy, args.motion
