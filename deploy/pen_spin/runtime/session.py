# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Own deployment resources and close the receiver before its ZMQ context."""

from __future__ import annotations

import signal
from dataclasses import dataclass

import numpy as np
from lib.config_loader import get_gains, get_timing
from lib.motion_clip import MotionClip

from runtime import hand as hand_mod
from runtime import policy as policy_mod
from runtime import preflight, safety, streams
from runtime import scene as scene_mod


class CameraNotReady(RuntimeError):
    """Startup refused before acquiring the physical hand or renderer."""


@dataclass(frozen=True)
class Mode:
    """Select whether observations come from hardware or the reference clip."""

    open_loop: bool = False
    obj_from_ref: bool = False

    @property
    def needs_camera(self) -> bool:
        """Live object sensing is off whenever the object comes from the clip."""
        return not (self.obj_from_ref or self.open_loop)


class Session:
    """Resources for one run. Build with ``Session.open(args)``."""

    def __init__(self, *, loaded, ctrl_dt, motion, scene, hand, objpose, viewer,
                 keys, mode, pen_host, pen_port, policy_path, motion_path):
        self.loaded = loaded
        self.ctrl_dt = ctrl_dt
        self.motion = motion
        self.scene = scene
        self.hand = hand
        self.objpose = objpose
        self.viewer = viewer
        self.keys = keys
        self.mode = mode
        self.pen_host = pen_host
        self.pen_port = pen_port
        self.policy_path = policy_path
        self.motion_path = motion_path
        self._closed = False

    # ----- construction -----------------------------------------------------

    @classmethod
    def open(cls, args, *, keyboard=None) -> "Session":
        """Acquire everything, in the order the dependencies demand.

        The order is not arbitrary: the checkpoint's ctrl_dt sets every other
        rate in the run, so the policy loads first; the scene must exist before
        the home pose can be read from it; the camera is checked before the hand
        is connected; the viewer needs the scene.
        """
        from runtime import cli

        if not np.isfinite(args.pen_axis_offset_m):
            raise ValueError("--pen-axis-offset-m must be finite")
        policy_path, motion_path = cli.resolve_sources(args)
        loaded = policy_mod.load(policy_path)
        # The rate is a DEPLOYMENT property, not a per-checkpoint one: one rig,
        # one control loop, one number in config/control.yaml. loaded.ctrl_dt is
        # only what the checkpoint claims — preflight compares them.
        ctrl_dt = float(get_timing()["ctrl_dt"])

        motion = cls._open_reference(args, ctrl_dt, motion_path)

        # Built on every run, --sim or not: the home pose and the palm pose
        # come out of the compiled model, and --jmode is what patches the palm.
        scene = scene_mod.build(jmode=args.jmode, ghost=args.sim)

        mode = Mode(open_loop=args.open_loop, obj_from_ref=args.obj_from_ref)

        # Keys are complete at construction: the viewer's callback needs the sink
        # to exist before the window opens, and the terminal is optional.
        keys = safety.Keys(keyboard)
        # The session exists before anything is acquired, so a failure part-way
        # releases what was already opened.
        session = cls(loaded=loaded, ctrl_dt=ctrl_dt, motion=motion, scene=scene,
                      hand=None, objpose=None, viewer=None, keys=keys,
                      mode=mode, pen_host=args.pen_host, pen_port=args.pen_port,
                      policy_path=policy_path, motion_path=motion_path)
        try:
            if mode.needs_camera:
                receiver = streams.open_camera(port=args.pen_port, host=args.pen_host)
                session.objpose = streams.ObjectPose(
                    receiver, source_frame=args.pen_source_frame,
                    axis_offset_m=args.pen_axis_offset_m)
                if args.pen_axis_offset_m:
                    print(f"[camera observation] offset toward A_top: "
                          f"{args.pen_axis_offset_m * 1000:+g} mm "
                          "along rotating pen axis (policy + mirror)", flush=True)
                print(f"[startup] waiting up to 2s for camera {args.pen_host}:{args.pen_port}",
                      flush=True)
                check = preflight.check_camera(session.objpose, required=True, warmup_s=2.0)[0]
                if not check.ok:
                    raise CameraNotReady(check.detail)
            # Do not spend SDK discovery time, or create a GL render thread,
            # when the camera prerequisite is already missing.
            import time

            started = time.monotonic()
            print("[startup] connecting hand" if not args.no_hardware
                  else "[startup] opening stub hand", flush=True)
            session.hand = cls._open_hand(args, scene, ctrl_dt)
            print(f"[startup] hand ready in {time.monotonic() - started:.2f}s", flush=True)
            if args.sim:
                from runtime import viz

                session.viewer = viz.open_viewer(
                    scene, key_callback=safety.viewer_key_callback(keys))
            return session
        except BaseException:
            session.close()
            raise

    @classmethod
    def _open_reference(cls, args, ctrl_dt, motion_path):
        """Load the reference clip at the deployment control rate."""
        return MotionClip(motion_path, fps=1.0 / ctrl_dt,
                          hold_last_frame=args.hold_last_frame)

    @staticmethod
    def _open_hand(args, scene, ctrl_dt):
        if args.no_hardware:
            seed = np.asarray(scene.home_qpos, np.float64).reshape(5, 4)
            lower, upper = hand_mod.joint_limits()
            return hand_mod.StubHand(lower, upper, seed)
        cfg, g = get_gains(), args.gains
        # servo_hz is a hardware streaming rate and lives under timing:, not
        # gains:; interp_s is the ramp BETWEEN two commands and must therefore be
        # exactly one control period. A scalar kp= / effort= on --gains replaces
        # the config's per-joint overrides of that kind; kp.<joint>= /
        # effort.<joint>= add to them.
        hand = hand_mod.connect_real(
            serial_number=args.hand_sn,
            kp=g.get("kp", cfg["kp"]), kd=g.get("kd", cfg["kd"]),
            servo_hz=g.get("servo", get_timing()["servo_hz"]), interp_s=ctrl_dt,
            kp_overrides={**({} if "kp" in g else cfg["kp_joint"]),
                          **g.get("kp_joint", {})},
            effort_overrides={**({} if "effort" in g else cfg["effort_joint"]),
                              **g.get("effort_joint", {})})
        hand.write_joint_effort_limit(g.get("effort", cfg["effort"]))
        return hand

    # ----- what the run needs to know about itself --------------------------

    @property
    def start_pose(self) -> np.ndarray:
        """Where the hand eases TO before the operator commits.

        The reference clip starts at frame 0.
        """
        return self.motion.reference_qpos[0]

    def describe(self) -> list[str]:
        # The timing line prints what control.yaml's periods and seconds actually
        # came to for THIS checkpoint. Both numbers move with ctrl_dt, so seeing
        # them resolved is the only way to notice a checkpoint whose rate is not
        # the one the deployment was set up for.
        dt = self.ctrl_dt
        pen = (f"camera zmq {self.pen_host}:{self.pen_port}" if self.objpose is not None
               else "from ref clip (no camera)")
        return [
            f"policy  {self.policy_path}",
            f"clip   {self.motion_path}",
            ("end    hold final reference indefinitely; policy remains active "
             "(r / Ctrl+C to stop)" if self.motion.hold_last_frame
             else "end    stop at clip end"),
            f"pen    {pen}",
            f"scene  {self.scene.describe()}",
            f"policy {self.loaded.describe()}",
            "actuate fixed wrist; 20 finger joints",
            f"timing {1.0 / dt:.0f}Hz ctrl_dt={dt * 1000:.0f}ms interp={dt * 1000:.0f}ms "
            f"servo={get_timing()['servo_hz']:.0f}Hz; "
            "camera samples are consumed at their latest available age",
        ]

    # ----- teardown ---------------------------------------------------------

    def close(self) -> list[str]:
        """Release in dependency order. Returns lingering non-daemon thread names.

        SIGINT is ignored for the duration — a second Ctrl+C here would abort
        teardown half-done and leave the joints energised — and restored at the
        end, because leaving it ignored takes away the operator's escape hatch if
        anything downstream hangs.
        """
        import threading

        if self._closed:
            return []
        self._closed = True
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (ValueError, OSError):
            pass  # not the main thread — best effort
        print("\n  shutting down...")
        steps = [
            ("joints disabled",
             (lambda: self.hand.write_joint_enabled(False)) if self.hand else None),
            ("camera closed", self.objpose.receiver.close if self.objpose else None),
            ("hand closed", getattr(self.hand, "close", None)),
            ("viewer closed", self.viewer.close if self.viewer else None),
        ]
        for label, fn in steps:
            if fn is None:
                continue
            try:
                fn()
                print(f"    {label}")
            except Exception as e:  # noqa: BLE001 — teardown reports, continues
                print(f"    WARNING {label} failed: {type(e).__name__}: {e}")
        safety.restore_terminal()
        try:
            signal.signal(signal.SIGINT, signal.default_int_handler)
        except (ValueError, OSError):
            pass
        lingering = [t.name for t in threading.enumerate()
                     if t is not threading.current_thread() and not t.daemon]
        if lingering:
            print(f"  non-daemon threads still alive: {', '.join(lingering)}"
                  " — the entry point will exit without waiting for them.")
        print("  shutdown complete.")
        return lingering
