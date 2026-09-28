#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Standalone camera-pose viewer: ZMQ -> tag -> palm -> MuJoCo scene.

No checkpoint, reference clip, physics stepping, or motor commands.
By default the real hand's encoder feedback drives the rendered joints.
Use --no-hand for a static hand without an SDK connection.
"""
from __future__ import annotations

import argparse
import sys
import time
from contextlib import ExitStack

import numpy as np

from lib.ghost_overlay import add_axes
from lib.hand_feedback import HandFeedback
from lib.math_utils import matrix_from_quat
from lib.pen_frame import source_to_tracking
from lib.tag_frame import lift_wrist_local_to_scene, pen_in_palm
from lib.zmq_bridge import PEN_PORT, PenReceiver
from runtime.scene import build
from runtime.viz import SceneHandles


class LivePenScene:
    """Mirror the measured pen and hand independently; hide an untracked pen."""

    def __init__(self, scene, *, source_frame: str):
        self.scene = scene
        self.source_frame = source_frame
        self.handles = SceneHandles(scene.model)
        self.geom_ids = np.flatnonzero(
            scene.model.geom_bodyid == self.handles.obj_body_id)
        self.rgba = scene.model.geom_rgba[self.geom_ids].copy()
        self.matids = scene.model.geom_matid[self.geom_ids].copy()
        # Hide the authored spawn immediately, before opening the window.
        self.update(np.zeros(3), np.array([1., 0., 0., 0.]), None, 0.25)

    def update(self, pos_tag, quat_tag, age, stale_s, *, hand_qpos=None):
        import mujoco

        model, data = self.scene.model, self.scene.data
        if hand_qpos is not None:
            joints = np.asarray(hand_qpos, dtype=np.float64)
            if joints.shape != (20,) or not np.isfinite(joints).all():
                raise ValueError("hand_qpos must contain 20 finite joint angles in radians")
            # SceneHandles resolves each address by canonical joint NAME, not
            # by assuming the fingers occupy the first 20 qpos entries.
            data.qpos[self.handles.finger_qposadr] = joints
        fresh = age is not None and age <= stale_s
        if fresh:
            pos, quat = pen_in_palm(pos_tag, quat_tag)
            if self.source_frame == "source-z":
                quat = source_to_tracking(quat)
            pos, quat = lift_wrist_local_to_scene(
                pos, quat, self.scene.palm_pos, self.scene.palm_quat)
            q0 = self.handles.obj_qposadr
            data.qpos[q0:q0 + 7] = np.r_[pos, quat]
            model.geom_rgba[self.geom_ids] = self.rgba
            model.geom_matid[self.geom_ids] = self.matids
        else:
            # Textured materials can override geom alpha, so hide both.
            model.geom_matid[self.geom_ids] = -1
            model.geom_rgba[self.geom_ids, 3] = 0
        mujoco.mj_forward(model, data)
        return "LIVE" if fresh else "WAITING" if age is None else "STALE"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pen-host", default="localhost")
    ap.add_argument("--pen-port", type=int, default=PEN_PORT)
    ap.add_argument("--pen-source-frame", choices=("tracking-x", "source-z"),
                    default="source-z",
                    help="observer pen body frame; source-z is the shipped tag layout")
    ap.add_argument("--jmode", choices=("neg90", "default"), default="default",
                    help="default: level jig base, hand pitched down 10 degrees; "
                         "neg90: legacy flat-hand scene")
    hand_args = ap.add_mutually_exclusive_group()
    hand_args.add_argument("--no-hand", action="store_true",
                           help="static hand; do not connect to the hardware")
    hand_args.add_argument("--hand-sn", help="serial number; otherwise discover the right hand")
    hand_args.add_argument("--hand-address", help="direct device address in <device-ip>:<port> format")
    ap.add_argument("--rate-hz", type=float, default=60.)
    ap.add_argument("--stale-s", type=float, default=0.25,
                    help="hide the pen after this many seconds without a valid pose")
    ap.add_argument("--hand-stale-s", type=float, default=0.25,
                    help="hold displayed joints when any joint feedback is older than this")
    args = ap.parse_args(argv)
    if not np.isfinite(args.rate_hz) or args.rate_hz <= 0:
        ap.error("--rate-hz must be finite and positive")
    if not np.isfinite(args.stale_s) or args.stale_s <= 0:
        ap.error("--stale-s must be finite and positive")
    if not np.isfinite(args.hand_stale_s) or args.hand_stale_s <= 0:
        ap.error("--hand-stale-s must be finite and positive")

    import mujoco
    import mujoco.viewer

    scene = build(jmode=args.jmode, ghost=False)
    print(f"[pen viewer] {scene.describe()}", flush=True)
    live = LivePenScene(scene, source_frame=args.pen_source_frame)
    state = hand_state = None
    try:
        with ExitStack() as resources:
            hand = None
            if not args.no_hand:
                print("[pen viewer] connecting to right-hand feedback (read only)...", flush=True)
                try:
                    hand = HandFeedback(serial_number=args.hand_sn, address=args.hand_address)
                except Exception as exc:
                    print(f"[pen viewer] cannot read hand feedback: {exc}\n"
                          "Check hand power/connection or select --hand-sn. "
                          "Use --no-hand for a static viewer.", file=sys.stderr, flush=True)
                    return 2
                resources.callback(hand.close)
                print(f"[pen viewer] hand {hand.serial_number}: joint_states only; no motor commands", flush=True)
            else:
                print("[pen viewer] hand = static reference (--no-hand)", flush=True)
            receiver = PenReceiver(host=args.pen_host, port=args.pen_port)
            resources.callback(receiver.close)
            print(f"[pen viewer] listening on tcp://{args.pen_host}:{args.pen_port}", flush=True)
            viewer = resources.enter_context(mujoco.viewer.launch_passive(scene.model, scene.data))
            with viewer.lock():
                viewer.cam.lookat[:] = scene.palm_pos
                viewer.cam.distance = 0.65
                viewer.cam.azimuth = 120
                viewer.cam.elevation = -25
            while viewer.is_running():
                start = time.monotonic()
                pos, quat, age = receiver.latest()
                joints = None
                hand_text = "STATIC (--no-hand)"
                if hand is not None:
                    joints, hand_age, fresh_count = hand.latest(args.hand_stale_s)
                    status = "LIVE" if joints is not None else "WAITING" if hand_age is None else "STALE"
                    hand_text = f"{status} {fresh_count}/20"
                    if hand_text != hand_state:
                        print(f"[pen viewer] hand {hand_text}", flush=True)
                        hand_state = hand_text
                with viewer.lock():
                    current = live.update(pos, quat, age, args.stale_s, hand_qpos=joints)
                    viewer.user_scn.ngeom = 0
                    if current == "LIVE":
                        bid = live.handles.obj_body_id
                        add_axes(viewer.user_scn, scene.data.xpos[bid],
                                 matrix_from_quat(scene.data.xquat[bid]), scale=0.06)
                if current != state:
                    print(f"[pen viewer] {current}", flush=True)
                    state = current
                age_text = "--" if age is None else f"{age * 1000:.0f} ms"
                viewer.set_texts((None, None,
                    "Pen pose\nSample age\nHand",
                    f"{current}\n{age_text}\n{hand_text}"))
                viewer.sync()
                time.sleep(max(0., 1. / args.rate_hz - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
