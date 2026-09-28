# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Start-up checks run before the hand moves.

Each check returns a structured result, and fatal failures stop the run before
hardware is enabled.

Here every check runs first, returns a value instead of printing, and the run is
refused if any FATAL one failed. Checks are ordered cheapest-first so a broken
checkpoint does not wait on a camera timeout.

The strongest check is ``obs_builds``: it assembles a real observation from the
real clip at frame 0 and compares its width against the checkpoint. That is the
mismatch which otherwise surfaces as a dimension error mid-flight, or — when the
dims happen to agree — as a policy quietly reading the wrong layout.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from runtime import obs as obs_mod


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = True

    @property
    def mark(self) -> str:
        if self.ok:
            return "OK  "
        return "FAIL" if self.fatal else "WARN"


def check_clip(motion) -> list[Check]:
    out = []
    n = int(getattr(motion, "num_frames", 0) or 0)
    out.append(Check("clip.frames", n > 0, f"{n} frames"))
    ap = np.asarray(motion.axis_points)
    out.append(Check("clip.axis_points", ap.shape == (2, 3), f"shape {ap.shape}"))
    return out


def check_ctrl_dt(loaded, ctrl_dt: float) -> list[Check]:
    """The rate the loop runs at, vs the rate the checkpoint was trained at.

    ``config/control.yaml`` owns the physical loop rate and the fixed pen-spin
    config records the training rate. They must agree at 50 Hz.
    """
    hz = 1.0 / ctrl_dt
    ok = abs(loaded.ctrl_dt - ctrl_dt) < 1e-9
    return [Check("policy.ctrl_dt", ok,
                  f"{hz:.0f} Hz (control.yaml)" if ok else
                  f"{hz:.0f} Hz (control.yaml) but this policy was trained at "
                  f"{1.0 / loaded.ctrl_dt:.0f} Hz",
                  fatal=False)]


def check_obs(motion, loaded, *, open_loop: bool, obj_from_ref: bool) -> list[Check]:
    """Assemble one real observation and verify its width against the policy."""
    try:
        v = obs_mod.build_observation(
            motion, 0,
            np.zeros(20, dtype=np.float32),
            np.zeros(20, dtype=np.float32),
            np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]),
            last_raw_action=np.zeros(20, dtype=np.float32),
            open_loop=open_loop, obj_from_ref=obj_from_ref,
        )
    except Exception as e:  # noqa: BLE001 — a failed dry build IS the finding
        return [Check("obs.builds", False, f"{type(e).__name__}: {e}")]
    ok = v.shape[0] == loaded.obs_dim
    return [Check("obs.builds", ok,
                  f"built {v.shape[0]}D, policy wants {loaded.obs_dim}D")]


def check_hand(hand) -> list[Check]:
    """Limits must be finite and ordered — a bad read here would make the clamp
    a no-op or invert it, and the clamp is the last guard before a finger."""
    try:
        lo = np.asarray(hand.read_joint_lower_limit(), dtype=np.float64)
        hi = np.asarray(hand.read_joint_upper_limit(), dtype=np.float64)
    except Exception as e:  # noqa: BLE001
        return [Check("hand.limits", False, f"{type(e).__name__}: {e}")]
    shape_ok = lo.shape == (5, 4) and hi.shape == (5, 4)
    finite = bool(np.isfinite(lo).all() and np.isfinite(hi).all())
    ordered = bool((hi > lo).all())
    detail = f"shape {lo.shape} finite={finite} ordered={ordered}"
    kind = "real Wuji Hand 2" if getattr(hand, "is_real", False) else "stub (--no-hardware)"
    return [Check("hand.driver", True, kind, fatal=False),
            Check("hand.limits", shape_ok and finite and ordered, detail)]


# A startup liveness bound, not a control-loop threshold: it only has to separate
# a live observer from a dead one.
LIVE_STREAM_S = 1.0


def check_camera(objpose, *, required: bool, warmup_s: float = 0.5) -> list[Check]:
    """The camera observer must publish one valid pen pose before motion."""
    if not required:
        return [Check("camera.stream", True,
                      "skipped — object obs comes from the ref clip", fatal=False)]

    def live(snap):
        age = snap.get("pen_age_s")
        return objpose.has_pose(snap) and age is not None and age <= LIVE_STREAM_S

    # Stop waiting as soon as a live valid pose arrives. Merely having seen a
    # pose once does not authorize startup after the observer has stopped.
    deadline = time.monotonic() + warmup_s
    while True:
        snap = objpose.snapshot()
        if live(snap) or time.monotonic() >= deadline:
            break
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    age = snap.get("pen_age_s")
    ok = live(snap)
    detail = (f"age={'never' if age is None else f'{age * 1000:.0f}ms'} "
              f"rejects={objpose.receiver.rejected} "
              f"untracked={objpose.receiver.untracked}")
    if ok:
        return [Check("camera.stream", True, detail)]
    # untracked>0: frames arrive but carry no pen (tag not in view).
    # untracked==0: nothing has arrived on the socket at all.
    if objpose.receiver.seen:
        detail += (f"  || previously received a pen pose, but nothing for "
                   f"{LIVE_STREAM_S:g}s — the observer has stopped; restart it")
    elif objpose.receiver.untracked > 0:
        detail += ("  || observer is running and publishing, but the pen tag "
                   "has not been seen (world_fixed=false / no 'pen' key) — "
                   "bring the tag into the camera's view")
    else:
        detail += ("  || no frame has arrived on the pen ZMQ port — check that "
                   "the camera observer (scripts/pen_world_observer.py) is running")
    return [Check("camera.stream", False, detail)]


def run_all(session) -> list[Check]:
    """Every check, in cheapest-first order. One call, so a caller cannot forget
    one or run them out of order — which the assembled-by-hand list invited."""
    m = session.mode
    return (
        check_ctrl_dt(session.loaded, session.ctrl_dt)
        + check_clip(session.motion)
        + check_obs(session.motion, session.loaded,
                    open_loop=m.open_loop, obj_from_ref=not m.needs_camera)
        + check_hand(session.hand)
        + check_camera(session.objpose, required=m.needs_camera)
    )


def report(checks: list[Check], *, stream=None) -> bool:
    """Print the table; return True when it is safe to proceed."""
    import sys

    out = stream or sys.stdout
    width = max((len(c.name) for c in checks), default=0)
    print("\n  ── preflight ──", file=out)
    for c in checks:
        print(f"  [{c.mark}] {c.name:<{width}}  {c.detail}", file=out)
    fatal = [c for c in checks if not c.ok and c.fatal]
    warn = [c for c in checks if not c.ok and not c.fatal]
    if warn:
        print(f"  {len(warn)} warning(s) — proceeding.", file=out)
    if fatal:
        print(f"  {len(fatal)} FATAL — refusing to run:", file=out)
        for c in fatal:
            print(f"    {c.name}: {c.detail}", file=out)
        return False
    print("  all checks passed.\n", file=out)
    return True
