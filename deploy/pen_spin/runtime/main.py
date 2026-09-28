# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Run one deployment: open a Session, check it, drive the loop, close it.

This branch runs one fixed-wrist pen-spin policy contract. The wrist body name
selects the sensing/calibration frame; only the 20 finger joints are actuated.

    python -m runtime --policy <policy.onnx> --motion <clip.npz>
"""

from __future__ import annotations

import signal

from lib import log as log_mod

from runtime import cli, preflight, safety
from runtime.loop import ControlLoop
from runtime.session import CameraNotReady, Session


def main(argv=None) -> int:
    """Returns a process exit code. Safe to call from a test: nothing here exits
    the interpreter (see runtime/__main__.py for why that separation matters)."""
    args = cli.build_parser().parse_args(argv)

    def _sigint(_sig, _frm):
        raise KeyboardInterrupt

    # SIGTERM and SIGHUP take the Ctrl+C path, which opens the hand and disables
    # the joints; otherwise they end the process with the joints energised.
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _sigint)
    # Ctrl+Z would stop the loop with the hand energised, and a stopped process
    # cannot take the Ctrl+C that follows.
    signal.signal(signal.SIGTSTP, signal.SIG_IGN)

    with safety.Keyboard() as keyboard:
        try:
            session = Session.open(args, keyboard=keyboard)
        except CameraNotReady as exc:
            print(f"\n  [FAIL] camera.stream: {exc}")
            print("  Hand was not connected; no MuJoCo window was opened.")
            print("  Start: pixi run -e pen-spin-deploy pen-observer --source camera --preview")
            return 2
        except KeyboardInterrupt:
            return 130
        try:
            for line in session.describe():
                print(f"  {line}")
            runlog = log_mod.setup()
            print(f"  run log: {runlog}" if runlog else
                  f"  run log: unavailable ({log_mod.LOGDIR} not writable)")
            if not preflight.report(preflight.run_all(session)):
                return 2

            loop = ControlLoop(session)
            with session.hand.realtime_controller() as ctrl:
                _ease_to_start(session, ctrl, args.init_curl_deg, loop)
                # Committing happens INSIDE the loop now: it holds in FROZEN,
                # refreshing the viewer live, until Enter. Doing it there rather
                # than in a separate wait means one place advances the clip clock
                # and one place reads keys.
                if loop.run(ctrl, keys=session.keys):
                    print("\n  [r] reset — opening the hand.")
                    safety.ease_to_zero(session.hand, ctrl)
        except KeyboardInterrupt:
            safety.ease_to_zero_on_interrupt(session.hand)
        finally:
            session.close()
    return 0


def _ease_to_start(session, ctrl, curl_deg: float, loop) -> None:
    """Move from wherever the fingers are to the reference start pose, then stop.

    The hand ends energised and stationary — the last point a human can call the
    run off, and the window in which they place the object into it.
    """
    start = safety.pregrasp_curl(session.start_pose, curl_deg,
                                 loop.lower, loop.upper)
    err = safety.ease_to_initial(session.hand, ctrl, start)
    curl = f", flexion +{curl_deg:.1f} deg pre-grasp" if curl_deg else ""
    print(f"  eased to start pose (max error {err:.2f} deg{curl})")
    # Keys go to the MuJoCo window (the terminal also works, but the window takes
    # focus the moment it opens, so naming it first is what matches reality).
    where = "MuJoCo window" if session.viewer is not None else "terminal"
    print(f"\n  >>> The hand is frozen at the start pose. Place the object in the hand now."
          f"\n      Solid = camera measurement; translucent = reference."
          f"\n      In the {where}, press:  Enter once = pre-grasp / first policy frame"
          f"\n                                Enter twice = start tracking"
          f"\n                                r = open and reset  Ctrl+C = abort",
          flush=True)
    session.keys.drain()   # the Enter that launched us must not read as a command
