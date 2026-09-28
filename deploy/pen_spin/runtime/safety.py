# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Everything between "the program started" and "the policy is driving".

These are small functions, but they are the ones that decide whether a real hand
lurches. Two rules they encode:

  * **Never jump.** Both eases interpolate from the hand's ACTUAL current
    position with a smoothstep profile, so the first commanded target is where
    the fingers already are. Commanding the destination directly — which is what
    happens if you skip this and go straight to the loop — is a step input into
    an impedance controller holding a physical hand.
  * **Never move unasked.** The operator commits with Enter, after the pose ease
    and after preflight, with the hand energised but stationary. This is the
    last point at which a human can decide the setup looks wrong.

The zero-pose ease is the abort path: it runs on 'r' and on shutdown so the hand
ends somewhere safe rather than frozen mid-grasp.
"""

from __future__ import annotations

import sys
import termios
import threading
import time
import tty
from collections import deque

import numpy as np

EASE_DURATION_S = 3.0
EASE_RATE_HZ = 50.0


def _ease(ctrl, current_5x4, target_5x4, duration: float) -> None:
    """Smoothstep from current to target."""
    steps = max(1, int(duration * EASE_RATE_HZ))
    dt = duration / steps
    current = np.asarray(current_5x4, dtype=np.float64).reshape(5, 4)
    target = np.asarray(target_5x4, dtype=np.float64).reshape(5, 4)
    for i in range(steps):
        t = (i + 1) / steps
        smooth = t * t * (3 - 2 * t)
        ctrl.set_joint_target_position(current + smooth * (target - current))
        time.sleep(dt)


def pregrasp_curl(init_qpos20, curl_deg: float, lower, upper) -> np.ndarray:
    """Close the 15 FLEXION joints `curl_deg` beyond the reference start pose.

    The hand has to be *holding* the object when the policy takes over. Easing to
    the clip's frame 0 exactly leaves the fingers where the recording had them,
    which on a real object is often just shy of contact — the policy then starts
    from "not gripping" and the object drops before it can react.

    Flexion only: indices 4f+{0,2,3} (MCP-flex / PIP / DIP). The abduction joint
    of each finger (4f+1) is left alone — squeezing it splays the fingers instead
    of closing them. Positive is flex; clipped to the mechanical range.

    Affects the pre-grasp hold ONLY. Control-loop targets are untouched.
    """
    q = np.asarray(init_qpos20, dtype=np.float64).reshape(-1).copy()
    if curl_deg == 0.0:
        return q
    curl = np.zeros(20, dtype=np.float64)
    curl[[4 * f + j for f in range(5) for j in (0, 2, 3)]] = np.deg2rad(curl_deg)
    return np.clip(q + curl,
                   np.asarray(lower, dtype=np.float64).reshape(20),
                   np.asarray(upper, dtype=np.float64).reshape(20))


def ease_to_initial(hand, ctrl, init_qpos20, *, duration: float = EASE_DURATION_S) -> float:
    """Ease to the given start pose. Returns the max residual error in degrees."""
    current = np.asarray(hand.read_joint_actual_position(), dtype=np.float64).reshape(5, 4)
    target = np.asarray(init_qpos20, dtype=np.float64).reshape(5, 4)
    hand.write_joint_enabled(True)
    _ease(ctrl, current, target, duration)
    actual = np.asarray(hand.read_joint_actual_position(), dtype=np.float64).reshape(5, 4)
    return float(np.rad2deg(np.abs(actual - target).max()))


def ease_to_zero(hand, ctrl, *, duration: float = EASE_DURATION_S) -> None:
    """Abort path: open the hand. Runs on reset and on interrupt."""
    current = np.asarray(hand.read_joint_actual_position(), dtype=np.float64).reshape(5, 4)
    _ease(ctrl, current, np.zeros((5, 4)), duration)


def ease_to_zero_on_interrupt(hand, *, duration: float = EASE_DURATION_S) -> None:
    """Ctrl+C path: open the hand BEFORE teardown disables the joints.

    Disabling from wherever the fingers happened to be leaves the hand limp — it
    drops whatever it was holding and can slam. So the interrupt path opens the
    hand first, then hands off to shutdown.

    Opens its own controller: by the time a KeyboardInterrupt reaches here the
    loop's controller context has already unwound.

    Teardown guards:
      * interrupted before the hand connected -> nothing to ease
      * a SECOND Ctrl+C during the ease -> abandon it and let teardown proceed,
        because an operator hitting it twice wants the thing to stop, now
      * anything else -> report, never block teardown
    """
    if hand is None:
        return
    print("\n  [Ctrl+C] easing hand open before shutdown...")
    try:
        with hand.realtime_controller() as ctrl:
            ease_to_zero(hand, ctrl, duration=duration)
        print("  hand open.")
    except KeyboardInterrupt:
        print("  [Ctrl+C] ease abandoned — disabling joints now.")
    except Exception as e:  # noqa: BLE001 — teardown path, never block
        print(f"  WARNING zero-pose ease failed: {type(e).__name__}: {e}")


def restore_terminal() -> None:
    """Undo cbreak mode. Idempotent and silent — best effort by definition.

    Called from teardown as well as the context manager's exit: an exception on
    an unusual path must not leave the operator's shell without line editing.
    """
    try:
        fd = sys.stdin.fileno()
        attrs = termios.tcgetattr(fd)
        attrs[3] |= termios.ICANON | termios.ECHO   # lflags
        termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
    except Exception:  # noqa: BLE001 — not a tty, or already sane
        pass


class Keys:
    """Collect terminal and viewer input using the same control-key protocol."""

    def __init__(self, keyboard=None):
        self.keyboard = keyboard
        self._q: deque[str] = deque()
        self._lock = threading.Lock()

    # Called from the viewer's render thread — hence the lock.
    def push(self, ch: str) -> None:
        with self._lock:
            self._q.append(ch)

    def get(self) -> str | None:
        with self._lock:
            if self._q:
                return self._q.popleft()
        return self.keyboard.get_key() if self.keyboard is not None else None

    def drain(self) -> None:
        while self.get() is not None:
            pass

    @property
    def interactive(self) -> bool:
        """True when the TERMINAL can deliver keys.

        Only about the terminal — the viewer is a separate path the caller knows
        about, and buffered keys are not evidence of anyone watching. Callers who
        need "is anybody there at all" check this OR the viewer.
        """
        return self.keyboard is not None and self.keyboard.fd is not None


# GLFW keycodes the viewer forwards. Enter arrives as either the main or the
# keypad key depending on which one the operator hits.
GLFW_ENTER, GLFW_KP_ENTER, GLFW_R = 257, 335, 82


def viewer_key_callback(keys: Keys):
    """key_callback for mujoco.viewer.launch_passive -> Keys."""
    def _cb(keycode: int) -> None:
        if keycode in (GLFW_ENTER, GLFW_KP_ENTER):
            keys.push("\n")
        elif keycode == GLFW_R:
            keys.push("r")
    return _cb


class Keyboard:
    """Non-blocking single-key reads. No-ops when stdin is not a tty, so the
    same code path runs headless in the gate."""

    def __init__(self):
        self.fd = None
        self._saved = None

    def __enter__(self):
        try:
            self.fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        except Exception:  # noqa: BLE001 — not a tty; degrade to no-op
            self.fd = None
        return self

    def __exit__(self, *_):
        if self.fd is not None and self._saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)
        return False

    def get_key(self):
        if self.fd is None:
            return None
        import select

        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None
