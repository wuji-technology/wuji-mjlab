# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Hand driver facade over ``wuji_sdk`` for Wuji Hand 2.

The adapter exposes one stable 20-joint surface to the runtime. The method
names below are the SDK-facing contract:

    Hand2Compat()
      .write_joint_effort_limit(a)
      .read_joint_upper_limit() (5,4)    <- config/control.yaml joint_limits
      .read_joint_lower_limit() (5,4)       (hand.xml-consistent, so the
      .read_joint_actual_position() (5,4)    SDK<->MuJoCo identity check holds)
      .write_joint_enabled(bool)
      .realtime_controller(...) ctx      ~ ctrl.get_joint_actual_position()
                                           ctrl.set_joint_target_position(5x4)

The SDK state IDs map to the same finger-major order as the shared MJCF;
commands stream position targets through the SDK's impedance controller.

There is no command-side low-pass filter: smoothing is the MIT impedance
(kp/kd) plus the servo-thread interpolation below.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from lib.config_loader import get_joint_limits

DEFAULT_KP = 20.0
DEFAULT_KD = 0.1

# Commands faster than the firmware loop overwrite its target latch.
DEFAULT_SERVO_HZ = 1000.0
# The runtime must override this fallback with its control period to avoid interrupted ramps.
DEFAULT_INTERP_S = 0.05


class _Hand2Controller:
    """The object yielded by ``realtime_controller``."""

    def __init__(self, hand: "Hand2Compat"):
        self._hand = hand

    def get_joint_actual_position(self) -> np.ndarray:
        return self._hand.read_joint_actual_position()

    def set_joint_target_position(self, target) -> None:
        self._hand._write_target(np.asarray(target, dtype=np.float64).reshape(20))


class _ControllerContext:
    def __init__(self, hand: "Hand2Compat"):
        self._hand = hand

    def __enter__(self) -> _Hand2Controller:
        self._hand._ensure_enabled()
        return _Hand2Controller(self._hand)

    def __exit__(self, *exc) -> bool:
        return False  # motors stay energized across contexts


class Hand2Compat:
    """Wuji Hand 2 over wuji_sdk — the only supported hand."""

    def __init__(
        self,
        serial_number: str | None = None,
        kp: float = DEFAULT_KP,
        kd: float = DEFAULT_KD,
        servo_hz: float = DEFAULT_SERVO_HZ,
        interp_s: float = DEFAULT_INTERP_S,
        # Per-joint kp overrides on top of the global (kp, kd):
        # {flat_idx: kp} with flat_idx = finger*4 + joint_within_finger
        # (finger1..5 × joint1..4, e.g. finger5_joint2 -> 4*4+1 = 17).
        # The device mit_params resource is natively a 20-element array.
        kp_overrides: dict | None = None,
        # Per-joint effort-limit (A) overrides on top of the global scalar
        # set via write_joint_effort_limit; same {flat_idx: amps} indexing.
        # effort_limit().set is polymorphic: scalar = all 20, list[20] =
        # per-joint (flat finger-major scan order, same as mit_params).
        effort_overrides: dict | None = None,
    ):
        import wuji_sdk  # lazy: keep --no-hardware / --sim runnable without the SDK

        self._sdk = wuji_sdk
        self._mgr = wuji_sdk.SdkManager.instance()
        opts = wuji_sdk.ConnectOptions(enable_bridge=False)
        sn = serial_number
        if not sn:
            devs = self._mgr.scan()
            if len(devs) == 0:
                raise RuntimeError(
                    "no Wuji hand found on the bus (USB/UDP) — is the hand "
                    "powered on and on this host's network/USB?"
                )
            if len(devs) > 1:
                listing = ", ".join(
                    f"sn={d.sn} ip={d.ip} tp={d.transport_type}" for d in devs
                )
                raise RuntimeError(
                    f"{len(devs)} hands on the bus ({listing}) — pass a "
                    "serial_number to pin one"
                )
            sn = devs[0].sn
            print(f"[hand2] one hand found: sn={sn}  ip={devs[0].ip}  "
                  f"transport={devs[0].transport_type}")
        self._hand = self._mgr.connect(
            sn=sn, device_name="wuji_hand_2", options=opts
        )
        print(f"[hand2] connected: {self._hand.serial_number}")
        n_online = self._hand.online_joints_count().get()
        if n_online == 0:
            raise RuntimeError(f"hand {sn}: no joints online")
        print(f"[hand2] joints online: {n_online}")

        self._kp, self._kd = float(kp), float(kd)
        self._kp_overrides = {int(k): float(v)
                              for k, v in (kp_overrides or {}).items()}
        self._effort_overrides = {int(k): float(v)
                                  for k, v in (effort_overrides or {}).items()}
        self._effort_limit_a = 0.5  # overwritten by write_joint_effort_limit
        self._pub = None
        self._enabled = False

        # Servo-thread state (all under _servo_lock).
        self._servo_hz = float(servo_hz)
        self._interp_s = float(interp_s)
        self._servo_lock = threading.Lock()
        self._servo_thread: threading.Thread | None = None
        self._servo_stop = threading.Event()
        self._servo_sent = 0
        self._servo_t0 = 0.0
        self._seg_start: np.ndarray | None = None  # interp segment origin
        self._seg_target: np.ndarray | None = None
        self._seg_t0 = 0.0
        self._cmd: np.ndarray | None = None  # last streamed command

        # Deploy-side joint limits (control.yaml == hand.xml, gen-2 values).
        lims = get_joint_limits()  # (20, 2) [lower, upper]
        self._lower = lims[:, 0].reshape(5, 4).copy()
        self._upper = lims[:, 1].reshape(5, 4).copy()

        self._latest = np.zeros(20, dtype=np.float64)
        self._got_state = False
        self._sub = self._hand.joint_states().subscribe_with_callback(
            self._on_state
        )
        # Refuse to move blind: interpolations start from the true pose.
        t0 = time.monotonic()
        while not self._got_state:
            if time.monotonic() - t0 > 2.0:
                raise RuntimeError(
                    "no joint_states frame within 2s of connecting"
                )
            time.sleep(0.02)

    def write_joint_effort_limit(self, amps: float) -> None:
        self._effort_limit_a = float(amps)
        if self._enabled:
            self._apply_effort_limit()

    def read_joint_upper_limit(self) -> np.ndarray:
        return self._upper.copy()

    def read_joint_lower_limit(self) -> np.ndarray:
        return self._lower.copy()

    def read_joint_actual_position(self) -> np.ndarray:
        return self._latest.reshape(5, 4).copy()

    def write_joint_enabled(self, enabled) -> None:
        # callers pass an array/bool; any truthy -> energize, falsy -> limp.
        if np.any(np.asarray(enabled)):
            self._ensure_enabled()
        else:
            self._stop_servo()
            try:
                self._hand.disable()
            finally:
                self._enabled = False
                self._pub = None
            print("[hand2] motors DISABLED")

    def realtime_controller(self, *_, **__) -> _ControllerContext:
        return _ControllerContext(self)

    def close(self) -> None:
        self._stop_servo()
        try:
            self._hand.disable()
        except Exception:
            pass
        try:
            self._sub.close()
        finally:
            self._mgr.disconnect_all()

    # ---- internals -----------------------------------------------------

    def _apply_effort_limit(self) -> None:
        """Write the current limit: global scalar, or per-joint list[20]
        merged with overrides. Read-back verifies overrides landed
        (get() returns None for offline joints)."""
        if not self._effort_overrides:
            self._hand.effort_limit().set(self._effort_limit_a)
            return
        self._hand.effort_limit().set(
            [self._effort_overrides.get(i, self._effort_limit_a)
             for i in range(20)]
        )
        back = self._hand.effort_limit().get()
        for i, want in sorted(self._effort_overrides.items()):
            got = back[i]
            ok = got is not None and abs(float(got) - want) < 1e-3
            print(f"[hand2] effort override joint[{i}] "
                  f"(finger{i // 4 + 1}_joint{i % 4 + 1}): "
                  f"{'offline' if got is None else f'{float(got):g}A'} "
                  f"({'OK' if ok else 'MISMATCH!'})")

    def _ensure_enabled(self) -> None:
        if self._enabled:
            return
        self._apply_effort_limit()
        if self._kp_overrides:
            # Per-joint array write: global (kp, kd) everywhere, kp swapped
            # on the override indices. Read-back verifies the write landed.
            MP = self._sdk.MitParam
            arr = [MP(self._kp_overrides.get(i, self._kp), self._kd)
                   for i in range(20)]
            self._hand.mit_params().set(arr)
            back = self._hand.mit_params().get()
            for i, want in self._kp_overrides.items():
                got = float(back[i].kp)
                mark = "OK" if abs(got - want) < 1e-3 else "MISMATCH!"
                print(f"[hand2] kp override joint[{i}] "
                      f"(finger{i // 4 + 1}_joint{i % 4 + 1}): "
                      f"kp={got:g} ({mark})")
        else:
            self._hand.mit_params().set((self._kp, self._kd))
        self._hand.enable()
        time.sleep(0.25)  # let the low-level driver arm before streaming
        self._pub = self._hand.joint_command().publish()
        self._enabled = True
        # Seed the servo stream from the TRUE pose so the first interp
        # segment starts where the hand actually is (no snap), then start
        # the high-rate streamer.
        with self._servo_lock:
            self._cmd = self._latest.copy()
            self._seg_start = self._cmd.copy()
            self._seg_target = self._cmd.copy()
            self._seg_t0 = time.monotonic()
        self._servo_stop.clear()
        self._servo_thread = threading.Thread(
            target=self._servo_loop, daemon=True, name="hand2-servo"
        )
        self._servo_thread.start()
        eff_ovr = ("(" + ",".join(
            f"finger{i // 4 + 1}_joint{i % 4 + 1}={v:g}A"
            for i, v in sorted(self._effort_overrides.items())) + ") "
            if self._effort_overrides else "")
        print(
            f"[hand2] motors ENABLED  effort_limit={self._effort_limit_a}A "
            f"{eff_ovr}"
            f"kp={self._kp} kd={self._kd}  servo={self._servo_hz:.0f}Hz "
            f"(linear interp over {self._interp_s * 1000:.0f}ms)"
        )

    def _stop_servo(self) -> None:
        self._servo_stop.set()
        t = self._servo_thread
        if t is not None and t.is_alive():
            t.join(timeout=1.0)
        self._servo_thread = None
        if self._servo_sent and self._servo_t0:
            el = time.monotonic() - self._servo_t0
            if el > 0.5:
                print(f"[hand2] servo achieved {self._servo_sent / el:.0f} Hz "
                      f"avg over {el:.1f}s (target {self._servo_hz:.0f} Hz)")

    def _write_target(self, q20: np.ndarray) -> None:
        """Start a new interpolation segment: current streamed cmd → q20.

        The control-rate caller never touches the wire — the servo thread streams
        the linear ramp at servo_hz and holds q20 once the segment ends.
        """
        if self._pub is None:
            raise RuntimeError("set_joint_target_position before enable")
        with self._servo_lock:
            self._seg_start = (
                self._cmd.copy() if self._cmd is not None else q20.copy()
            )
            self._seg_target = q20.copy()
            self._seg_t0 = time.monotonic()

    def _servo_loop(self) -> None:
        JC = self._sdk.JointCommand
        dt = 1.0 / self._servo_hz
        self._servo_sent = 0
        self._servo_t0 = time.monotonic()
        next_t = time.monotonic()
        while not self._servo_stop.is_set():
            now = time.monotonic()
            with self._servo_lock:
                s = min(max((now - self._seg_t0) / self._interp_s, 0.0), 1.0)
                cmd = self._seg_start + s * (self._seg_target - self._seg_start)
                self._cmd = cmd
                pub = self._pub
            if pub is None:
                break
            try:
                pub.send([JC(float(x), 0.0, 0.0) for x in cmd])
                self._servo_sent += 1
            except Exception as e:  # noqa: BLE001 — surface, don't spin
                print(f"[hand2] servo stream error: {e} — servo stopped")
                break
            next_t += dt
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()  # fell behind: resync, don't burst

    def _on_state(self, frame) -> None:
        q = self._latest
        for e in frame.joints:  # nid 1-based, stride 5 per finger
            finger, jw = divmod(e.nid - 1, 5)
            if jw < 4 and 0 <= finger < 5:
                q[finger * 4 + jw] = e.position
        self._got_state = True
