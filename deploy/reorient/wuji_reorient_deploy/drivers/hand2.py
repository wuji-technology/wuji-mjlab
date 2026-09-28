# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji Hand 2 real-hardware driver (wuji_sdk).

wuji_sdk joint layout: nid is 1-based with stride 5 per finger (1-4, 6-9, ...).
Joint index in our flat 20-vector = finger*4 + joint_within_finger, i.e.
finger1(thumb)..finger5(pinky) × joint1..4 — matching mjcf ``*_finger{1-5}_joint{1-4}``.
"""
from __future__ import annotations

import time

import numpy as np

from ..constants import Constants
from .base import HandDriverBase


def _mit_params_value(
  kp: float | np.ndarray, kd: float | np.ndarray, num_joints: int
) -> tuple[float, float] | list[tuple[float, float]]:
  if isinstance(kp, np.ndarray) or isinstance(kd, np.ndarray):
    kp_arr = np.broadcast_to(kp, (num_joints,))
    kd_arr = np.broadcast_to(kd, (num_joints,))
    return [(float(p), float(d)) for p, d in zip(kp_arr, kd_arr, strict=True)]
  return (kp, kd)


class WujiHand2Driver(HandDriverBase):
  """Real Wuji Hand 2 via wuji_sdk, MIT impedance holding toward joint targets."""

  def __init__(
    self,
    constants: Constants,
    hand_side: str = "right",
    serial_number: str | None = None,
    kp: float | np.ndarray = 20.0,
    kd: float | np.ndarray = 0.1,
    effort_limit_a: float = 1.5,
    energize: bool = True,
  ):
    """Connect to a Wuji Hand 2 and start its feedback subscriptions."""
    super().__init__(constants, hand_side)
    import wuji_sdk

    self._sdk = wuji_sdk
    self._mgr = wuji_sdk.SdkManager.instance()
    opts = wuji_sdk.ConnectOptions(enable_bridge=False)
    try:
      if serial_number:
        self._hand = self._mgr.connect(
          sn=serial_number, device_name="wuji_hand_2", options=opts
        )
      else:
        self._hand = self._mgr.auto_connect(device_name="wuji_hand_2")
    except Exception as e:
      self._diagnose(serial_number, e)
      raise
    print(f"[hand_driver] connected: {self._hand.serial_number}")
    n_online = self._hand.online_joints_count().get()
    if n_online != constants.num_joints:
      raise RuntimeError(
        f"Hand {self._hand.serial_number}: {n_online}/{constants.num_joints} joints online — "
        "refusing to move with an incomplete hand (missing joints would silently "
        "read back the default pose)")


    # MIT is the firmware default (no control_mode()), but enable() IS required to
    # energize the motors — without it the hand accepts commands but never moves.
    self._JointCommand = wuji_sdk.JointCommand
    self._hand.effort_limit().set(effort_limit_a)
    self._hand.mit_params().set(_mit_params_value(kp, kd, constants.num_joints))

    # Any failure after this point must close the connection before propagating.
    # When energized, this is also what prevents an __init__ failure from leaving
    # the physical hand live and holding torque before the caller owns the driver.
    self._pub = None
    self._sub = None
    self._diag_sub = None
    self._errors: dict[int, int] = {}
    self._temps: dict[int, float] = {}
    self._dismissed: set[tuple[int, int]] = set()
    self.energized = bool(energize)
    if self.energized:
      self._hand.enable()
    try:
      time.sleep(0.25)  # let the low-level driver settle before opening streams
      self._pub = self._hand.joint_command().publish()

      self._latest = constants.default_joint_pos.copy()
      self._joint_t: np.ndarray = np.full(
        constants.num_joints, -np.inf, dtype=np.float64
      )
      self._got_state = False
      self._last_state_t = time.monotonic()
      self._sub = self._hand.joint_states().subscribe_with_callback(self._on_state)

      try:
        self._diag_sub = self._hand.joint_diagnostics().subscribe_with_callback(
          self._on_diag)
      except Exception as e:
        self._diag_sub = None
        print(f"[hand_driver] ⚠ joint_diagnostics unavailable ({e}) — "
              "fault detection limited to stale-feedback checks")

      t0 = time.monotonic()
      while True:
        missing = self.missing_feedback()
        if not missing:
          break
        if time.monotonic() - t0 > 2.0:
          raise RuntimeError(
            "missing fresh joint_states feedback within 2s for: "
            + ", ".join(missing))
        time.sleep(0.02)
    except BaseException:
      self.close()
      raise

  def _diagnose(self, serial_number, err) -> None:
    print(f"\n[hand_driver] CONNECT FAILED for sn={serial_number or 'auto'}: {err}")
    try:
      devs = self._mgr.scan()
    except Exception as e:
      print(f"[hand_driver] scan also failed: {e}")
      return
    print(f"[hand_driver] scan found {len(devs)} device(s):")
    ips = {}
    for d in devs:
      key = f"{d.ip}"
      ips.setdefault(key, []).append(d.sn)
      print(f"    sn={d.sn}  ip={d.ip}  transport={d.transport_type}")
    dup = {ip: sns for ip, sns in ips.items() if len(sns) > 1}
    if dup:
      print("[hand_driver] ⚠ IP CONFLICT — multiple devices share an IP; the SDK "
            "cannot reach a specific one reliably:")
      for ip, sns in dup.items():
        print(f"    {ip} -> {sns}")
      print("[hand_driver] fix: give each hand a unique static IP, or power on "
            "only the target hand and run without --sn (auto-connect).")

  def _on_state(self, frame) -> None:
    now = time.monotonic()
    self._last_state_t = now
    self._got_state = True
    q = self._latest
    for e in frame.joints:  # nid 1-based, stride 5 per finger
      finger, jw = divmod(e.nid - 1, 5)
      if jw < 4 and 0 <= finger < 5:
        index = finger * 4 + jw
        q[index] = e.position
        self._joint_t[index] = now

  def missing_feedback(self, max_age_s: float = 0.5) -> tuple[str, ...]:
    """Joint names with no fresh per-joint feedback."""
    now = time.monotonic()
    names = self.joint_names_in_encoder_order()
    return tuple(
      name
      for name, updated_at in zip(names, self._joint_t, strict=True)
      if now - updated_at > max_age_s
    )

  def _on_diag(self, frame) -> None:
    for e in frame.joints:
      nid = int(e.nid)
      self._temps[nid] = float(e.mcu_temp_c_fb)
      code = int(e.error_code_current)
      if code != 0:
        self._errors[nid] = code

  @staticmethod
  def _joint_name(nid: int) -> str:
    finger, jw = divmod(nid - 1, 5)
    return (f"finger{finger + 1} joint{jw + 1}" if jw < 4
            else f"finger{finger + 1} node{jw + 1}")

  def temp_summary(self) -> str | None:
    temps = self._temps.copy()
    if not temps:
      return None
    hot_nid, hi = max(temps.items(), key=lambda item: item[1])
    lo = min(temps.values())
    mean = sum(temps.values()) / len(temps)
    return (
      f"{lo:.1f}-{hi:.1f}°C "
      f"(mean {mean:.1f}, hot: {self._joint_name(hot_nid)} {hi:.1f})"
    )

  def fault(self) -> str | None:
    for nid, code in list(self._errors.items()):
      if (nid, code) in self._dismissed:
        continue
      detail, severity = f"0x{code:04X}", ""
      try:
        d = self._hand.describe_error(code)
        if isinstance(d, dict):
          severity = str(d.get("severity", ""))
          detail = (f"0x{code:04X} {d.get('name')} [{severity}] "
                    f"{d.get('desc')} — {d.get('resolution')}")
      except Exception:
        pass
      if "warn" in severity.lower():
        print(f"[hand_driver] ⚠ joint nid={nid} ({self._joint_name(nid)}) "
              f"warning {detail}")
        self._dismissed.add((nid, code))
        continue
      return f"joint nid={nid} ({self._joint_name(nid)}) error {detail}"
    missing = self.missing_feedback()
    if missing:
      return f"joint feedback missing or stale: {', '.join(missing)}"
    stale = time.monotonic() - self._last_state_t
    if stale > 0.5:
      return f"joint_states feed stale for {stale:.1f}s (bus/comm lost?)"
    return None

  def write_target(self, qpos: np.ndarray) -> None:
    qpos = np.asarray(qpos, dtype=np.float64)
    n = self.constants.num_joints
    if qpos.shape != (n,):
      raise ValueError(f"expected ({n},), got {qpos.shape}")
    # Never stream NaN/Inf to real MIT-impedance motors:
    # np.clip upstream passes NaN through and the EMA would latch it permanently.
    # Explicit raise (not assert) so guards survive `python -O`.
    if not np.isfinite(qpos).all():
      raise ValueError("WujiHand2Driver refuses NaN/Inf")
    JC = self._JointCommand
    self._pub.send([JC(float(x), 0.0, 0.0) for x in qpos])

  def read_encoders(self) -> np.ndarray:
    return self._latest.copy()

  def close(self) -> None:
    try:
      self._hand.disable()
    except Exception as e:
      print(f"[hand_driver] disable() failed: {e}")
    try:
      if self._diag_sub is not None:
        self._diag_sub.close()
    except Exception:
      pass
    try:
      if self._sub is not None:
        self._sub.close()
    finally:
      self._mgr.disconnect_all()
