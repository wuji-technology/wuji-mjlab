#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji Hand cube-reorient policy runner (process 2 of 3), gen1 + gen2."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
from wuji_reorient_deploy import config_loader
from wuji_reorient_deploy.constants import load_constants
from wuji_reorient_deploy.drivers import make_driver
from wuji_reorient_deploy.math_utils import quat_inv, quat_mul
from wuji_reorient_deploy.obs_builder import ActionPostprocessor, ObsAssembler
from wuji_reorient_deploy.onnx_policy import ONNXPolicy
from wuji_reorient_deploy.zmq_bridge import CubeReceiver, GoalPublisher, JointPublisher

_POLICY_DIR = Path(__file__).resolve().parents[1] / "policies"


def _rand_quat() -> np.ndarray:
  q = np.random.randn(4)
  return q / np.linalg.norm(q)


def _geodesic(a: np.ndarray, b: np.ndarray) -> float:
  """Angle (rad) between two unit quaternions (w,x,y,z)."""
  d = quat_mul(a, quat_inv(b))
  return 2.0 * float(np.arccos(min(1.0, abs(d[0]))))


def _aligned_quats() -> list[np.ndarray]:
  """The 24 proper rotations of a cube (face-up, axis-aligned) as (w,x,y,z)."""
  import itertools

  from scipy.spatial.transform import Rotation

  out = []
  for perm in itertools.permutations(range(3)):
    for signs in itertools.product((1.0, -1.0), repeat=3):
      m = np.zeros((3, 3))
      for i, p in enumerate(perm):
        m[i, p] = signs[i]
      if np.linalg.det(m) > 0:
        x, y, z, w = Rotation.from_matrix(m).as_quat()
        out.append(np.array([w, x, y, z], dtype=np.float64))
  return out


class AutoGoal:
  """Goal generator, switch on accumulated within-threshold time or timeout."""

  def __init__(self, success_threshold: float, hold_s: float, timeout_s: float,
               ctrl_dt: float, mode: str = "so3", initial_goal: np.ndarray | None = None,
               rest_s: float = 0.0):
    self.thr = success_threshold
    self.hold_steps = max(1, int(hold_s / ctrl_dt))
    self.timeout_steps = max(1, int(timeout_s / ctrl_dt))
    self.rest_steps = max(0, int(rest_s / ctrl_dt))
    self.mode = mode
    self._aligned = _aligned_quats() if mode == "aligned" else None
    if initial_goal is not None:
      self.goal = np.asarray(initial_goal, dtype=np.float64)
      self.goal /= np.linalg.norm(self.goal)
    else:
      self.goal = self._sample(np.array([1.0, 0.0, 0.0, 0.0]))
    self._hold = 0
    self._age = 0
    self._rest = 0
    self.reached = 0
    self.timedout = 0

  def _sample(self, prev: np.ndarray) -> np.ndarray:
    if self.mode == "aligned":
      for _ in range(200):
        q = self._aligned[np.random.randint(len(self._aligned))]
        if _geodesic(q, prev) > np.pi / 2:
          return q
      return self._aligned[np.random.randint(len(self._aligned))]
    for _ in range(100):
      q = _rand_quat()
      if _geodesic(q, prev) > np.pi / 2:
        return q
    return _rand_quat()

  def err_deg(self, cube_quat_tag: np.ndarray) -> float:
    return float(np.degrees(_geodesic(cube_quat_tag, self.goal)))

  def update(self, cube_quat_tag: np.ndarray) -> np.ndarray:
    if self._rest > 0:
      self._rest -= 1
      if self._rest == 0:
        self.goal = self._sample(self.goal)
        self._hold = 0
        self._age = 0
      return self.goal
    self._age += 1
    if _geodesic(cube_quat_tag, self.goal) < self.thr:
      self._hold += 1  # accumulated within-threshold time; no reset on exits
    if self._hold >= self.hold_steps:
      self.reached += 1
      if self.rest_steps > 0:
        self._rest = self.rest_steps
      else:
        self.goal = self._sample(self.goal)
        self._hold = 0
        self._age = 0
    elif self._age >= self.timeout_steps:
      self.timedout += 1
      self.goal = self._sample(self.goal)
      self._hold = 0
      self._age = 0
    return self.goal


def _shutdown(driver, rec: dict | None, log_npz: str | None) -> None:
  print("[run_policy] disabling hand…")
  try:
    driver.close()
  except BaseException as e:  # noqa: BLE001 - a failed disable must still be reported
    print(f"[run_policy] ⚠ driver.close() FAILED: {e} — HAND MAY STILL BE LIVE")

  if rec is not None and rec["t"] and log_npz:
    try:
      np.savez(log_npz, **{k: np.asarray(v) for k, v in rec.items()})
      print(f"[run_policy] per-tick log ({len(rec['t'])} ticks) → {log_npz}")
    except BaseException as e:  # noqa: BLE001 - log loss must never mask shutdown
      print(f"[run_policy] ⚠ per-tick log NOT saved: {e}")


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--gen", type=int, choices=(1, 2), default=2,
                help="hand generation (default: 2)")
  p.add_argument("--hand", choices=["right"], required=True)
  p.add_argument("--ckpt", type=str, default=None,
                 help="ONNX path (default policies/policy_hand{gen}_<hand>.onnx)")
  p.add_argument("--sn", type=str, default=None,
                 help="Hand serial number (default: config/control.yaml, else auto-connect)")
  p.add_argument("--mock", action="store_true", help="Run without hardware (MockHandDriver)")
  p.add_argument("--goal-mode", choices=["so3", "aligned"], default="so3",
                 help="'so3'=uniform goals (as trained); 'aligned'=face-up axis-aligned (simplified)")
  p.add_argument("--initial-goal", default=None,
                 help="first goal quat w,x,y,z in tag frame (e.g. 1,0,0,0 = red/TOP up); "
                      "default = sampled")
  p.add_argument(
    "--cube-wait",
    type=float,
    default=5.0,
    help="seconds to wait for the first valid cube pose before entering the main loop "
         "(default: 5.0)",
  )
  p.add_argument(
    "--cube-timeout",
    type=float,
    default=0.5,
    help="stop if the cube pose is older than this many seconds; 0 disables "
         "(default: 0.5, matching the Wuji Hand 2 driver's joint-state fault timeout)",
  )
  p.add_argument("--servo-hz", type=float, default=None)
  p.add_argument("--effort-limit", type=float, default=None)
  p.add_argument("--kp", type=float, default=None)
  p.add_argument("--kd", type=float, default=None)
  p.add_argument("--pd-scale", type=float, default=None,
                 help="scale training-time per-joint kp; kd follows sqrt(scale) "
                      "to preserve damping ratio (ignored when --kp or --kd is set; "
                      "default comes from config pd_scale)")
  p.add_argument("--kp-scale", type=float, default=None,
                 help="override --pd-scale for training-time per-joint kp "
                      "(ignored when --kp or --kd is set)")
  p.add_argument("--kd-scale", type=float, default=None,
                 help="override --pd-scale for training-time per-joint kd "
                      "(ignored when --kp or --kd is set)")
  p.add_argument("--ema-alpha", type=float, default=None,
                 help="action smoothing (lower = slower/smoother; trained value 0.5)")
  p.add_argument("--action-scale", type=float, default=None,
                 help="per-action motion size (lower = smaller motions; trained value 0.5)")
  p.add_argument("--success-threshold", type=float, default=None)
  p.add_argument("--goal-hold", type=float, default=None)
  p.add_argument("--goal-timeout", type=float, default=None)
  p.add_argument("--goal-rest", type=float, default=None,
                 help="seconds to keep a reached goal before switching "
                      "(default 3; config goal.rest_s)")
  p.add_argument("--no-rest", action="store_true",
                 help="disable the post-goal rest")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--log-npz", default=None,
                 help="record per-tick obs/action/target/encoders/cube/goal to this "
                      "npz on exit (e.g. /tmp/run_log.npz)")
  args = p.parse_args()
  if not math.isfinite(args.cube_wait) or args.cube_wait < 0:
    p.error("--cube-wait must be a finite non-negative number")
  if not math.isfinite(args.cube_timeout) or args.cube_timeout < 0:
    p.error("--cube-timeout must be a finite non-negative number")
  np.random.seed(args.seed)

  ctl, gl = config_loader.control(), config_loader.goal()
  def pick(cli, cfg_dict, key, fallback):
    return cli if cli is not None else float(cfg_dict.get(key, fallback))
  effort_limit = (args.effort_limit if args.gen == 1
                  else pick(args.effort_limit, ctl, "effort_limit_a", 1.5))
  servo_hz = pick(args.servo_hz, ctl, "servo_hz", 100.0)
  success_threshold = pick(args.success_threshold, gl, "success_threshold_rad", 0.4)
  goal_hold = pick(args.goal_hold, gl, "hold_s", 0.5)
  goal_timeout = pick(args.goal_timeout, gl, "timeout_s", 8.0)
  goal_rest = 0.0 if args.no_rest else pick(args.goal_rest, gl, "rest_s", 3.0)
  sn = args.sn or config_loader.serial_number(args.hand) or None

  const = load_constants(args.gen, args.hand)
  if args.kp is not None or args.kd is not None:
    kp = pick(args.kp, ctl, "kp", 20.0)
    kd = pick(args.kd, ctl, "kd", 0.1)
  else:
    pd_scale = args.pd_scale if args.pd_scale is not None else float(ctl.get("pd_scale", 2.0))
    kp_scale = args.kp_scale if args.kp_scale is not None else pd_scale
    kd_scale = args.kd_scale if args.kd_scale is not None else pd_scale ** 0.5
    kp = const.sim_kp * kp_scale
    kd = const.sim_kd * kd_scale
  ckpt = args.ckpt or str(_POLICY_DIR / f"policy_hand{args.gen}_{args.hand}.onnx")
  policy = ONNXPolicy(ckpt, const)
  cfg = policy.config
  ctrl_dt = float(cfg.get("ctrl_dt", 0.05))
  ema_alpha = args.ema_alpha if args.ema_alpha is not None \
      else float(ctl.get("ema_alpha", cfg.get("ema_alpha", 0.5)))
  action_scale = args.action_scale if args.action_scale is not None \
      else float(ctl.get("action_scale", cfg.get("action_scale", 0.5)))
  if ema_alpha != cfg.get("ema_alpha", 0.5) or action_scale != cfg.get("action_scale", 0.5):
    print(f"[run_policy] OVERRIDE ema_alpha={ema_alpha} action_scale={action_scale} "
          f"(trained: ema={cfg.get('ema_alpha')} scale={cfg.get('action_scale')})")
  post = ActionPostprocessor(
    const,
    action_scale=action_scale,
    ema_alpha=ema_alpha,
    warmup_time_s=cfg.get("warmup_time_s", 0.4),
    ctrl_dt=ctrl_dt,
  )
  asm = ObsAssembler(const)
  init_goal = None
  if args.initial_goal is not None:
    init_goal = np.array([float(x) for x in args.initial_goal.split(",")], dtype=np.float64)
  goal_gen = AutoGoal(success_threshold, goal_hold, goal_timeout, ctrl_dt,
                      mode=args.goal_mode, initial_goal=init_goal, rest_s=goal_rest)

  driver_kwargs = {"kp": kp, "kd": kd}
  if args.gen == 2 or args.effort_limit is not None:
    driver_kwargs["effort_limit_a"] = effort_limit
  driver = make_driver(
    args.gen, args.hand, sn, mock=args.mock, constants=const, **driver_kwargs,
  )
  rec = {k: [] for k in ("t", "obs", "raw", "target", "enc", "cube_pos",
                         "cube_quat", "goal", "err_deg")} if args.log_npz else None
  # Everything after the driver is live must reach the finally, so the hand is
  # disabled on any exit path.
  try:
    goal_pub = GoalPublisher()
    joint_pub = JointPublisher()
    cube_recv = CubeReceiver()
    if not cube_recv.wait_for_fix(args.cube_wait):
      raise SystemExit(
        f"[run_policy] no valid cube pose within {args.cube_wait}s — is "
        "cube_world_observer.py running, is its zmq port the same as "
        "config/control.yaml, and has the world frame been fixed?"
      )

    mode = "MOCK (no hardware)" if args.mock else f"REAL sn={sn or 'auto-connect'}"
    kp_repr = (f"per-joint[{kp.min():.3f}-{kp.max():.3f}] kp_scale={kp_scale} kd_scale={kd_scale}"
               if isinstance(kp, np.ndarray) else f"{kp}")
    displayed_effort_limit = 0.5 if effort_limit is None else effort_limit
    print(f"[run_policy] gen={args.gen} hand={args.hand} ckpt={Path(ckpt).name} ctrl_dt={ctrl_dt} "
          f"driver={mode} kp={kp_repr} effort={displayed_effort_limit}A goal_rest={goal_rest}s")
    print(f"[run_policy] policy config: {cfg}")
    print("[run_policy] homing…")
    driver.home()
    if not args.mock:
      input("[run_policy] hand homed. Press ENTER to let the policy take "
            "control of the real hand (Ctrl+C to abort)... ")

    servo_dt = 1.0 / servo_hz
    last_raw = np.zeros(const.action_dim)
    cube_pos, cube_quat = cube_recv.latest()
    goal = goal_gen.goal
    target = post.default_target
    enc = driver.read_encoders()
    obs = asm.build(enc, target, cube_pos, cube_quat, goal, last_raw)

    print("[run_policy] running. Ctrl+C to stop.")
    last_report = 0.0
    last_temp_report = 0.0
    t_start = time.monotonic()
    while True:
      hw_fault = driver.fault()
      if hw_fault:
        print(f"\n[run_policy] ⛔ HAND FAULT — {hw_fault}")
        print("[run_policy] stopping WITHOUT homing (won't drive faulted joints); "
              "motors will be disabled.")
        break

      age = cube_recv.age_s()
      if args.cube_timeout > 0 and (age is None or age > args.cube_timeout):
        shown = "never" if age is None else f"{age:.2f}s"
        print(f"\n[run_policy] ⛔ VISION STALE — no cube pose for {shown} "
              "(camera down / tag occluded / world frame lost); "
              "motors will be disabled.")
        break

      t0 = time.monotonic()
      raw = policy(obs)
      prev_target = target
      target = post.process(raw)

      n_sub = max(1, int(ctrl_dt / servo_dt))
      for i in range(1, n_sub + 1):
        s = i / n_sub
        driver.write_target(prev_target + s * (target - prev_target))
        goal_pub.publish(goal)
        joint_pub.publish(target, enc)
        rem = t0 + servo_dt * i - time.monotonic()
        if rem > 0:
          time.sleep(rem)

      cube_pos, cube_quat = cube_recv.latest()
      err_before = goal_gen.err_deg(cube_quat)
      goal = goal_gen.update(cube_quat)
      enc = driver.read_encoders()
      obs = asm.build(enc, target, cube_pos, cube_quat, goal, last_raw)
      last_raw = raw

      if rec is not None:
        rec["t"].append(time.monotonic() - t_start)
        rec["obs"].append(obs.copy())
        rec["raw"].append(raw.copy())
        rec["target"].append(target.copy())
        rec["enc"].append(enc.copy())
        rec["cube_pos"].append(cube_pos.copy())
        rec["cube_quat"].append(cube_quat.copy())
        rec["goal"].append(goal.copy())
        rec["err_deg"].append(err_before)

      now = time.monotonic()
      if now - last_report > 5.0:
        report_age = cube_recv.age_s()
        shown_age = "never" if report_age is None else f"{report_age:.2f}s"
        print(f"[run_policy] cube↔goal err={err_before:6.1f}°  "
              f"cube_pos_tag={np.round(cube_pos, 3)}  cube_msgs={cube_recv.count}  "
              f"cube_age={shown_age}  "
              f"reached={goal_gen.reached} timedout={goal_gen.timedout}", flush=True)
        last_report = now
      if now - last_temp_report > 10.0:
        summary = driver.temp_summary()
        if summary is not None:
          print(f"[run_policy] temp: {summary}", flush=True)
        last_temp_report = now
  except KeyboardInterrupt:
    print("\n[run_policy] Ctrl+C — homing hand…")
    try:
      driver.home()
    except Exception:
      pass
  finally:
    _shutdown(driver, rec, args.log_npz)


if __name__ == "__main__":
  main()
