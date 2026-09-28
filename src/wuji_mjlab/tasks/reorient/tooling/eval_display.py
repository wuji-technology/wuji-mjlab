# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Real-time terminal display for reorient eval progress."""

from __future__ import annotations

import sys
import time

import numpy as np


class EvalDisplay:
  """Real-time terminal display for evaluation progress."""

  DISPLAY_LINES = 20

  def __init__(self):
    self._initialized = False
    self._is_tty = sys.stdout.isatty()
    self._last_update = 0

  def _init(self):
    if self._initialized:
      return
    if self._is_tty:
      sys.stdout.write("\033[2J\033[H")
      sys.stdout.flush()
    self._initialized = True

  def update(
    self,
    trial: int,
    num_trials: int,
    sim_elapsed: float,
    trial_timeout: float,
    ori_error: float,
    success_threshold: float,
    successes: int,
    drops: int,
    timeouts: int,
    result: str = "",
    force: bool = False,
    hold_counter: int = 0,
    hold_steps: int = 1,
    contact_info: dict | None = None,
    actuator_force: np.ndarray | None = None,
    cube_cvel: np.ndarray | None = None,
    cube_cacc: np.ndarray | None = None,
  ):
    now = time.time()
    if not force and now - self._last_update < 0.1:
      return
    self._last_update = now
    self._init()

    completed = successes + drops + timeouts
    rate = successes / max(completed, 1) * 100

    pct = sim_elapsed / trial_timeout
    bar_len = 30
    filled = int(bar_len * min(pct, 1.0))
    bar = "#" * filled + "-" * (bar_len - filled)

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(
      f"  EVAL  Trial {trial + 1:3d}/{num_trials}  |  "
      f"Success Rate: {rate:5.1f}%  ({successes}/{max(completed, 1)})"
    )
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  Time:  [{bar}] {sim_elapsed:5.1f}s / {trial_timeout:.0f}s")
    lines.append(
      f"  Ori Error: {np.degrees(ori_error):6.1f} deg  "
      f"(threshold: {np.degrees(success_threshold):.1f} deg)"
    )
    err_bar_len = 40
    err_ratio = min(ori_error / np.pi, 1.0)
    thresh_pos = int(err_bar_len * (success_threshold / np.pi))
    err_pos = int(err_bar_len * err_ratio)
    err_bar = list("." * err_bar_len)
    if thresh_pos < err_bar_len:
      err_bar[thresh_pos] = "|"
    if err_pos < err_bar_len:
      err_bar[err_pos] = "*"
    lines.append(
      f"  Error:  [{''.join(err_bar)}] {'OK' if ori_error < success_threshold else ''}"
    )
    hold_bar_len = 30
    hold_filled = int(hold_bar_len * min(hold_counter / max(hold_steps, 1), 1.0))
    hold_bar = "#" * hold_filled + "-" * (hold_bar_len - hold_filled)
    hold_status = "HOLDING" if hold_counter > 0 else ""
    lines.append(
      f"  Hold:   [{hold_bar}] {hold_counter:3d}/{hold_steps} steps  {hold_status}"
    )
    lines.append("")

    lines.append("-" * 70)
    if contact_info and contact_info["contacts"]:
      cl = contact_info["contacts"]
      rf = contact_info["resultant_force"]
      rt = contact_info["resultant_torque"]
      rf_mag = np.linalg.norm(rf)
      rt_mag = np.linalg.norm(rt)
      lines.append(
        f"  Contacts ({len(cl)} active)  |  "
        f"Resultant: F={rf_mag:>7.2f}N  T={rt_mag:>7.3f}Nm"
      )
      lines.append(
        f"    F=[{rf[0]:+7.2f}, {rf[1]:+7.2f}, {rf[2]:+7.2f}]N  "
        f"T=[{rt[0]:+7.3f}, {rt[1]:+7.3f}, {rt[2]:+7.3f}]Nm"
      )
      for ci in cl[:5]:
        g1, g2 = ci["geom1"], ci["geom2"]
        fn, ft = ci["normal_force"], ci["total_force"]
        px, py, pz = ci["pos"]
        lines.append(
          f"    {g1:>20s} <-> {g2:<20s}  "
          f"F={ft:6.2f}N (n={fn:6.2f})  "
          f"pos=({px:+.3f},{py:+.3f},{pz:+.3f})"
        )
      if len(cl) > 5:
        lines.append(f"    ... and {len(cl) - 5} more contacts")
    else:
      lines.append("  Contacts: none")
    contact_section_height = 10
    while len(lines) < 10 + contact_section_height:
      lines.append("")
    lines.append("-" * 70)

    if cube_cvel is not None:
      angvel = cube_cvel[:3]
      linvel = cube_cvel[3:]
      angacc = cube_cacc[:3] if cube_cacc is not None else np.zeros(3)
      linacc = cube_cacc[3:] if cube_cacc is not None else np.zeros(3)
      lines.append(
        f"  Lin Vel: |v|={np.linalg.norm(linvel):7.3f} m/s     "
        f"Lin Acc: |a|={np.linalg.norm(linacc):7.1f} m/s^2"
      )
      lines.append(
        f"  Ang Vel: |w|={np.linalg.norm(angvel):7.3f} rad/s   "
        f"Ang Acc: |a|={np.linalg.norm(angacc):7.1f} rad/s^2"
      )
      lines.append("-" * 70)

    if actuator_force is not None and len(actuator_force) >= 20:
      lines.append("  Joint Torques (Nm):")
      lines.append(
        f"  {'':8s}  {'Finger1':>10s}  {'Finger2':>10s}  "
        f"{'Finger3':>10s}  {'Finger4':>10s}  {'Finger5':>10s}"
      )
      for j in range(4):
        vals = [actuator_force[f * 4 + j] for f in range(5)]
        lines.append(
          f"  Joint {j + 1}:  {vals[0]:>+10.4f}  {vals[1]:>+10.4f}  "
          f"{vals[2]:>+10.4f}  {vals[3]:>+10.4f}  {vals[4]:>+10.4f}"
        )
      lines.append("-" * 70)

    lines.append(
      f"  Successes: {successes:3d}  |  Drops: {drops:3d}  |  Timeouts: {timeouts:3d}"
    )
    if result:
      color = (
        "\033[92m"
        if result == "success"
        else "\033[91m"
        if result == "drop"
        else "\033[93m"
      )
      lines.append(f"  Last result: {color}{result.upper()}\033[0m")
    else:
      lines.append("")
    lines.append("=" * 70)
    lines.append("  Close viewer window to abort early")

    if self._is_tty:
      output = "\033[H"
      for i, line in enumerate(lines):
        output += f"\033[{i + 1};1H\033[K{line}"
      sys.stdout.write(output)
      sys.stdout.flush()
    else:
      print("\n".join(lines), flush=True)
