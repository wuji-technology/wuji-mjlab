# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Importable core for the reorient ONNX success-rate CLI.

Hosts the per-policy obs builder, ONNX path resolver, structured eval
config/result dataclasses, and the programmatic eval entry point
``run_eval()`` so the thin CLI wrapper at
``scripts/eval_success_rate.py`` reduces to a few lines.

The CLI script is responsible for setting any environment variables that
must be in place before MuJoCo GL initialises (e.g. ``MUJOCO_GL``); this
module only sets up Python state and is safe to import without spawning a
sim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

from wuji_mjlab.tasks.reorient.tooling.eval_display import EvalDisplay
from wuji_mjlab.tasks.reorient.tooling.scene_builder import (
  HoldState,
  SceneMetadata,
  apply_action,
  build_reorient_scene,
  check_hold,
  compute_cube_ori_error_6d,
  compute_cube_pos_in_tag,
  compute_joint_pos_normalized,
  compute_joint_pos_target_error,
  get_contact_info,
  load_config,
  quat_error_magnitude,
  random_quat_uniform,
  reset_scene,
  set_goal_mocap,
)

__all__ = [
  "EvalConfig",
  "EvalResult",
  "TrialOutcome",
  "EvalDisplay",
  "ObsBuilder",
  "resolve_onnx_path",
  "run_eval",
  "main",
]

# Resolve project root (where rsl_rl writes ``logs/`` by default).
# parents[4] walks tooling -> reorient -> tasks -> wuji_mjlab -> src -> repo root.
_TOOLING_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TOOLING_DIR.parents[4]


# =============================================================================
# ONNX path resolution
# =============================================================================


def resolve_onnx_path(onnx_input: str) -> Path:
  """Resolve ONNX path from input string.

  Tries: direct path → logs/<name>/policy.onnx → logs/<task>/<name>/policy.onnx
  → partial name match in logs tree.
  """
  onnx_path = Path(onnx_input)
  if onnx_path.exists():
    return onnx_path

  logs_dir = _PROJECT_ROOT / "logs"
  if logs_dir.exists():
    candidate = logs_dir / onnx_input / "policy.onnx"
    if candidate.exists():
      return candidate
    for task_dir in sorted(logs_dir.iterdir()):
      if not task_dir.is_dir():
        continue
      candidate = task_dir / onnx_input / "policy.onnx"
      if candidate.exists():
        return candidate
      for exp_dir in sorted(task_dir.iterdir(), reverse=True):
        if onnx_input in exp_dir.name:
          candidate = exp_dir / "policy.onnx"
          if candidate.exists():
            return candidate

  print(f"ERROR: ONNX file not found: {onnx_input}")
  sys.exit(1)


# =============================================================================
# Obs builder
# =============================================================================


class ObsBuilder:
  """Build observation vector with per-term history buffers.

  Each term has its own history length. flatten_history_dim=True produces
  term-major ordering: [term_A_t0..term_A_tH, term_B_t0..term_B_tH, ...]

  Obs terms (matching reorient_env_cfg.py policy group):
    1. noisy_joint_angles  (A,) × H=1 = A    — joint_pos_limit_normalized
    2. qpos_error          (A,) × H=1 = A    — joint_pos_target_error
    3. cube_pos_in_tag      (3,) × H=1 =  3   — cube_pos_in_tag (absolute tag frame)
    4. cube_ori_error       (6,) × H=1 =  6   — goal_rot_err_6d
    5. action_history      (A,) × H=1 = A    — previous_raw_action
  """

  def __init__(self, history_length: int = 1, action_dim: int = 20):
    term_dims = {
      "joint_angles": action_dim,
      "qpos_error": action_dim,
      "cube_pos_in_tag": 3,
      "cube_ori_error": 6,
      "action_history": action_dim,
    }
    # (dim, H) per term — all terms share the same history at the moment
    self.TERM_SPECS = {
      name: (dim, history_length) for name, dim in term_dims.items()
    }
    self._buffers: dict[str, deque] = {}
    self._init_buffers()

  def _init_buffers(self) -> None:
    """Create empty deque buffers for each term.

    The first post-reset frame is backfilled across the whole history,
    matching mjlab CircularBuffer.append() semantics.
    """
    for name, (_dim, H) in self.TERM_SPECS.items():
      self._buffers[name] = deque(maxlen=H)

  def reset(self) -> None:
    """Clear all buffers.

    The next build() call backfills each term history with that first frame.
    """
    self._init_buffers()

  def _append_with_backfill(self, name: str, value: np.ndarray) -> None:
    """Append a term value, backfilling first frame across history."""
    value = np.asarray(value, dtype=np.float32)
    buf = self._buffers[name]
    history_len = self.TERM_SPECS[name][1]
    if len(buf) == 0:
      buf.extend(value.copy() for _ in range(history_len))
    else:
      buf.append(value.copy())

  @property
  def obs_size(self) -> int:
    return sum(dim * H for dim, H in self.TERM_SPECS.values())

  def build(
    self,
    scene: SceneMetadata,
    prev_target: np.ndarray,
    goal_quat: np.ndarray,
    last_action: np.ndarray,
  ) -> np.ndarray:
    """Compute current obs terms, push into history, return flattened obs.

    CircularBuffer.buffer returns chronological order (oldest → newest),
    which matches deque's natural iteration order.

    Args:
        last_action: Raw action from the previous policy step. This matches
            training's ``previous_raw_action`` observation term semantics.
    """
    # Compute raw term values
    joint_angles = compute_joint_pos_normalized(scene)
    qpos_error = compute_joint_pos_target_error(scene, prev_target)
    cube_pos_in_tag = compute_cube_pos_in_tag(scene)
    cube_ori_error = compute_cube_ori_error_6d(scene, goal_quat)
    action_obs = last_action.astype(np.float32)

    # Push into history buffers (append = newest at end)
    self._append_with_backfill("joint_angles", joint_angles)
    self._append_with_backfill("qpos_error", qpos_error)
    self._append_with_backfill("cube_pos_in_tag", cube_pos_in_tag)
    self._append_with_backfill("cube_ori_error", cube_ori_error)
    self._append_with_backfill("action_history", action_obs)

    # Flatten: term-major, oldest → newest within each term
    parts = []
    for name in self.TERM_SPECS:
      buf = self._buffers[name]
      for obs_t in buf:  # deque iterates oldest → newest
        parts.append(obs_t)

    return np.concatenate(parts).astype(np.float32)


# =============================================================================
# Structured config / result dataclasses
# =============================================================================


@dataclass(frozen=True)
class EvalConfig:
  """All inputs needed to evaluate a reorient ONNX policy."""

  onnx_path: Path
  num_trials: int = 100
  trial_timeout: float = 14.0
  success_threshold: float = 0.2
  success_hold_steps: int = 5
  goal_switch_delay: int = 20
  drop_height: float = -0.15
  no_viewer: bool = False
  action_scale: float | None = None
  ema_alpha: float | None = None
  cube_edge_m: float | None = None
  json_output: Path | None = None
  warmup_time_s: float = 0.4
  task_id: str | None = None


@dataclass(frozen=True)
class TrialOutcome:
  trial_idx: int
  status: Literal["success", "drop", "timeout"]
  time_to_first_success_s: float | None
  goal_reaches: int
  final_ori_error_rad: float
  min_ori_error_rad: float


@dataclass(frozen=True)
class EvalResult:
  config: EvalConfig
  onnx_path_resolved: Path
  success_rate: float
  drop_rate: float
  timeout_rate: float
  mean_goal_reaches: float
  mean_time_to_first_success_s: float | None
  mean_min_ori_error_rad: float
  trials: list[TrialOutcome] = field(default_factory=list)
  train_config: dict[str, Any] = field(default_factory=dict)

  def to_json(self) -> dict[str, Any]:
    """JSON-serializable dict; paths rendered as strings."""
    cfg = self.config
    return {
      "config": {
        "onnx_path": str(cfg.onnx_path),
        "num_trials": cfg.num_trials,
        "trial_timeout": cfg.trial_timeout,
        "success_threshold": cfg.success_threshold,
        "success_hold_steps": cfg.success_hold_steps,
        "goal_switch_delay": cfg.goal_switch_delay,
        "drop_height": cfg.drop_height,
        "no_viewer": cfg.no_viewer,
        "action_scale": cfg.action_scale,
        "ema_alpha": cfg.ema_alpha,
        "cube_edge_m": cfg.cube_edge_m,
        "json_output": str(cfg.json_output) if cfg.json_output is not None else None,
        "warmup_time_s": cfg.warmup_time_s,
        "task_id": cfg.task_id,
      },
      "onnx_path_resolved": str(self.onnx_path_resolved),
      "success_rate": self.success_rate,
      "drop_rate": self.drop_rate,
      "timeout_rate": self.timeout_rate,
      "mean_goal_reaches": self.mean_goal_reaches,
      "mean_time_to_first_success_s": self.mean_time_to_first_success_s,
      "mean_min_ori_error_rad": self.mean_min_ori_error_rad,
      "trials": [
        {
          "trial_idx": t.trial_idx,
          "status": t.status,
          "time_to_first_success_s": t.time_to_first_success_s,
          "goal_reaches": t.goal_reaches,
          "final_ori_error_rad": t.final_ori_error_rad,
          "min_ori_error_rad": t.min_ori_error_rad,
        }
        for t in self.trials
      ],
      "train_config": dict(self.train_config),
    }


# =============================================================================
# Programmatic eval entry
# =============================================================================


def run_eval(config: EvalConfig) -> EvalResult:
  """Pure programmatic eval entry. Spawns mjlab/mujoco state internally."""
  # ----- Resolve ONNX path and load config -----
  onnx_path = resolve_onnx_path(str(config.onnx_path))
  train_config = load_config(onnx_path.parent)

  # ----- Cube size: explicit override > train_config["cube_edge_m"] > scene default (54mm) -----
  cube_edge_m = (
    config.cube_edge_m
    if config.cube_edge_m is not None
    else train_config.get("cube_edge_m")
  )
  task_id = config.task_id or train_config.get("task_id") or "WujiHand_Reorient"

  # ----- Build scene -----
  sim_dt = train_config.get("sim_dt", 0.01)
  ctrl_dt = train_config.get("ctrl_dt", 0.05)
  scene = build_reorient_scene(
    sim_dt=sim_dt,
    ctrl_dt=ctrl_dt,
    cube_edge_m=cube_edge_m,
    task_id=task_id,
  )

  # ----- Load ONNX policy -----
  print(f"Loading ONNX model: {onnx_path}")
  session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  input_name = session.get_inputs()[0].name
  output_names = [o.name for o in session.get_outputs()]
  onnx_obs_size = session.get_inputs()[0].shape[1]

  # ----- Determine parameters from config -----
  action_scale = config.action_scale or train_config.get("action_scale", 0.5)
  ema_alpha = config.ema_alpha or train_config.get("ema_alpha", 0.5)
  history_len = train_config.get("history_len", 1)

  # Validate obs size
  obs_builder = ObsBuilder(history_length=history_len, action_dim=scene.action_dim)
  expected_obs = obs_builder.obs_size
  if expected_obs != onnx_obs_size:
    print(f"ERROR: Expected obs size {expected_obs} but ONNX expects {onnx_obs_size}.")
    sys.exit(1)

  # ----- Eval parameters -----
  num_trials = config.num_trials
  trial_timeout = config.trial_timeout
  success_threshold = config.success_threshold
  success_hold_steps = config.success_hold_steps
  goal_switch_delay = config.goal_switch_delay
  warmup_time_s = config.warmup_time_s

  # Drop detection: convert relative offset to absolute z threshold.
  # The mjlab hand sits at z=0.5 with a ground plane at z=0 that catches the
  # cube, so we use: default_cube_z + drop_height (e.g. 0.52 + (-0.15) = 0.37).
  drop_z_threshold = scene.default_cube_pos[2] + config.drop_height

  # ----- Print config -----
  print(f"\n{'=' * 60}")
  print("Automated Success Rate Evaluation (mjlab)")
  print(f"{'=' * 60}")
  print(f"ONNX:      {onnx_path}")
  print(f"Trials:    {num_trials}")
  print(f"Timeout:   {trial_timeout}s per trial")
  print(
    f"Threshold: {success_threshold:.2f} rad ({np.degrees(success_threshold):.1f} deg)"
  )
  print(f"Drop h:    {config.drop_height}m (abs z threshold: {drop_z_threshold:.2f}m)")
  cube_edge_desc = (
    f"{cube_edge_m * 1000:.1f} mm"
    if cube_edge_m is not None
    else "54.0 mm (baseline default)"
  )
  print(f"Cube:      {cube_edge_desc}")
  print(
    f"Control:   absolute (history={history_len}, scale={action_scale}, ema={ema_alpha})"
  )
  print(
    f"Hold:      {success_hold_steps} control steps ({success_hold_steps * ctrl_dt:.1f}s), switch delay={goal_switch_delay}"
  )
  print(f"Obs size:  {obs_builder.obs_size}")
  print(f"Scene:     task={task_id}, nq={scene.model.nq}, nu={scene.model.nu}")
  print(f"Viewer:    {'disabled' if config.no_viewer else 'enabled'}")
  print(f"{'=' * 60}\n")

  time.sleep(0.5)

  # ----- Statistics -----
  successes = 0
  drops = 0
  timeouts_count = 0
  trial_outcomes: list[TrialOutcome] = []
  last_result_str = ""

  display = EvalDisplay()

  # ----- Cube motion log -----
  motion_log_path = onnx_path.parent / "cube_motion_log.csv"
  motion_log = open(motion_log_path, "w")
  motion_log.write("time,trial,linvel,linacc,angvel,angacc\n")
  gravity = scene.model.opt.gravity
  print(f"Cube motion log: {motion_log_path}")

  # ----- Helper: run a single trial step -----
  def run_policy_step(
    goal_quat: np.ndarray,
    prev_target: np.ndarray,
    last_action: np.ndarray,
    episode_step: int,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One control step: build obs → infer → apply action. Returns (action, new_prev_target, obs)."""
    obs = obs_builder.build(scene, prev_target, goal_quat, last_action)
    onnx_input = {input_name: obs.reshape(1, -1)}
    action = session.run(output_names, onnx_input)[0][0]
    new_target = apply_action(
      scene,
      action,
      prev_target,
      episode_step,
      action_scale=action_scale,
      ema_alpha=ema_alpha,
      warmup_time_s=warmup_time_s,
    )
    return action, new_target, obs

  # ----- Run evaluation -----
  def run_eval_loop(viewer=None):
    nonlocal successes, drops, timeouts_count, last_result_str

    need_reset = True
    # Persistent state across trials (carry forward on success, reset on drop/timeout)
    prev_target = scene.default_joint_pos.copy()
    last_action = np.zeros(scene.action_dim, dtype=np.float32)
    episode_step = 0

    for trial in range(num_trials):
      if viewer is not None and not viewer.is_running():
        print("\nViewer closed. Stopping evaluation.")
        break

      # Full reset on drop/timeout (first trial is always need_reset=True)
      if need_reset:
        reset_scene(scene)
        obs_builder.reset()
        prev_target = scene.default_joint_pos.copy()
        last_action = np.zeros(scene.action_dim, dtype=np.float32)
        episode_step = 0

      # Set random goal (>90 deg from current cube orientation)
      current_cube_quat = scene.cube_quat
      goal_q = random_quat_uniform()
      for _ in range(1000):
        goal_q = random_quat_uniform()
        if quat_error_magnitude(goal_q, current_cube_quat) >= np.pi / 2:
          break
      set_goal_mocap(scene, goal_q)
      mujoco.mj_forward(scene.model, scene.data)

      # Per-trial tracking (always reset)
      trial_start_sim = scene.data.time
      trial_start_wall = time.time()
      result = "timeout"
      min_ori_error = np.pi
      current_ori_error = np.pi
      hold_state = HoldState()
      time_to_first_success_s: float | None = None
      goal_reaches = 0
      was_above_threshold = True  # for rising-edge goal reach counting

      while True:
        if viewer is not None and not viewer.is_running():
          break

        sim_elapsed = scene.data.time - trial_start_sim

        # Timeout check
        if sim_elapsed >= trial_timeout:
          result = "timeout"
          break

        # Policy step
        action, prev_target, obs = run_policy_step(
          goal_q,
          prev_target,
          last_action,
          episode_step,
        )
        last_action = action.copy()
        episode_step += 1

        # Physics
        for _ in range(scene.n_substeps):
          mujoco.mj_step(scene.model, scene.data)

        # Keep goal visualization directly above the live cube.
        set_goal_mocap(scene, goal_q)

        # Check drop
        cube_z = scene.cube_pos[2]
        if cube_z < drop_z_threshold:
          result = "drop"
          break

        # Check success (two-phase state machine)
        current_ori_error = quat_error_magnitude(scene.cube_quat, goal_q)
        min_ori_error = min(min_ori_error, current_ori_error)
        hold_state, hold_events = check_hold(
          current_ori_error,
          hold_state,
          threshold=success_threshold,
          success_hold_steps=success_hold_steps,
          goal_switch_delay=goal_switch_delay,
        )
        # Count goal reaches as rising-edge crossings into the threshold.
        if current_ori_error < success_threshold:
          if was_above_threshold:
            goal_reaches += 1
            if time_to_first_success_s is None:
              time_to_first_success_s = scene.data.time - trial_start_sim
          was_above_threshold = False
        else:
          was_above_threshold = True
        if hold_events.success_achieved:
          result = "success"
          break

        # Log cube motion
        cvel = scene.data.cvel[scene.cube_body_id]
        cacc = scene.data.cacc[scene.cube_body_id]
        lv = np.linalg.norm(cvel[3:])
        la = np.linalg.norm(cacc[3:] - gravity)
        av = np.linalg.norm(cvel[:3])
        aa = np.linalg.norm(cacc[:3])
        motion_log.write(
          f"{scene.data.time:.4f},{trial + 1},{lv:.6f},{la:.4f},{av:.6f},{aa:.4f}\n"
        )

        # Contact info and display
        contacts = get_contact_info(scene.model, scene.data)
        display.update(
          trial=trial,
          num_trials=num_trials,
          sim_elapsed=sim_elapsed,
          trial_timeout=trial_timeout,
          ori_error=current_ori_error,
          success_threshold=success_threshold,
          successes=successes,
          drops=drops,
          timeouts=timeouts_count,
          result=last_result_str,
          hold_counter=hold_state.hold_counter,
          hold_steps=success_hold_steps,
          contact_info=contacts,
          actuator_force=scene.data.actuator_force,
          cube_cvel=scene.data.cvel[scene.cube_body_id],
          cube_cacc=(
            scene.data.cacc[scene.cube_body_id] - np.array([0, 0, 0, *gravity])
          ),
        )

        # Real-time pacing (only with viewer)
        if viewer is not None:
          target_wall = trial_start_wall + (scene.data.time - trial_start_sim)
          sleep_time = target_wall - time.time()
          if sleep_time > 0.001:
            time.sleep(sleep_time)
          viewer.sync()

      # Skip recording if viewer closed mid-trial
      if viewer is not None and not viewer.is_running() and result == "timeout":
        break

      # Record result
      if result == "success":
        successes += 1
        need_reset = False
      elif result == "drop":
        drops += 1
        need_reset = True
      else:
        timeouts_count += 1
        need_reset = True

      last_result_str = result
      trial_outcomes.append(
        TrialOutcome(
          trial_idx=trial,
          status=result,  # type: ignore[arg-type]
          time_to_first_success_s=time_to_first_success_s,
          goal_reaches=goal_reaches,
          final_ori_error_rad=float(current_ori_error),
          min_ori_error_rad=float(min_ori_error),
        )
      )

      # Force display update showing result
      display.update(
        trial=trial,
        num_trials=num_trials,
        sim_elapsed=scene.data.time - trial_start_sim,
        trial_timeout=trial_timeout,
        ori_error=current_ori_error,
        success_threshold=success_threshold,
        successes=successes,
        drops=drops,
        timeouts=timeouts_count,
        result=result,
        force=True,
        hold_counter=hold_state.hold_counter,
        hold_steps=success_hold_steps,
        contact_info=get_contact_info(scene.model, scene.data),
        actuator_force=scene.data.actuator_force,
        cube_cvel=scene.data.cvel[scene.cube_body_id],
        cube_cacc=(scene.data.cacc[scene.cube_body_id] - np.array([0, 0, 0, *gravity])),
      )
      if viewer is not None and viewer.is_running():
        viewer.sync()

  # ----- Launch -----
  if config.no_viewer:
    run_eval_loop(viewer=None)
  else:
    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
      run_eval_loop(viewer=viewer)

  # ===== Close motion log =====
  motion_log.close()
  print(f"\nCube motion log saved to: {motion_log_path}")

  # ===== Aggregate rates =====
  completed = max(len(trial_outcomes), 1)
  success_rate = successes / completed
  drop_rate = drops / completed
  timeout_rate = timeouts_count / completed
  mean_goal_reaches = (
    float(np.mean([t.goal_reaches for t in trial_outcomes])) if trial_outcomes else 0.0
  )
  first_successes = [
    t.time_to_first_success_s
    for t in trial_outcomes
    if t.time_to_first_success_s is not None
  ]
  mean_time_to_first_success_s = (
    float(np.mean(first_successes)) if first_successes else None
  )
  mean_min_ori_error_rad = (
    float(np.mean([t.min_ori_error_rad for t in trial_outcomes]))
    if trial_outcomes
    else 0.0
  )

  # ===== Side-effect: write eval_results.json next to the policy =====
  legacy_results_path = onnx_path.parent / "eval_results.json"
  legacy_payload = {
    "onnx_path": str(onnx_path),
    "num_trials": num_trials,
    "completed": len(trial_outcomes),
    "success_threshold_rad": success_threshold,
    "success_hold_steps": success_hold_steps,
    "goal_switch_delay": goal_switch_delay,
    "trial_timeout_s": trial_timeout,
    "successes": successes,
    "drops": drops,
    "timeouts": timeouts_count,
    "success_rate": success_rate,
    "trials": [
      {
        "trial": t.trial_idx + 1,
        "result": t.status,
        "min_ori_error_deg": float(np.degrees(t.min_ori_error_rad)),
        "goal_reaches": t.goal_reaches,
        "time_to_first_success_s": t.time_to_first_success_s,
      }
      for t in trial_outcomes
    ],
  }
  try:
    with open(legacy_results_path, "w") as f:
      json.dump(legacy_payload, f, indent=2)
    print(f"\n  Results saved to: {legacy_results_path}")
  except Exception as e:
    print(f"\n  Warning: Could not save results: {e}")

  return EvalResult(
    config=config,
    onnx_path_resolved=onnx_path,
    success_rate=success_rate,
    drop_rate=drop_rate,
    timeout_rate=timeout_rate,
    mean_goal_reaches=mean_goal_reaches,
    mean_time_to_first_success_s=mean_time_to_first_success_s,
    mean_min_ori_error_rad=mean_min_ori_error_rad,
    trials=trial_outcomes,
    train_config=train_config,
  )


# =============================================================================
# CLI plumbing
# =============================================================================


def _parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Evaluate reorient policy success rate in MuJoCo (mjlab)."
  )
  parser.add_argument("onnx_path", help="Path to ONNX model (or experiment name)")
  parser.add_argument(
    "--num-trials", type=int, default=100, help="Number of test trials"
  )
  parser.add_argument(
    "--trial-timeout", type=float, default=14.0, help="Trial timeout (seconds)"
  )
  parser.add_argument(
    "--success-threshold", type=float, default=0.2, help="Success angle threshold (rad)"
  )
  parser.add_argument(
    "--success-hold-steps",
    type=int,
    default=5,
    help=(
      "Consecutive steps at goal to confirm success "
      "(must match training InHandReorientCommandCfg.success_hold_steps)"
    ),
  )
  parser.add_argument(
    "--goal-switch-delay",
    type=int,
    default=20,
    help=(
      "Steps in SUCCESS_WINDOW before goal switches "
      "(must match training InHandReorientCommandCfg.goal_switch_delay)"
    ),
  )
  parser.add_argument(
    "--drop-height", type=float, default=-0.15, help="Drop detection height (m)"
  )
  parser.add_argument(
    "--no-viewer", action="store_true", help="Disable viewer (headless)"
  )
  parser.add_argument(
    "--action-scale", type=float, default=None, help="Override action scale"
  )
  parser.add_argument(
    "--ema-alpha", type=float, default=None, help="Override EMA alpha"
  )
  parser.add_argument(
    "--cube-edge-m",
    type=float,
    default=None,
    help=(
      "Override cube edge length in metres (e.g. 0.0405 for 40.5 mm). "
      "When unset, falls back to train_config['cube_edge_m'] if present, "
      "else the scene default (54 mm)."
    ),
  )
  parser.add_argument(
    "--json-output",
    type=Path,
    default=None,
    help="If set, write structured EvalResult to this path as JSON.",
  )
  parser.add_argument(
    "--task-id",
    default=None,
    help="Override task id for standalone scene construction.",
  )
  return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> EvalConfig:
  return EvalConfig(
    onnx_path=Path(args.onnx_path),
    num_trials=args.num_trials,
    trial_timeout=args.trial_timeout,
    success_threshold=args.success_threshold,
    success_hold_steps=args.success_hold_steps,
    goal_switch_delay=args.goal_switch_delay,
    drop_height=args.drop_height,
    no_viewer=args.no_viewer,
    action_scale=args.action_scale,
    ema_alpha=args.ema_alpha,
    cube_edge_m=args.cube_edge_m,
    json_output=args.json_output,
    task_id=args.task_id,
  )


def _print_terminal_summary(result: EvalResult) -> None:
  cfg = result.config
  ctrl_dt = result.train_config.get("ctrl_dt", 0.05)
  completed = len(result.trials)
  successes = sum(1 for t in result.trials if t.status == "success")
  drops = sum(1 for t in result.trials if t.status == "drop")
  timeouts_count = sum(1 for t in result.trials if t.status == "timeout")

  if sys.stdout.isatty():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

  print(f"\n{'=' * 60}")
  print("  EVALUATION COMPLETE")
  print(f"{'=' * 60}")
  print(f"  Model:     {result.onnx_path_resolved.name}")
  print(f"  Trials:    {completed} / {cfg.num_trials}")
  print(
    f"  Threshold: {cfg.success_threshold:.2f} rad "
    f"({np.degrees(cfg.success_threshold):.1f} deg)"
  )
  print(
    f"  Hold:      {cfg.success_hold_steps} steps "
    f"({cfg.success_hold_steps * ctrl_dt:.1f}s), "
    f"switch delay={cfg.goal_switch_delay}"
  )
  print(f"{'=' * 60}")
  print(f"  Successes: {successes:4d}  ({successes / max(completed, 1) * 100:5.1f}%)")
  print(f"  Drops:     {drops:4d}  ({drops / max(completed, 1) * 100:5.1f}%)")
  print(
    f"  Timeouts:  {timeouts_count:4d}  ({timeouts_count / max(completed, 1) * 100:5.1f}%)"
  )
  print(f"{'=' * 60}")

  if result.trials:
    errors = [float(np.degrees(t.min_ori_error_rad)) for t in result.trials]
    print(f"  Avg min orientation error:    {np.mean(errors):6.1f} deg")
    print(f"  Median min orientation error: {np.median(errors):6.1f} deg")
    print(f"  Mean goal reaches / trial:    {result.mean_goal_reaches:6.2f}")
    if result.mean_time_to_first_success_s is not None:
      print(
        f"  Mean time-to-first-success:   {result.mean_time_to_first_success_s:6.2f} s"
      )

  print(f"{'=' * 60}")

  # === Cube motion statistics (read back from CSV side-effect) ===
  motion_log_path = result.onnx_path_resolved.parent / "cube_motion_log.csv"
  try:
    import csv

    with open(motion_log_path) as f:
      reader = csv.DictReader(f)
      rows = list(reader)
    if rows:
      lv_mag = np.array([float(r["linvel"]) for r in rows])
      la_mag = np.array([float(r["linacc"]) for r in rows])
      av_mag = np.array([float(r["angvel"]) for r in rows])
      aa_mag = np.array([float(r["angacc"]) for r in rows])

      print(f"\n{'=' * 60}")
      print(f"  Cube Motion Statistics ({len(rows)} samples)")
      print(f"{'=' * 60}")
      hdr = (
        f"  {'':16s}  {'mean':>8s}  {'p50':>8s}  {'p95':>8s}  {'p99':>8s}  {'max':>8s}"
      )
      print(hdr)
      for label, arr in [
        ("Lin Vel (m/s)", lv_mag),
        ("Lin Acc (m/s2)", la_mag),
        ("Ang Vel (rad/s)", av_mag),
        ("Ang Acc (rad/s2)", aa_mag),
      ]:
        fmt = ".3f" if "Vel" in label else ".1f"
        print(
          f"  {label:16s}  {np.mean(arr):>8{fmt}}  {np.median(arr):>8{fmt}}  "
          f"{np.percentile(arr, 95):>8{fmt}}  {np.percentile(arr, 99):>8{fmt}}  "
          f"{np.max(arr):>8{fmt}}"
        )
      print(f"{'=' * 60}")
  except Exception as e:
    print(f"  Warning: Could not parse motion log: {e}")

  print()


def main() -> None:
  args = _parse_cli_args()
  config = _config_from_args(args)
  result = run_eval(config)
  if config.json_output is not None:
    config.json_output.write_text(json.dumps(result.to_json(), indent=2))
    print(f"  Structured result written to: {config.json_output}")
  _print_terminal_summary(result)
