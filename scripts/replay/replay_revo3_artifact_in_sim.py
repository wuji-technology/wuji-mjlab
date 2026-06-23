# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Replay a saved Revo3 play artifact back inside the simulator.

The Revo3 216-D artifact stores raw policy observations and raw policy actions.
In the default mode this script does not run the policy; it simply feeds the
saved action sequence back through the task action manager and records a
comparison log against the saved rollout fields.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# CI and sandboxed shells can make the default import-time cache locations
# read-only. Keep this before importing mjlab.
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import wuji_mjlab.tasks  # noqa: F401
import yaml
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from scripts.play.play_rsl_rl_zero_wrist_pitch import (
  _DEFAULT_FIXED_GOAL_AXIS,
  _DEFAULT_FIXED_GOAL_DEG,
  _as_offset_tuple,
  _force_fixed_object_init,
  _force_fixed_reorient_goal,
  _force_single_reorient_goal,
  _force_zero_robot_pitch_reset,
  _resolve_fixed_object_offset,
)
from wuji_mjlab.utils.cli_override_utils import DEFAULT_TASK_ALIASES, resolve_task
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

INPUT_DIM = 216
ACTION_DIM = 21
INPUT_NAME = "obs"
OUTPUT_NAME = "actions"
QPOS_ERROR_SLICE = slice(63, 126)


Mode = Literal["action_open_loop", "policy_recompute"]
ViewerKind = Literal["none", "auto", "native", "viser"]
ActionSource = Literal["auto", "action_raw", "action_clipped"]
PolicyStepSource = Literal["recomputed", "saved"]


@dataclass(frozen=True)
class ReplayOptions:
  artifact_dir: Path
  task: str | None
  mode: Mode
  viewer: ViewerKind
  num_envs: int
  device: str | None
  max_steps: int | None
  start_index: int
  loop: bool
  output_dir: Path | None
  action_source: ActionSource
  policy_step_source: PolicyStepSource
  compare_only: bool
  fixed_object_init: bool | None
  single_goal: bool | None
  fixed_goal: bool | None
  fixed_goal_axis: str | None
  fixed_goal_deg: float | None
  cube_offset_in_palm: tuple[float, float, float] | None
  no_terminations: bool | None
  video: bool
  video_dir: Path | None
  video_length: int | None
  video_width: int | None
  video_height: int | None
  viewer_frame_sleep: float


class _NoOpPolicy:
  def __call__(self, obs: Any) -> torch.Tensor:
    del obs
    raise RuntimeError("Replay viewer is synced manually and should not call a policy.")


class _FiniteViewer(AbstractContextManager):
  def __init__(
    self,
    viewer: ViewerKind,
    env: RslRlVecEnvWrapper,
    *,
    frame_sleep: float,
  ) -> None:
    self._requested = viewer
    self._env = env
    self._frame_sleep = max(float(frame_sleep), 0.0)
    self._viewer: NativeMujocoViewer | ViserPlayViewer | None = None

  def __enter__(self) -> "_FiniteViewer":
    resolved = _resolve_viewer(self._requested)
    if resolved == "none":
      return self
    if resolved == "native":
      self._viewer = NativeMujocoViewer(self._env, _NoOpPolicy())
    elif resolved == "viser":
      self._viewer = ViserPlayViewer(self._env, _NoOpPolicy())
    else:
      raise RuntimeError(f"Unsupported viewer backend: {resolved}")
    self._viewer.setup()
    self.sync()
    return self

  def sync(self) -> None:
    if self._viewer is None:
      return
    if isinstance(self._viewer, NativeMujocoViewer) and not self._viewer.is_running():
      return
    self._viewer.sync_env_to_viewer()
    if self._frame_sleep > 0.0:
      time.sleep(self._frame_sleep)

  def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool | None:
    if self._viewer is not None:
      self._viewer.close()
    return None


def _resolve_viewer(viewer: ViewerKind) -> Literal["none", "native", "viser"]:
  if viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return "native" if has_display else "none"
  return viewer


def _load_yaml(path: Path) -> dict[str, Any]:
  if not path.exists():
    raise FileNotFoundError(f"Missing YAML file: {path}")
  with path.open(encoding="utf-8") as f:
    data = yaml.safe_load(f)
  if not isinstance(data, dict):
    raise ValueError(f"Expected YAML mapping in {path}")
  return data


def _load_npz(path: Path) -> dict[str, np.ndarray]:
  if not path.exists():
    raise FileNotFoundError(f"Missing NPZ file: {path}")
  with np.load(path, allow_pickle=False) as data:
    return {key: data[key] for key in data.files}


def _require_array(
  data: dict[str, np.ndarray],
  key: str,
  *,
  ndim: int | None = None,
  trailing_dim: int | None = None,
) -> np.ndarray:
  if key not in data:
    raise ValueError(f"replay_dataset.npz missing required array {key!r}.")
  array = data[key]
  if ndim is not None and array.ndim != ndim:
    raise ValueError(f"{key} has shape {array.shape}; expected {ndim} dimensions.")
  if trailing_dim is not None and (array.ndim == 0 or array.shape[-1] != trailing_dim):
    raise ValueError(
      f"{key} has shape {array.shape}; expected trailing dim {trailing_dim}."
    )
  return array


def _manifest_action_preference(
  manifest: dict[str, Any],
  dataset: dict[str, np.ndarray],
  requested: ActionSource,
) -> tuple[str, str]:
  if requested != "auto":
    if requested not in dataset:
      raise ValueError(f"Requested action source {requested!r} is not in dataset.")
    return requested, "requested_by_cli"

  action_block = manifest.get("action")
  outputs = manifest.get("outputs") or []
  output_semantics = ""
  if outputs and isinstance(outputs[0], dict):
    output_semantics = str(outputs[0].get("semantics", "")).lower()

  if (
    "action_raw" in dataset
    and isinstance(action_block, dict)
    and action_block.get("raw_action_clamp") is not None
  ):
    return (
      "action_raw",
      "manifest_declares_raw_policy_action_and_task_side_raw_action_clamp",
    )
  if "raw" in output_semantics and "action_raw" in dataset:
    return "action_raw", "manifest_output_semantics_declares_raw_action"
  if "action_clipped" in dataset:
    return "action_clipped", "fallback_to_saved_clipped_action"
  if "action_raw" in dataset:
    return "action_raw", "fallback_to_saved_raw_action"
  raise ValueError("Dataset contains neither action_raw nor action_clipped.")


def _resolve_replay_indices(
  dataset: dict[str, np.ndarray],
  max_steps: int | None,
  *,
  start_index: int,
  loop: bool,
) -> tuple[int, int, int, np.ndarray]:
  frame_index = _require_array(dataset, "frame_index", ndim=1)
  total_frames = int(frame_index.shape[0])
  if total_frames <= 0:
    raise ValueError("Replay dataset contains no frames to replay.")

  start_index = int(start_index)
  if start_index < 0:
    raise ValueError("--start-index must be a non-negative integer.")
  if start_index >= total_frames:
    raise ValueError(
      f"Requested --start-index {start_index}, but replay_dataset.npz contains "
      f"only {total_frames} frames."
    )
  available_frames = total_frames - start_index

  if max_steps is not None:
    steps = int(max_steps)
    if steps <= 0:
      raise ValueError("--max-steps must be a positive integer.")
  else:
    steps = available_frames

  if steps <= 0:
    raise ValueError("Replay dataset contains no frames to replay.")
  if steps > available_frames and not loop:
    raise ValueError(
      f"Requested --max-steps {steps}, but replay_dataset.npz contains only "
      f"{available_frames} frames from --start-index {start_index} "
      f"(total frames: {total_frames}). Export a longer artifact with a larger "
      "--num-steps value, lower --start-index, or pass --loop to explicitly "
      "repeat the selected saved trajectory segment."
    )

  replay_indices = np.arange(steps, dtype=np.int64)
  if steps > available_frames:
    replay_indices %= available_frames
  replay_indices += start_index
  return steps, total_frames, available_frames, replay_indices


def _resolve_bool(
  value: bool | None,
  manifest: dict[str, Any],
  name: str,
  fallback: bool,
) -> bool:
  if value is not None:
    return bool(value)
  rollout_flags = manifest.get("rollout_flags")
  if isinstance(rollout_flags, dict) and name in rollout_flags:
    return bool(rollout_flags[name])
  return fallback


def _resolve_fixed_goal_axis(options: ReplayOptions, manifest: dict[str, Any]) -> str:
  if options.fixed_goal_axis is not None:
    return options.fixed_goal_axis
  rollout_flags = manifest.get("rollout_flags")
  if isinstance(rollout_flags, dict) and rollout_flags.get("fixed_goal_axis"):
    return str(rollout_flags["fixed_goal_axis"])
  fixed_goal = manifest.get("fixed_goal")
  if isinstance(fixed_goal, dict) and fixed_goal.get("axis"):
    return str(fixed_goal["axis"])
  return _DEFAULT_FIXED_GOAL_AXIS


def _resolve_fixed_goal_deg(options: ReplayOptions, manifest: dict[str, Any]) -> float:
  if options.fixed_goal_deg is not None:
    return float(options.fixed_goal_deg)
  rollout_flags = manifest.get("rollout_flags")
  if (
    isinstance(rollout_flags, dict) and rollout_flags.get("fixed_goal_deg") is not None
  ):
    return float(rollout_flags["fixed_goal_deg"])
  fixed_goal = manifest.get("fixed_goal")
  if isinstance(fixed_goal, dict) and fixed_goal.get("degrees") is not None:
    return float(fixed_goal["degrees"])
  return _DEFAULT_FIXED_GOAL_DEG


def _resolve_task_id(options: ReplayOptions, manifest: dict[str, Any]) -> str:
  task_name = options.task or str(manifest.get("task_name") or "")
  if not task_name:
    raise ValueError("Task must be provided either by --task or policy_manifest.yaml.")
  return resolve_task(
    task_name,
    task_aliases=DEFAULT_TASK_ALIASES,
    all_tasks=list_tasks(),
  )


def _prepare_env_cfgs(
  task_id: str,
  options: ReplayOptions,
  manifest: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
  if options.num_envs != 1:
    raise ValueError("This artifact replay script supports only --num-envs 1.")

  env_cfg, agent_cfg = prepare_task_cfgs(task_id, [], play=True)

  # The artifact was produced through the zero-wrist-pitch play path. Apply the
  # same play-time config edit here, without changing the task implementation.
  _force_zero_robot_pitch_reset(env_cfg)

  fixed_object_init = _resolve_bool(
    options.fixed_object_init,
    manifest,
    "fixed_object_init",
    True,
  )
  single_goal = _resolve_bool(options.single_goal, manifest, "single_goal", True)
  fixed_goal = _resolve_bool(options.fixed_goal, manifest, "fixed_goal", True)
  no_terminations = _resolve_bool(
    options.no_terminations,
    manifest,
    "no_terminations",
    True,
  )
  fixed_goal_axis = _resolve_fixed_goal_axis(options, manifest)
  fixed_goal_deg = _resolve_fixed_goal_deg(options, manifest)

  if fixed_object_init:
    fixed_object_offset = _resolve_fixed_object_offset(
      task_id,
      env_cfg,
      options.cube_offset_in_palm,
    )
    _force_fixed_object_init(env_cfg, cube_offset_in_palm=fixed_object_offset)
  if single_goal:
    _force_single_reorient_goal(env_cfg)
  if fixed_goal:
    _force_fixed_reorient_goal(
      env_cfg,
      axis=fixed_goal_axis,
      deg=fixed_goal_deg,
    )
  if no_terminations:
    env_cfg.terminations = {}

  env_cfg.scene.num_envs = 1
  if options.video_width is not None:
    env_cfg.viewer.width = options.video_width
  if options.video_height is not None:
    env_cfg.viewer.height = options.video_height

  resolved = {
    "task": task_id,
    "num_envs": 1,
    "fixed_object_init": fixed_object_init,
    "single_goal": single_goal,
    "fixed_goal": fixed_goal,
    "fixed_goal_axis": fixed_goal_axis,
    "fixed_goal_deg": fixed_goal_deg,
    "cube_offset_in_palm": (
      list(options.cube_offset_in_palm)
      if options.cube_offset_in_palm is not None
      else None
    ),
    "no_terminations": no_terminations,
    "zero_robot_pitch_reset": True,
  }
  return env_cfg, agent_cfg, resolved


def _as_np(tensor: torch.Tensor) -> np.ndarray:
  return tensor.detach().cpu().numpy()


def _policy_obs_np(obs: Any) -> np.ndarray:
  policy_obs = obs["policy"]
  return _as_np(policy_obs).astype(np.float32, copy=False)


def _collect_state(env: RslRlVecEnvWrapper) -> dict[str, np.ndarray]:
  unwrapped = env.unwrapped
  robot = unwrapped.scene["robot"]
  obj = unwrapped.scene["object"]
  action_term = unwrapped.action_manager.get_term("joint_pos")
  target_ids = action_term.target_ids
  command = unwrapped.command_manager.get_term("reorient_command")

  object_state = torch.cat(
    [obj.data.root_link_pos_w, obj.data.root_link_quat_w],
    dim=-1,
  )
  return {
    "joint_pos_policy_order": _as_np(robot.data.joint_pos[:, target_ids]).astype(
      np.float32
    ),
    "object_state": _as_np(object_state).astype(np.float32),
    "goal_state": _as_np(command.goal_quat).astype(np.float32),
    "processed_action": _as_np(action_term.processed_action).astype(np.float32),
  }


def _validate_policy_joint_order(
  env: RslRlVecEnvWrapper,
  manifest: dict[str, Any],
  dataset: dict[str, np.ndarray],
) -> list[str]:
  action_term = env.unwrapped.action_manager.get_term("joint_pos")
  target_names = list(action_term.target_names)
  manifest_order = manifest.get("policy_joint_order")
  if manifest_order is not None and list(manifest_order) != target_names:
    raise ValueError(
      "Environment action target order does not match manifest policy_joint_order."
    )
  if "joint_names_policy_order" in dataset:
    saved_order = [str(v) for v in dataset["joint_names_policy_order"].tolist()]
    if saved_order != target_names:
      raise ValueError(
        "Environment action target order does not match replay_dataset joint names."
      )
  return target_names


def _safe_row(data: dict[str, np.ndarray], key: str, index: int) -> np.ndarray | None:
  array = data.get(key)
  if array is None:
    return None
  if array.ndim == 0 or index >= array.shape[0]:
    return None
  return array[index]


def _append_optional(
  out: dict[str, list[np.ndarray]],
  key: str,
  value: np.ndarray | None,
) -> None:
  if value is None:
    return
  out.setdefault(key, []).append(np.asarray(value).copy())


def _row_max_abs(lhs: np.ndarray | None, rhs: np.ndarray | None) -> float:
  if lhs is None or rhs is None:
    return float("nan")
  if lhs.shape != rhs.shape:
    return float("nan")
  return float(np.max(np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))))


def _stack_optional(rows: dict[str, list[np.ndarray]], payload: dict[str, Any]) -> None:
  for key, values in rows.items():
    if values:
      payload[key] = np.asarray(values)


def _compute_onnx_actions(
  artifact_dir: Path,
  dataset: dict[str, np.ndarray],
  replay_indices: np.ndarray,
) -> tuple[np.ndarray, float]:
  import onnxruntime as ort

  raw = _require_array(
    dataset,
    "policy_input_raw",
    ndim=2,
    trailing_dim=INPUT_DIM,
  )[replay_indices].astype(np.float32)
  saved_action_raw = _require_array(
    dataset,
    "action_raw",
    ndim=2,
    trailing_dim=ACTION_DIM,
  )[replay_indices].astype(np.float32)

  session = ort.InferenceSession(
    str(artifact_dir / "policy.onnx"),
    providers=["CPUExecutionProvider"],
  )
  recomputed: list[np.ndarray] = []
  for row in raw:
    output = session.run([OUTPUT_NAME], {INPUT_NAME: row.reshape(1, -1)})[0]
    recomputed.append(output.reshape(-1).astype(np.float32))
  actions = np.asarray(recomputed, dtype=np.float32)
  max_abs = float(np.max(np.abs(actions - saved_action_raw)))
  return actions, max_abs


def _make_env(
  env_cfg: Any,
  agent_cfg: Any,
  *,
  device: str,
  render_mode: str | None,
  video: bool,
  video_dir: Path,
  video_length: int,
) -> RslRlVecEnvWrapper:
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  if video:
    env = VideoRecorder(
      env,
      video_folder=video_dir,
      step_trigger=lambda step: step == 0,
      video_length=video_length,
      name_prefix="sim-replay",
      disable_logger=False,
    )
  return RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)


def _reproduction_status(dataset: dict[str, np.ndarray]) -> tuple[str, bool, list[str]]:
  full_state_keys = {
    "sim_qpos",
    "sim_qvel",
    "sim_ctrl",
    "sim_act",
    "sim_qacc_warmstart",
    "rng_state",
  }
  if full_state_keys.issubset(dataset.keys()):
    return "full_state_fields_present_but_not_restored_by_this_script", False, []
  missing = [
    "initial MuJoCo qpos/qvel/act/ctrl and warm-start state",
    "robot/object velocities and any applied forces",
    "action-manager previous target/effective-action state",
    "observation-history buffers",
    "command-manager timers and sampled goal state",
    "randomization-expanded model parameters",
    "torch/numpy/python RNG states",
  ]
  return "not_guaranteed", False, missing


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
  with path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(summary, f, sort_keys=False)


def _mtime_ns(path: Path) -> int | None:
  try:
    return path.stat().st_mtime_ns
  except FileNotFoundError:
    return None


def _generated_video_path(
  expected_path: Path,
  before_mtime_ns: int | None,
) -> Path | None:
  after_mtime_ns = _mtime_ns(expected_path)
  if after_mtime_ns is None:
    return None
  if before_mtime_ns is None or after_mtime_ns != before_mtime_ns:
    return expected_path
  return None


def _require_positive_optional_int(name: str, value: int | None) -> None:
  if value is not None and int(value) <= 0:
    raise ValueError(f"{name} must be a positive integer.")


def _str_to_bool(value: bool | str) -> bool:
  if isinstance(value, bool):
    return value
  normalized = value.strip().lower()
  if normalized in {"1", "true", "t", "yes", "y", "on"}:
    return True
  if normalized in {"0", "false", "f", "no", "n", "off"}:
    return False
  raise argparse.ArgumentTypeError(
    "expected a boolean value such as true/false, yes/no, or 1/0"
  )


def _run_compare_only(
  options: ReplayOptions,
  manifest: dict[str, Any],
  dataset: dict[str, np.ndarray],
  *,
  steps: int,
  total_frames: int,
  replay_indices: np.ndarray,
  output_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
  recomputed_actions, recompute_max_abs = _compute_onnx_actions(
    options.artifact_dir,
    dataset,
    replay_indices,
  )
  saved_action_raw = dataset["action_raw"][replay_indices].astype(np.float32)
  log_path = output_dir / "sim_replay_log.npz"
  summary_path = output_dir / "sim_replay_summary.yaml"
  np.savez(
    log_path,
    replay_step=np.arange(steps, dtype=np.int64),
    frame_index=dataset["frame_index"][replay_indices],
    saved_action=saved_action_raw,
    replayed_action=recomputed_actions,
    policy_recomputed_action=recomputed_actions,
    action_abs_error=np.abs(recomputed_actions - saved_action_raw),
  )
  status, exact, missing = _reproduction_status(dataset)
  summary = {
    "artifact_path": str(options.artifact_dir),
    "artifact_dir": str(options.artifact_dir),
    "mode": options.mode,
    "compare_only": True,
    "steps": steps,
    "total_available_frames": total_frames,
    "start_index": options.start_index,
    "available_frames_after_start": total_frames - options.start_index,
    "pre_roll_steps": 0,
    "replayed_frames": steps,
    "requested_max_steps": options.max_steps,
    "loop_enabled": options.loop,
    "env_step_dt": None,
    "replay_duration_sec": None,
    "model_input": "policy_input_raw",
    "policy_input_normalized_used": False,
    "max_abs_diff_recomputed_vs_action_raw": recompute_max_abs,
    "max_abs_diff_saved_vs_replayed_action": recompute_max_abs,
    "video_enabled": False,
    "output_video": None,
    "output_log": str(log_path),
    "output_summary": str(summary_path),
    "state_reproduction": {
      "status": status,
      "exact_reproduction_guaranteed": exact,
      "missing_fields_for_exact_reproduction": missing,
    },
    "manifest_task": manifest.get("task_name"),
  }
  _write_summary(summary_path, summary)
  return log_path, summary_path, summary


def run_replay(options: ReplayOptions) -> tuple[Path, Path, dict[str, Any]]:
  _require_positive_optional_int("--video-length", options.video_length)
  _require_positive_optional_int("--video-width", options.video_width)
  _require_positive_optional_int("--video-height", options.video_height)

  artifact_dir = options.artifact_dir
  manifest = _load_yaml(artifact_dir / "policy_manifest.yaml")
  dataset = _load_npz(artifact_dir / "replay_dataset.npz")
  steps, total_frames, available_frames, replay_indices = _resolve_replay_indices(
    dataset,
    options.max_steps,
    start_index=options.start_index,
    loop=options.loop,
  )
  print(f"[INFO] replay dataset frames available: {total_frames}")
  print(f"[INFO] replay start index: {options.start_index}")
  print(f"[INFO] replay frames available from start: {available_frames}")
  print(f"[INFO] replaying frames: {steps}")
  if options.loop and steps > available_frames:
    print(
      "[INFO] --loop enabled; selected saved replay rows will repeat explicitly."
    )
  output_dir = options.output_dir or artifact_dir
  output_dir.mkdir(parents=True, exist_ok=True)

  action_source, action_source_reason = _manifest_action_preference(
    manifest,
    dataset,
    options.action_source,
  )
  saved_actions = _require_array(
    dataset,
    action_source,
    ndim=2,
    trailing_dim=ACTION_DIM,
  )[replay_indices].astype(np.float32)
  pre_roll_indices = np.arange(options.start_index, dtype=np.int64)
  pre_roll_actions = _require_array(
    dataset,
    action_source,
    ndim=2,
    trailing_dim=ACTION_DIM,
  )[pre_roll_indices].astype(np.float32)
  action_raw = _require_array(
    dataset,
    "action_raw",
    ndim=2,
    trailing_dim=ACTION_DIM,
  )[replay_indices].astype(np.float32)

  recomputed_actions: np.ndarray | None = None
  recompute_max_abs: float | None = None
  if options.mode == "policy_recompute":
    recomputed_actions, recompute_max_abs = _compute_onnx_actions(
      artifact_dir,
      dataset,
      replay_indices,
    )
    if (
      pre_roll_indices.size > 0
      and options.policy_step_source == "recomputed"
      and not options.compare_only
    ):
      pre_roll_actions, _pre_roll_recompute_max_abs = _compute_onnx_actions(
        artifact_dir,
        dataset,
        pre_roll_indices,
      )
    if options.compare_only:
      return _run_compare_only(
        options,
        manifest,
        dataset,
        steps=steps,
        total_frames=total_frames,
        replay_indices=replay_indices,
        output_dir=output_dir,
      )

  task_id = _resolve_task_id(options, manifest)
  env_cfg, agent_cfg, resolved_env = _prepare_env_cfgs(task_id, options, manifest)
  configure_torch_backends()
  device = options.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  video_dir = options.video_dir or output_dir / "videos"
  video_length = options.video_length or steps
  if options.video and video_length > steps:
    print(
      f"[WARN] --video-length {video_length} exceeds replayed frames {steps}; "
      "the video will contain at most the replayed frames. Export a longer "
      "artifact or use --max-steps with --loop for an explicitly looped video."
    )
  render_mode = "rgb_array" if options.video else None
  expected_video_path = video_dir / "sim-replay-step-0.mp4"
  expected_video_mtime = _mtime_ns(expected_video_path) if options.video else None
  output_video_path: Path | None = None
  try:
    env = _make_env(
      env_cfg,
      agent_cfg,
      device=device,
      render_mode=render_mode,
      video=options.video,
      video_dir=video_dir,
      video_length=video_length,
    )
  except Exception as exc:
    if options.video:
      raise RuntimeError(
        "Video recording was requested, but the rgb_array renderer could not "
        "be initialized. Check the OpenGL/EGL/OSMesa/display setup, or rerun "
        "without --video for numerical replay."
      ) from exc
    raise

  rows: dict[str, list[np.ndarray]] = {}
  scalar_rows: dict[str, list[float | int | bool]] = {
    "frame_index": [],
    "timestamp_sec": [],
    "joint_pos_abs_error_max": [],
    "object_state_abs_error_max": [],
    "goal_state_abs_error_max": [],
    "qpos_error_abs_error_max": [],
    "target_post_clip_abs_error_max": [],
    "done": [],
  }
  replay_step_rows: list[int] = []
  replay_time_rows: list[float] = []
  env_step_dt = float(env.unwrapped.step_dt)

  try:
    obs, _extras = env.reset()
    _validate_policy_joint_order(env, manifest, dataset)

    if pre_roll_actions.shape[0] > 0:
      print(
        f"[INFO] pre-rolling {pre_roll_actions.shape[0]} saved frames before "
        "visible replay."
      )
      for action_row in pre_roll_actions:
        action_tensor = torch.as_tensor(
          action_row.reshape(1, -1),
          dtype=torch.float32,
          device=env.unwrapped.device,
        )
        obs, _rew, _dones, _extras = env.step(action_tensor)

    with _FiniteViewer(
      options.viewer,
      env,
      frame_sleep=options.viewer_frame_sleep,
    ) as viewer:
      for local_i in range(steps):
        dataset_i = int(replay_indices[local_i])
        frame_index = int(dataset["frame_index"][dataset_i])
        current_obs = _policy_obs_np(obs)[0].copy()
        current_state = _collect_state(env)

        if (
          options.mode == "policy_recompute"
          and options.policy_step_source == "recomputed"
        ):
          assert recomputed_actions is not None
          replay_action = recomputed_actions[local_i]
        else:
          replay_action = saved_actions[local_i]

        saved_joint_pos = _safe_row(dataset, "joint_pos_policy_order", dataset_i)
        saved_object_state = _safe_row(dataset, "object_state", dataset_i)
        saved_goal_state = _safe_row(dataset, "goal_state", dataset_i)
        saved_qpos_error = _safe_row(dataset, "policy_input_raw", dataset_i)
        if saved_qpos_error is not None:
          saved_qpos_error = saved_qpos_error[QPOS_ERROR_SLICE]
        current_qpos_error = current_obs[QPOS_ERROR_SLICE]

        replay_step_rows.append(local_i)
        replay_time_rows.append(float(local_i * env_step_dt))
        scalar_rows["frame_index"].append(frame_index)
        scalar_rows["timestamp_sec"].append(float(frame_index * env_step_dt))
        scalar_rows["joint_pos_abs_error_max"].append(
          _row_max_abs(
            current_state["joint_pos_policy_order"][0],
            saved_joint_pos,
          )
        )
        scalar_rows["object_state_abs_error_max"].append(
          _row_max_abs(current_state["object_state"][0], saved_object_state)
        )
        scalar_rows["goal_state_abs_error_max"].append(
          _row_max_abs(current_state["goal_state"][0], saved_goal_state)
        )
        scalar_rows["qpos_error_abs_error_max"].append(
          _row_max_abs(current_qpos_error, saved_qpos_error)
        )

        _append_optional(rows, "policy_input_raw_current", current_obs)
        _append_optional(
          rows,
          "policy_input_raw_saved",
          _safe_row(dataset, "policy_input_raw", dataset_i),
        )
        _append_optional(rows, "saved_action", saved_actions[local_i])
        _append_optional(rows, "replayed_action", replay_action)
        _append_optional(rows, "action_raw", action_raw[local_i])
        if recomputed_actions is not None:
          _append_optional(
            rows, "policy_recomputed_action", recomputed_actions[local_i]
          )
        _append_optional(
          rows,
          "current_joint_pos_policy_order",
          current_state["joint_pos_policy_order"][0],
        )
        _append_optional(rows, "saved_joint_pos_policy_order", saved_joint_pos)
        _append_optional(rows, "current_object_state", current_state["object_state"][0])
        _append_optional(rows, "saved_object_state", saved_object_state)
        _append_optional(rows, "current_goal_state", current_state["goal_state"][0])
        _append_optional(rows, "saved_goal_state", saved_goal_state)
        _append_optional(rows, "current_qpos_error", current_qpos_error)
        _append_optional(rows, "saved_qpos_error", saved_qpos_error)

        action_tensor = torch.as_tensor(
          replay_action.reshape(1, -1),
          dtype=torch.float32,
          device=env.unwrapped.device,
        )
        obs, _rew, dones, _extras = env.step(action_tensor)
        done = bool(_as_np(dones).astype(bool)[0])
        scalar_rows["done"].append(done)

        post_state = _collect_state(env)
        saved_target = _safe_row(
          dataset,
          "target_post_clip_policy_order",
          dataset_i,
        )
        _append_optional(
          rows,
          "target_post_clip_policy_order",
          post_state["processed_action"][0],
        )
        _append_optional(rows, "saved_target_post_clip_policy_order", saved_target)
        _append_optional(
          rows,
          "post_step_joint_pos_policy_order",
          post_state["joint_pos_policy_order"][0],
        )
        _append_optional(rows, "post_step_object_state", post_state["object_state"][0])
        _append_optional(rows, "post_step_goal_state", post_state["goal_state"][0])
        scalar_rows["target_post_clip_abs_error_max"].append(
          _row_max_abs(post_state["processed_action"][0], saved_target)
        )
        viewer.sync()

  finally:
    env.close()
    if options.video:
      output_video_path = _generated_video_path(
        expected_video_path,
        expected_video_mtime,
      )

  log_path = output_dir / "sim_replay_log.npz"
  summary_path = output_dir / "sim_replay_summary.yaml"
  payload: dict[str, Any] = {
    "replay_step": np.asarray(replay_step_rows, dtype=np.int64),
    "replay_time_sec": np.asarray(replay_time_rows, dtype=np.float64),
    "frame_index": np.asarray(scalar_rows["frame_index"], dtype=np.int64),
    "timestamp_sec": np.asarray(scalar_rows["timestamp_sec"], dtype=np.float64),
    "done": np.asarray(scalar_rows["done"], dtype=bool),
    "action_source": np.asarray(action_source),
    "joint_names_policy_order": np.asarray(
      manifest.get("policy_joint_order", []), dtype=str
    ),
    "joint_pos_abs_error_max": np.asarray(
      scalar_rows["joint_pos_abs_error_max"],
      dtype=np.float64,
    ),
    "object_state_abs_error_max": np.asarray(
      scalar_rows["object_state_abs_error_max"],
      dtype=np.float64,
    ),
    "goal_state_abs_error_max": np.asarray(
      scalar_rows["goal_state_abs_error_max"],
      dtype=np.float64,
    ),
    "qpos_error_abs_error_max": np.asarray(
      scalar_rows["qpos_error_abs_error_max"],
      dtype=np.float64,
    ),
    "target_post_clip_abs_error_max": np.asarray(
      scalar_rows["target_post_clip_abs_error_max"],
      dtype=np.float64,
    ),
  }
  _stack_optional(rows, payload)
  if "saved_action" in payload and "replayed_action" in payload:
    payload["action_abs_error"] = np.abs(
      payload["saved_action"].astype(np.float64)
      - payload["replayed_action"].astype(np.float64)
    )
  if "policy_recomputed_action" in payload:
    payload["recompute_action_abs_error"] = np.abs(
      payload["policy_recomputed_action"].astype(np.float64)
      - payload["action_raw"].astype(np.float64)
    )
  np.savez(log_path, **payload)

  max_saved_vs_replayed = float(np.max(payload["action_abs_error"]))
  status, exact, missing = _reproduction_status(dataset)
  summary = {
    "artifact_path": str(artifact_dir),
    "artifact_dir": str(artifact_dir),
    "mode": options.mode,
    "task": task_id,
    "steps": steps,
    "total_available_frames": total_frames,
    "start_index": options.start_index,
    "available_frames_after_start": available_frames,
    "pre_roll_steps": int(options.start_index),
    "replayed_frames": steps,
    "requested_max_steps": options.max_steps,
    "loop_enabled": options.loop,
    "env_step_dt": env_step_dt,
    "replay_duration_sec": float(steps * env_step_dt),
    "device": device,
    "viewer": _resolve_viewer(options.viewer),
    "video_enabled": options.video,
    "video_dir": str(video_dir) if options.video else None,
    "video_length": video_length if options.video else None,
    "video_width": options.video_width if options.video else None,
    "video_height": options.video_height if options.video else None,
    "output_video": str(output_video_path) if output_video_path is not None else None,
    "action_source": action_source,
    "action_source_reason": action_source_reason,
    "policy_input_normalized_used": False,
    "policy_step_source": (
      options.policy_step_source if options.mode == "policy_recompute" else None
    ),
    "max_abs_diff_saved_vs_replayed_action": max_saved_vs_replayed,
    "max_abs_diff_recomputed_vs_action_raw": recompute_max_abs,
    "max_joint_pos_abs_error": float(np.nanmax(payload["joint_pos_abs_error_max"])),
    "max_object_state_abs_error": float(
      np.nanmax(payload["object_state_abs_error_max"])
    ),
    "max_goal_state_abs_error": float(np.nanmax(payload["goal_state_abs_error_max"])),
    "max_qpos_error_abs_error": float(np.nanmax(payload["qpos_error_abs_error_max"])),
    "max_target_post_clip_abs_error": float(
      np.nanmax(payload["target_post_clip_abs_error_max"])
    ),
    "environment": resolved_env,
    "output_log": str(log_path),
    "output_summary": str(summary_path),
    "state_reproduction": {
      "status": status,
      "exact_reproduction_guaranteed": exact,
      "missing_fields_for_exact_reproduction": missing,
    },
  }
  _write_summary(summary_path, summary)
  return log_path, summary_path, summary


def _parse_args() -> ReplayOptions:
  parser = argparse.ArgumentParser(
    description="Replay a Revo3 216-D play artifact inside simulation."
  )
  parser.add_argument("--artifact-dir", required=True, type=Path)
  parser.add_argument("--task", default=None)
  parser.add_argument(
    "--mode",
    choices=("action_open_loop", "policy_recompute"),
    default="action_open_loop",
  )
  parser.add_argument(
    "--viewer",
    choices=("none", "auto", "native", "viser"),
    default="none",
  )
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--device", default=None)
  parser.add_argument("--max-steps", type=int, default=None)
  parser.add_argument(
    "--start-index",
    type=int,
    default=0,
    help=(
      "Start replay from this row of replay_dataset.npz. Useful for skipping "
      "reset-time contact transients while preserving saved action semantics. "
      "Simulator replay pre-rolls the skipped rows before visible/logged replay."
    ),
  )
  parser.add_argument(
    "--loop",
    action="store_true",
    help=(
      "Explicitly repeat replay_dataset.npz rows if --max-steps exceeds the "
      "available frame count."
    ),
  )
  parser.add_argument("--output-dir", type=Path, default=None)
  parser.add_argument(
    "--action-source",
    choices=("auto", "action_raw", "action_clipped"),
    default="auto",
  )
  parser.add_argument(
    "--policy-step-source",
    choices=("recomputed", "saved"),
    default="recomputed",
    help="Only used in --mode policy_recompute.",
  )
  parser.add_argument(
    "--compare-only",
    action="store_true",
    help="Only run ONNX comparison in --mode policy_recompute; skip simulator.",
  )

  parser.add_argument(
    "--fixed-object-init",
    action=argparse.BooleanOptionalAction,
    default=None,
  )
  parser.add_argument(
    "--single-goal",
    action=argparse.BooleanOptionalAction,
    default=None,
  )
  parser.add_argument(
    "--fixed-goal",
    action=argparse.BooleanOptionalAction,
    default=None,
  )
  parser.add_argument(
    "--fixed-goal-axis",
    choices=("x", "y", "z"),
    default=None,
  )
  parser.add_argument("--fixed-goal-deg", type=float, default=None)
  parser.add_argument(
    "--cube-offset-in-palm",
    nargs=3,
    type=float,
    metavar=("X", "Y", "Z"),
    default=None,
  )
  parser.add_argument(
    "--no-terminations", dest="no_terminations", action="store_true", default=None
  )
  parser.add_argument(
    "--allow-terminations", dest="no_terminations", action="store_false"
  )

  parser.add_argument(
    "--video",
    nargs="?",
    const=True,
    default=False,
    type=_str_to_bool,
    help="Enable optional video recording; accepts optional true/false.",
  )
  parser.add_argument("--no-video", dest="video", action="store_false")
  parser.add_argument("--video-dir", type=Path, default=None)
  parser.add_argument("--video-length", type=int, default=None)
  parser.add_argument("--video-width", type=int, default=None)
  parser.add_argument("--video-height", type=int, default=None)
  parser.add_argument(
    "--viewer-frame-sleep",
    type=float,
    default=0.0,
    help="Seconds to sleep after each finite viewer sync.",
  )

  args = parser.parse_args()
  return ReplayOptions(
    artifact_dir=args.artifact_dir,
    task=args.task,
    mode=args.mode,
    viewer=args.viewer,
    num_envs=args.num_envs,
    device=args.device,
    max_steps=args.max_steps,
    start_index=args.start_index,
    loop=args.loop,
    output_dir=args.output_dir,
    action_source=args.action_source,
    policy_step_source=args.policy_step_source,
    compare_only=args.compare_only,
    fixed_object_init=args.fixed_object_init,
    single_goal=args.single_goal,
    fixed_goal=args.fixed_goal,
    fixed_goal_axis=args.fixed_goal_axis,
    fixed_goal_deg=args.fixed_goal_deg,
    cube_offset_in_palm=_as_offset_tuple(args.cube_offset_in_palm),
    no_terminations=args.no_terminations,
    video=args.video,
    video_dir=args.video_dir,
    video_length=args.video_length,
    video_width=args.video_width,
    video_height=args.video_height,
    viewer_frame_sleep=args.viewer_frame_sleep,
  )


def main() -> None:
  options = _parse_args()
  log_path, summary_path, summary = run_replay(options)
  print(f"[INFO] artifact: {summary.get('artifact_path', options.artifact_dir)}")
  print(f"[INFO] total available frames: {summary['total_available_frames']}")
  print(f"[INFO] replay start index: {summary['start_index']}")
  print(
    "[INFO] available frames after start: "
    f"{summary['available_frames_after_start']}"
  )
  if "pre_roll_steps" in summary:
    print(f"[INFO] pre-roll steps before visible replay: {summary['pre_roll_steps']}")
  print(f"[INFO] replayed frames: {summary['replayed_frames']}")
  env_step_dt = summary.get("env_step_dt")
  replay_duration_sec = summary.get("replay_duration_sec")
  if env_step_dt is not None and replay_duration_sec is not None:
    print(f"[INFO] env step dt: {env_step_dt:.9g} sec")
    print(f"[INFO] replay duration: {replay_duration_sec:.9g} sec")
  if summary.get("output_video"):
    print(f"[INFO] sim replay video: {summary['output_video']}")
  elif summary.get("video_enabled"):
    print("[WARN] video was requested, but no output video file was detected.")
  print(f"[INFO] sim replay log: {log_path}")
  print(f"[INFO] sim replay summary: {summary_path}")
  print(
    "[INFO] max |saved_action - replayed_action| = "
    f"{summary['max_abs_diff_saved_vs_replayed_action']:.9g}"
  )
  recompute = summary.get("max_abs_diff_recomputed_vs_action_raw")
  if recompute is not None:
    print(f"[INFO] max |policy.onnx(raw_216) - action_raw| = {recompute:.9g}")
  state = summary["state_reproduction"]
  print(
    "[INFO] simulation state reproduction: "
    f"{state['status']} "
    f"(exact guaranteed: {state['exact_reproduction_guaranteed']})"
  )


if __name__ == "__main__":
  main()
