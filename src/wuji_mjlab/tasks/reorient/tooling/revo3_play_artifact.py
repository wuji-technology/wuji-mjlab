# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Export and validate the actual Revo3 play-policy artifact.

This module is intentionally training-side only. It exports the real RSL-RL
single-input interface used by the Revo3 right-hand checkpoint:

  raw obs[216] -> EmpiricalNormalization -> actor MLP -> actions[21]

The saved normalized observation arrays are audit data only. The ONNX model
expects raw observations because the normalizer is part of the graph.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The sandbox/home profile used by CI and Codex can make ~/.cache and
# ~/.config read-only. Route import-time caches for Warp/Matplotlib to /tmp
# before importing mjlab, which imports both stacks indirectly.
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import yaml
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from onnx import TensorProto
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.modules.distribution import _MeanSliceDeterministicOutput

from scripts.play.play_rsl_rl_zero_wrist_pitch import (
  _DEFAULT_FIXED_GOAL_AXIS,
  _DEFAULT_FIXED_GOAL_DEG,
  _as_offset_tuple,
  _axis_angle_quat_wxyz,
  _force_fixed_object_init,
  _force_fixed_reorient_goal,
  _force_single_reorient_goal,
  _force_zero_robot_pitch_reset,
  _resolve_fixed_object_offset,
)
from wuji_mjlab.tasks.reorient.robot_bindings import REVO3_RIGHT_HAND_BINDING
from wuji_mjlab.tasks.reorient.tooling import onnx_export_core
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

SCHEMA_VERSION = "revo3_play_single_input_216_v1"
MODEL_INPUT_KIND = "raw_obs_with_normalizer_inside_model"
INPUT_NAME = "obs"
OUTPUT_NAME = "actions"
INPUT_DIM = 216
ACTION_DIM = 21
NORMALIZER_EPSILON = 1.0e-2
OLD_README_SCHEMA = "obs[126] + proprio_hist[30,42]"
ACTUAL_SCHEMA = "single_input[216]"

INPUT_SLICES: tuple[dict[str, Any], ...] = (
  {
    "name": "noisy_joint_angles",
    "start": 0,
    "end": 63,
    "shape": [3, 21],
    "semantics": (
      "Joint positions normalized by soft joint limits; play mode disables "
      "policy observation corruption, so this is clean despite the term name."
    ),
  },
  {
    "name": "qpos_error",
    "start": 63,
    "end": 126,
    "shape": [3, 21],
    "semantics": "normalized_joint_pos - normalized_processed_target",
  },
  {
    "name": "cube_pos_in_tag",
    "start": 126,
    "end": 135,
    "shape": [3, 3],
    "semantics": "Cube root position in wrist/tag frame.",
  },
  {
    "name": "cube_ori_error",
    "start": 135,
    "end": 153,
    "shape": [3, 6],
    "semantics": "6D rotation error between cube and goal in tag-frame convention.",
  },
  {
    "name": "action_history",
    "start": 153,
    "end": 216,
    "shape": [3, 21],
    "semantics": "Previous raw policy actions before action-manager processing.",
  },
)


@dataclass(frozen=True)
class ExportOptions:
  checkpoint: Path
  output_dir: Path
  task: str = "Revo3RightHand_Reorient_ReposeReward_FineTune"
  num_steps: int = 1000
  device: str | None = None
  fixed_object_init: bool = True
  single_goal: bool = True
  fixed_goal: bool = True
  fixed_goal_axis: str = _DEFAULT_FIXED_GOAL_AXIS
  fixed_goal_deg: float = _DEFAULT_FIXED_GOAL_DEG
  cube_offset_in_palm: tuple[float, float, float] | None = None
  no_terminations: bool = True


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _git_commit() -> str | None:
  try:
    result = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      check=True,
      capture_output=True,
      text=True,
    )
  except (OSError, subprocess.CalledProcessError):
    return None
  commit = result.stdout.strip()
  return commit or None


def _dtype_name(elem_type: int) -> str:
  if elem_type == TensorProto.FLOAT:
    return "float32"
  return TensorProto.DataType.Name(elem_type).lower()


def _value_info_shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
  dims: list[int | str] = []
  for dim in value_info.type.tensor_type.shape.dim:
    dims.append(dim.dim_param if dim.dim_param else int(dim.dim_value))
  return dims


def _load_actor_state(
  checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
  run_dir = checkpoint_path.parent
  agent_cfg = onnx_export_core._load_yaml(str(run_dir / "params" / "agent.yaml"))
  env_cfg = onnx_export_core._load_yaml(str(run_dir / "params" / "env.yaml"))
  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
  state_dict = onnx_export_core._resolve_export_state_dict(checkpoint)
  return state_dict, agent_cfg, env_cfg


def _build_exporter(
  state_dict: dict[str, Any],
  agent_cfg: dict[str, Any] | None,
) -> tuple[nn.Module, dict[str, Any], dict[str, torch.Tensor]]:
  arch = onnx_export_core._build_architecture(agent_cfg, state_dict)
  if int(arch["num_actor_obs"]) != INPUT_DIM:
    raise ValueError(
      f"Expected actual Revo3 actor obs dim {INPUT_DIM}, got {arch['num_actor_obs']}."
    )
  if int(arch["num_actions"]) != ACTION_DIM:
    raise ValueError(
      f"Expected actual Revo3 action dim {ACTION_DIM}, got {arch['num_actions']}."
    )
  onnx_export_core._check_unsupported(state_dict, arch, agent_cfg)

  norm_state = onnx_export_core._extract_normalizer_state_dict(state_dict)
  if norm_state:
    normalizer: nn.Module = EmpiricalNormalization(INPUT_DIM)
    normalizer.load_state_dict(norm_state)
    normalizer.eval()
  else:
    normalizer = nn.Identity()

  actor = onnx_export_core.MLP(
    input_dim=INPUT_DIM,
    output_dim=arch["actor_output_dim"],
    hidden_dims=arch["hidden_dims"],
    activation=arch["activation"],
  )
  actor.load_state_dict(onnx_export_core._extract_actor_mlp_state_dict(state_dict))
  actor.eval()

  if arch["state_dependent_std"]:
    actor = nn.Sequential(actor, _MeanSliceDeterministicOutput())

  exporter = onnx_export_core._StandaloneExporter(normalizer, actor)
  exporter.eval()
  return exporter, arch, norm_state


def _export_onnx(exporter: nn.Module, onnx_path: Path) -> dict[str, Any]:
  onnx_path.parent.mkdir(parents=True, exist_ok=True)
  dummy_input = torch.zeros(1, INPUT_DIM, dtype=torch.float32)
  torch.onnx.export(
    exporter,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=onnx_export_core._ONNX_OPSET_VERSION,
    input_names=[INPUT_NAME],
    output_names=[OUTPUT_NAME],
    dynamic_axes={},
    dynamo=False,
  )

  data_path = Path(str(onnx_path) + ".data")
  if data_path.exists():
    model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save(model, str(onnx_path), save_as_external_data=False)
    data_path.unlink()

  model = onnx.load(str(onnx_path))
  input_info = model.graph.input[0]
  output_info = model.graph.output[0]
  return {
    "input_name": input_info.name,
    "input_shape": _value_info_shape(input_info),
    "input_dtype": _dtype_name(input_info.type.tensor_type.elem_type),
    "output_name": output_info.name,
    "output_shape": _value_info_shape(output_info),
    "output_dtype": _dtype_name(output_info.type.tensor_type.elem_type),
  }


def _normalizer_arrays(
  norm_state: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
  if norm_state:
    mean = norm_state["_mean"].detach().cpu().numpy().reshape(-1).astype(np.float32)
    std = norm_state["_std"].detach().cpu().numpy().reshape(-1).astype(np.float32)
    var_t = norm_state.get("_var")
    count_t = norm_state.get("count")
    var = (
      var_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
      if var_t is not None
      else None
    )
    count = count_t.detach().cpu().numpy() if count_t is not None else None
  else:
    mean = np.zeros(INPUT_DIM, dtype=np.float32)
    std = np.ones(INPUT_DIM, dtype=np.float32)
    var = np.ones(INPUT_DIM, dtype=np.float32)
    count = np.array(0, dtype=np.int64)

  if mean.shape != (INPUT_DIM,) or std.shape != (INPUT_DIM,):
    raise ValueError(
      f"Normalizer shape mismatch: mean={mean.shape}, std={std.shape}, "
      f"expected ({INPUT_DIM},)."
    )
  return mean, std, var, count


def _write_normalizer_npz(path: Path, norm_state: dict[str, torch.Tensor]) -> None:
  mean, std, var, count = _normalizer_arrays(norm_state)
  payload: dict[str, np.ndarray] = {
    "input_mean": mean,
    "input_std": std,
    "epsilon": np.array(NORMALIZER_EPSILON, dtype=np.float32),
  }
  if var is not None:
    payload["input_var"] = var
  if count is not None:
    payload["count"] = count
  np.savez(path, **payload)


def _prepare_rollout_cfg(options: ExportOptions):
  env_cfg, agent_cfg = prepare_task_cfgs(options.task, [], play=True)
  _force_zero_robot_pitch_reset(env_cfg)
  if options.fixed_object_init:
    fixed_offset = _resolve_fixed_object_offset(
      options.task,
      env_cfg,
      options.cube_offset_in_palm,
    )
    _force_fixed_object_init(env_cfg, cube_offset_in_palm=fixed_offset)
  if options.single_goal:
    _force_single_reorient_goal(env_cfg)
  if options.fixed_goal:
    _force_fixed_reorient_goal(
      env_cfg,
      axis=options.fixed_goal_axis,
      deg=options.fixed_goal_deg,
    )
  if options.no_terminations:
    env_cfg.terminations = {}
  env_cfg.scene.num_envs = 1
  return env_cfg, agent_cfg


def _tensor_np(tensor: torch.Tensor) -> np.ndarray:
  return tensor.detach().cpu().numpy()


def _policy_obs_np(obs: Any) -> np.ndarray:
  policy_obs = obs["policy"]
  return _tensor_np(policy_obs).astype(np.float32, copy=False)


def _collect_state(env: RslRlVecEnvWrapper) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  unwrapped = env.unwrapped
  robot = unwrapped.scene["robot"]
  obj = unwrapped.scene["object"]
  action_term = unwrapped.action_manager.get_term("joint_pos")
  target_ids = action_term.target_ids
  joint_pos = _tensor_np(robot.data.joint_pos[:, target_ids]).astype(np.float32)
  object_state = torch.cat(
    [obj.data.root_link_pos_w, obj.data.root_link_quat_w],
    dim=-1,
  )
  command = unwrapped.command_manager.get_term("reorient_command")
  goal_state = command.goal_quat
  return (
    joint_pos,
    _tensor_np(object_state).astype(np.float32),
    _tensor_np(goal_state).astype(np.float32),
  )


def _normalize_inputs(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
  return ((raw - mean) / (std + NORMALIZER_EPSILON)).astype(np.float32)


def _run_rollout(
  options: ExportOptions,
  mean: np.ndarray,
  std: np.ndarray,
) -> dict[str, np.ndarray]:
  device = options.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg, agent_cfg = _prepare_rollout_cfg(options)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  try:
    runner_cls = load_runner_cls(options.task) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(
      str(options.checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    obs = wrapped.get_observations()
    action_term = wrapped.unwrapped.action_manager.get_term("joint_pos")
    if tuple(action_term.target_names) != tuple(REVO3_RIGHT_HAND_BINDING.joint_names):
      raise ValueError(
        "Policy action target order does not match REVO3_RIGHT_HAND_BINDING."
      )

    frame_index: list[int] = []
    timestamp_sec: list[float] = []
    policy_input_raw: list[np.ndarray] = []
    policy_input_normalized: list[np.ndarray] = []
    action_raw: list[np.ndarray] = []
    action_clipped: list[np.ndarray] = []
    target_post_clip: list[np.ndarray] = []
    joint_pos_policy_order: list[np.ndarray] = []
    object_state: list[np.ndarray] = []
    goal_state: list[np.ndarray] = []
    done: list[bool] = []
    reset: list[bool] = []
    episode_id: list[int] = []

    for step in range(options.num_steps):
      raw_obs = _policy_obs_np(obs)
      joint_pos, obj_state, goal = _collect_state(wrapped)
      normalized_obs = _normalize_inputs(raw_obs, mean, std)

      with torch.inference_mode():
        action = policy(obs.to(device))

      raw_action_np = _tensor_np(action).astype(np.float32)
      clipped_np = np.clip(raw_action_np, -1.0, 1.0).astype(np.float32)

      next_obs, _rew, dones, _extras = wrapped.step(action)
      processed = _tensor_np(action_term.processed_action).astype(np.float32)
      done_np = _tensor_np(dones).astype(bool)

      frame_index.append(step)
      timestamp_sec.append(step * wrapped.unwrapped.step_dt)
      policy_input_raw.append(raw_obs[0].copy())
      policy_input_normalized.append(normalized_obs[0].copy())
      action_raw.append(raw_action_np[0].copy())
      action_clipped.append(clipped_np[0].copy())
      target_post_clip.append(processed[0].copy())
      joint_pos_policy_order.append(joint_pos[0].copy())
      object_state.append(obj_state[0].copy())
      goal_state.append(goal[0].copy())
      done.append(bool(done_np[0]))
      reset.append(bool(done_np[0]))
      episode_id.append(0)

      obs = next_obs

    joint_names = np.array(REVO3_RIGHT_HAND_BINDING.joint_names, dtype=str)
    return {
      "frame_index": np.asarray(frame_index, dtype=np.int64),
      "timestamp_sec": np.asarray(timestamp_sec, dtype=np.float64),
      "policy_input_raw": np.asarray(policy_input_raw, dtype=np.float32),
      "policy_input_normalized": np.asarray(policy_input_normalized, dtype=np.float32),
      "action_raw": np.asarray(action_raw, dtype=np.float32),
      "action_clipped": np.asarray(action_clipped, dtype=np.float32),
      "target_post_clip_policy_order": np.asarray(target_post_clip, dtype=np.float32),
      "joint_pos_policy_order": np.asarray(joint_pos_policy_order, dtype=np.float32),
      "object_state": np.asarray(object_state, dtype=np.float32),
      "goal_state": np.asarray(goal_state, dtype=np.float32),
      "done": np.asarray(done, dtype=bool),
      "reset": np.asarray(reset, dtype=bool),
      "episode_id": np.asarray(episode_id, dtype=np.int64),
      "joint_names_policy_order": joint_names,
    }
  finally:
    wrapped.close()


def _rollout_flags(options: ExportOptions) -> dict[str, Any]:
  return {
    "task": options.task,
    "num_steps": options.num_steps,
    "fixed_object_init": options.fixed_object_init,
    "single_goal": options.single_goal,
    "fixed_goal": options.fixed_goal,
    "fixed_goal_axis": options.fixed_goal_axis,
    "fixed_goal_deg": options.fixed_goal_deg,
    "cube_offset_in_palm": (
      list(options.cube_offset_in_palm)
      if options.cube_offset_in_palm is not None
      else None
    ),
    "no_terminations": options.no_terminations,
    "device": options.device,
  }


def _build_manifest(
  *,
  options: ExportOptions,
  onnx_meta: dict[str, Any],
  checkpoint_sha256: str,
  model_sha256: str,
  normalizer_present: bool,
) -> dict[str, Any]:
  fixed_goal_quat = _axis_angle_quat_wxyz(
    options.fixed_goal_axis,
    options.fixed_goal_deg,
  )
  return {
    "schema_version": SCHEMA_VERSION,
    "task_name": options.task,
    "checkpoint": {
      "path": str(options.checkpoint),
      "sha256": checkpoint_sha256,
    },
    "export_time": datetime.now(timezone.utc).isoformat(),
    "training_repo_commit": _git_commit(),
    "play_script": "play_rsl_rl_zero_wrist_pitch.py",
    "rollout_flags": _rollout_flags(options),
    "model": {
      "file": "policy.onnx",
      "sha256": model_sha256,
      "format": "onnx",
    },
    "model_input_kind": MODEL_INPUT_KIND,
    "inputs": [
      {
        "name": onnx_meta["input_name"],
        "shape": onnx_meta["input_shape"],
        "dtype": onnx_meta["input_dtype"],
        "semantics": "raw concatenated policy observation",
      }
    ],
    "outputs": [
      {
        "name": onnx_meta["output_name"],
        "shape": onnx_meta["output_shape"],
        "dtype": onnx_meta["output_dtype"],
        "semantics": "raw deterministic policy action mean",
      }
    ],
    "input_field_slices": list(INPUT_SLICES),
    "policy_joint_order": list(REVO3_RIGHT_HAND_BINDING.joint_names),
    "units": {
      "joint_position": "rad",
      "cube_position": "m",
      "quaternion": "wxyz",
      "policy_action": "dimensionless",
    },
    "normalization": {
      "type": "EmpiricalNormalization" if normalizer_present else "Identity",
      "applied_inside_model": True,
      "external_preprocessing_required": False,
      "epsilon": NORMALIZER_EPSILON,
      "array_names": {
        "mean": "input_mean",
        "std": "input_std",
        "epsilon": "epsilon",
        "var": "input_var",
        "count": "count",
      },
    },
    "action": {
      "semantics": "default_joint_position_offset_to_absolute_target",
      "raw_action_clamp": [-1.0, 1.0],
      "action_scale": 0.5,
      "ema_alpha": 0.5,
      "warmup_time_s": 0.4,
      "target_clamp": "soft_joint_position_limits",
    },
    "fixed_goal": {
      "enabled": options.fixed_goal,
      "axis": options.fixed_goal_axis,
      "degrees": options.fixed_goal_deg,
      "quat_tag_wxyz": list(fixed_goal_quat),
    },
    "compatibility": {
      "matches_model_action_extraction_readme": False,
      "old_expected_schema": OLD_README_SCHEMA,
      "actual_schema": ACTUAL_SCHEMA,
      "reason": (
        "This checkpoint exposes one raw 216-D obs input with the empirical "
        "normalizer inside the policy graph, not obs[126] plus proprio_hist[30,42]."
      ),
    },
  }


def _write_readme(path: Path, options: ExportOptions) -> None:
  export_cmd = (
    "pixi run python -m "
    "wuji_mjlab.tasks.reorient.scripts.export_revo3_play_artifact export "
    f"--checkpoint {options.checkpoint} --output-dir {options.output_dir} "
    f"--task {options.task} --num-steps {options.num_steps} "
    "--fixed-object-init --single-goal --fixed-goal "
    f"--fixed-goal-axis {options.fixed_goal_axis} "
    f"--fixed-goal-deg {options.fixed_goal_deg} --no-terminations"
  )
  validate_cmd = (
    "pixi run python -m "
    "wuji_mjlab.tasks.reorient.scripts.export_revo3_play_artifact validate "
    f"--artifact-dir {options.output_dir}"
  )
  rollout_cmd = (
    "pixi run python scripts/play/play_rsl_rl_zero_wrist_pitch.py "
    f"--task {options.task} --checkpoint-file {options.checkpoint} "
    "--viewer native --num-envs 1 --fixed-object-init --single-goal "
    f"--fixed-goal --fixed-goal-axis {options.fixed_goal_axis} "
    f"--fixed-goal-deg {options.fixed_goal_deg} --no-terminations "
    "--video True --video-length 1000 --video-width 640 --video-height 480"
  )
  path.write_text(
    "\n".join(
      [
        "# Revo3 216-D Raw-Input Policy Artifact",
        "",
        f"Checkpoint: `{options.checkpoint}`",
        "",
        "This artifact records the actual checkpoint interface: one raw "
        "`obs[1,216]` ONNX input and one `actions[1,21]` output. The empirical "
        "normalizer is inside `policy.onnx`; do not externally normalize before "
        "calling ONNX.",
        "",
        "The saved `policy_input_normalized` arrays are audit-only and are used "
        "to verify `(raw - input_mean) / (input_std + epsilon)`.",
        "",
        "This does not strictly match the old `MODEL_ACTION_EXTRACTION_README.md` "
        "schema of `obs[126] + proprio_hist[30,42]`; the manifest marks that "
        "compatibility as false.",
        "",
        "## Human Rollout Command",
        "",
        "```bash",
        rollout_cmd,
        "```",
        "",
        "## Export Command",
        "",
        "```bash",
        export_cmd,
        "```",
        "",
        "## Validation Command",
        "",
        "```bash",
        validate_cmd,
        "```",
        "",
        "Deployment/replay consumers should read `policy_manifest.yaml` and feed "
        "`policy_input_raw`-style 216-D observations to ONNX.",
        "",
      ]
    ),
    encoding="utf-8",
  )


def export_artifact(options: ExportOptions) -> Path:
  options.output_dir.mkdir(parents=True, exist_ok=True)
  state_dict, agent_cfg, _env_cfg = _load_actor_state(options.checkpoint)
  exporter, _arch, norm_state = _build_exporter(state_dict, agent_cfg)

  onnx_path = options.output_dir / "policy.onnx"
  onnx_meta = _export_onnx(exporter, onnx_path)
  _write_normalizer_npz(options.output_dir / "obs_normalizer.npz", norm_state)
  mean, std, _var, _count = _normalizer_arrays(norm_state)

  rollout = _run_rollout(options, mean, std)
  np.savez(options.output_dir / "replay_dataset.npz", **rollout)
  golden = {
    "policy_input_raw": rollout["policy_input_raw"][:1],
    "policy_input_normalized": rollout["policy_input_normalized"][:1],
    "action_raw": rollout["action_raw"][:1],
    "action_clipped": rollout["action_clipped"][:1],
    "target_post_clip_policy_order": rollout["target_post_clip_policy_order"][:1],
    "joint_pos_policy_order": rollout["joint_pos_policy_order"][:1],
    "object_state": rollout["object_state"][:1],
    "goal_state": rollout["goal_state"][:1],
    "joint_names_policy_order": rollout["joint_names_policy_order"],
  }
  np.savez(options.output_dir / "golden_io.npz", **golden)

  manifest = _build_manifest(
    options=options,
    onnx_meta=onnx_meta,
    checkpoint_sha256=sha256_file(options.checkpoint),
    model_sha256=sha256_file(onnx_path),
    normalizer_present=bool(norm_state),
  )
  with (options.output_dir / "policy_manifest.yaml").open("w", encoding="utf-8") as f:
    yaml.safe_dump(manifest, f, sort_keys=False)
  _write_readme(options.output_dir / "README.md", options)
  return options.output_dir


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ValueError(message)


def _load_npz_required(path: Path) -> dict[str, np.ndarray]:
  _require(path.exists(), f"Missing required file: {path}")
  with np.load(path, allow_pickle=False) as data:
    return {key: data[key] for key in data.files}


def _check_finite(name: str, array: np.ndarray) -> None:
  if array.dtype.kind in {"U", "S", "O", "b", "i", "u"}:
    return
  _require(np.isfinite(array).all(), f"{name} contains non-finite values.")


def _validate_manifest(manifest: dict[str, Any]) -> None:
  _require(
    manifest.get("schema_version") == SCHEMA_VERSION,
    f"Unexpected schema_version: {manifest.get('schema_version')!r}",
  )
  _require(
    manifest.get("model_input_kind") == MODEL_INPUT_KIND,
    "Manifest must declare model_input_kind=raw_obs_with_normalizer_inside_model.",
  )
  normalization = manifest.get("normalization")
  _require(isinstance(normalization, dict), "Manifest missing normalization block.")
  _require(
    normalization.get("applied_inside_model") is True,
    "Manifest must declare normalization.applied_inside_model: true.",
  )
  _require(
    normalization.get("external_preprocessing_required") is False,
    "Manifest must declare normalization.external_preprocessing_required: false.",
  )
  _require(
    float(normalization.get("epsilon")) == NORMALIZER_EPSILON,
    "Manifest normalization epsilon must be 1e-2.",
  )
  compatibility = manifest.get("compatibility")
  _require(isinstance(compatibility, dict), "Manifest missing compatibility block.")
  _require(
    compatibility.get("matches_model_action_extraction_readme") is False,
    "Manifest must explicitly mark old README compatibility as false.",
  )
  _require(
    compatibility.get("old_expected_schema") == OLD_README_SCHEMA,
    "Manifest old_expected_schema mismatch.",
  )
  _require(
    compatibility.get("actual_schema") == ACTUAL_SCHEMA,
    "Manifest actual_schema mismatch.",
  )


def _validate_onnx_metadata(manifest: dict[str, Any], onnx_path: Path) -> ort.InferenceSession:
  session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  inputs = session.get_inputs()
  outputs = session.get_outputs()
  _require(len(inputs) == 1, f"Expected one ONNX input, got {len(inputs)}.")
  _require(len(outputs) == 1, f"Expected one ONNX output, got {len(outputs)}.")
  manifest_input = manifest["inputs"][0]
  manifest_output = manifest["outputs"][0]
  inp = inputs[0]
  out = outputs[0]
  _require(inp.name == manifest_input["name"] == INPUT_NAME, "ONNX input name mismatch.")
  _require(out.name == manifest_output["name"] == OUTPUT_NAME, "ONNX output name mismatch.")
  _require(inp.type == "tensor(float)", f"ONNX input dtype mismatch: {inp.type}")
  _require(out.type == "tensor(float)", f"ONNX output dtype mismatch: {out.type}")
  _require(list(inp.shape) == manifest_input["shape"], "ONNX input shape mismatch.")
  _require(list(out.shape) == manifest_output["shape"], "ONNX output shape mismatch.")
  _require(list(inp.shape) == [1, INPUT_DIM], "ONNX input must be [1,216].")
  _require(list(out.shape) == [1, ACTION_DIM], "ONNX output must be [1,21].")
  return session


def _validate_normalizer(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
  for key in ("input_mean", "input_std", "epsilon"):
    _require(key in data, f"Normalizer missing array {key!r}.")
  mean = data["input_mean"]
  std = data["input_std"]
  epsilon = float(data["epsilon"])
  _require(mean.shape == (INPUT_DIM,), f"input_mean shape {mean.shape}, expected (216,).")
  _require(std.shape == (INPUT_DIM,), f"input_std shape {std.shape}, expected (216,).")
  _require(
    np.isclose(epsilon, NORMALIZER_EPSILON),
    "Normalizer epsilon must be 1e-2.",
  )
  _check_finite("input_mean", mean)
  _check_finite("input_std", std)
  _require(np.all(std > 0.0), "input_std must be positive.")
  return mean.astype(np.float32), std.astype(np.float32), epsilon


def _validate_golden(
  session: ort.InferenceSession,
  normalizer: tuple[np.ndarray, np.ndarray, float],
  golden: dict[str, np.ndarray],
) -> None:
  required = (
    "policy_input_raw",
    "policy_input_normalized",
    "action_raw",
    "joint_names_policy_order",
  )
  for key in required:
    _require(key in golden, f"golden_io missing array {key!r}.")
  raw = golden["policy_input_raw"]
  normalized = golden["policy_input_normalized"]
  action_raw = golden["action_raw"]
  _require(raw.shape == (1, INPUT_DIM), f"policy_input_raw shape {raw.shape}.")
  _require(
    normalized.shape == (1, INPUT_DIM),
    f"policy_input_normalized shape {normalized.shape}.",
  )
  _require(action_raw.shape == (1, ACTION_DIM), f"action_raw shape {action_raw.shape}.")
  for key, value in golden.items():
    _check_finite(f"golden_io.{key}", value)

  mean, std, epsilon = normalizer
  expected_normalized = ((raw - mean) / (std + epsilon)).astype(np.float32)
  np.testing.assert_allclose(
    normalized,
    expected_normalized,
    rtol=1.0e-5,
    atol=1.0e-5,
    err_msg="policy_input_normalized does not match normalizer formula.",
  )

  # Deliberately feed only raw input to ONNX. Feeding normalized input would
  # apply the embedded normalizer twice.
  output = session.run([OUTPUT_NAME], {INPUT_NAME: raw.astype(np.float32)})[0]
  np.testing.assert_allclose(
    output,
    action_raw,
    rtol=1.0e-5,
    atol=1.0e-5,
    err_msg="policy.onnx(policy_input_raw) does not match golden action_raw.",
  )


def _validate_replay(data: dict[str, np.ndarray]) -> None:
  required = (
    "frame_index",
    "timestamp_sec",
    "policy_input_raw",
    "policy_input_normalized",
    "action_raw",
    "done",
    "reset",
    "episode_id",
    "joint_names_policy_order",
  )
  for key in required:
    _require(key in data, f"replay_dataset missing array {key!r}.")
  n = data["frame_index"].shape[0]
  _require(data["policy_input_raw"].shape == (n, INPUT_DIM), "Replay raw input shape mismatch.")
  _require(
    data["policy_input_normalized"].shape == (n, INPUT_DIM),
    "Replay normalized input shape mismatch.",
  )
  _require(data["action_raw"].shape == (n, ACTION_DIM), "Replay action shape mismatch.")
  for key, value in data.items():
    _check_finite(f"replay_dataset.{key}", value)


def validate_artifact_dir(artifact_dir: Path) -> None:
  manifest_path = artifact_dir / "policy_manifest.yaml"
  _require(manifest_path.exists(), f"Missing manifest: {manifest_path}")
  with manifest_path.open(encoding="utf-8") as f:
    manifest = yaml.safe_load(f)
  _require(isinstance(manifest, dict), "Manifest must parse to a mapping.")
  _validate_manifest(manifest)

  session = _validate_onnx_metadata(manifest, artifact_dir / "policy.onnx")
  normalizer = _validate_normalizer(_load_npz_required(artifact_dir / "obs_normalizer.npz"))
  golden = _load_npz_required(artifact_dir / "golden_io.npz")
  replay = _load_npz_required(artifact_dir / "replay_dataset.npz")
  _validate_golden(session, normalizer, golden)
  _validate_replay(replay)


def add_export_args(parser: argparse.ArgumentParser) -> None:
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--task", default="Revo3RightHand_Reorient_ReposeReward_FineTune")
  parser.add_argument("--num-steps", type=int, default=1000)
  parser.add_argument("--device", default=None)
  parser.add_argument("--fixed-object-init", action="store_true")
  parser.add_argument("--single-goal", action="store_true")
  parser.add_argument("--fixed-goal", action="store_true")
  parser.add_argument("--fixed-goal-axis", choices=("x", "y", "z"), default="z")
  parser.add_argument("--fixed-goal-deg", type=float, default=90.0)
  parser.add_argument("--cube-offset-in-palm", nargs=3, type=float, default=None)
  parser.add_argument("--no-terminations", action="store_true")


def options_from_args(args: argparse.Namespace) -> ExportOptions:
  return ExportOptions(
    checkpoint=args.checkpoint,
    output_dir=args.output_dir,
    task=args.task,
    num_steps=args.num_steps,
    device=args.device,
    fixed_object_init=args.fixed_object_init,
    single_goal=args.single_goal,
    fixed_goal=args.fixed_goal,
    fixed_goal_axis=args.fixed_goal_axis,
    fixed_goal_deg=args.fixed_goal_deg,
    cube_offset_in_palm=_as_offset_tuple(args.cube_offset_in_palm),
    no_terminations=args.no_terminations,
  )
