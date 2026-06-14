# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Importable core for the reorient ONNX export CLI.

Rebuilds the actor network and observation normalizer from a checkpoint
``state_dict`` without requiring task registration or environment
instantiation, then writes a single-file ONNX policy plus a derived
``config.json`` next to the checkpoint.

The thin CLI wrapper lives at ``scripts/export_onnx.py``; this module hosts
the pure, importable building blocks so that focused unit tests can exercise
``_build_architecture`` and ``_StandaloneExporter`` without spawning a sim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import yaml
from rsl_rl.modules import MLP, EmpiricalNormalization
from rsl_rl.modules.distribution import _MeanSliceDeterministicOutput

from wuji_mjlab.tasks.reorient.robot_bindings import (
  REVO3_RIGHT_HAND_BINDING,
  WUJI_RIGHT_HAND_BINDING,
  ReorientRobotBinding,
)

_ONNX_OPSET_VERSION = 13


class _StandaloneExporter(nn.Module):
  """Wraps normalizer + actor MLP for ONNX export.

  No output activation — matches act_inference() in actor_critic.py which
  returns raw actor output directly. Action clamping / rescaling is done
  downstream in each task's action processing.
  """

  def __init__(self, normalizer: nn.Module, actor: nn.Module):
    super().__init__()
    self.normalizer = normalizer
    self.actor = actor

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.actor(self.normalizer(x))


def _load_yaml(path: str) -> dict[str, Any] | None:
  """Load YAML; strip python tags so untrusted run dirs cannot trigger
  arbitrary object construction via yaml.unsafe_load."""
  if not os.path.exists(path):
    return None
  with open(path) as f:
    text = f.read()
  sanitized = re.sub(r"!!python/[^\s]+", "", text)
  try:
    data = yaml.safe_load(sanitized)
  except yaml.YAMLError:
    return None
  return data if isinstance(data, dict) else None


def _infer_architecture_from_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
  """Infer minimal actor shape information from checkpoint weights."""
  actor_weights: dict[int, torch.Size] = {}
  for key, value in state_dict.items():
    if not key.endswith(".weight"):
      continue
    if key.startswith("mlp."):
      parts = key.split(".")
      if len(parts) >= 3 and parts[1].isdigit():
        actor_weights[int(parts[1])] = value.shape
      continue
    if key.startswith("actor.mlp."):
      parts = key.split(".")
      if len(parts) >= 4 and parts[2].isdigit():
        actor_weights[int(parts[2])] = value.shape
      continue
    if key.startswith("actor."):
      parts = key.split(".")
      if len(parts) >= 3 and parts[1].isdigit():
        actor_weights[int(parts[1])] = value.shape

  if not actor_weights:
    raise ValueError("No actor weights found in checkpoint state_dict.")

  sorted_indices = sorted(actor_weights)
  last_weight = actor_weights[sorted_indices[-1]]
  return {
    "num_actor_obs": actor_weights[sorted_indices[0]][1],
    "hidden_dims": [actor_weights[idx][0] for idx in sorted_indices[:-1]],
    "actor_output_dim": last_weight[0],
  }


def _build_architecture(
  agent_cfg: dict[str, Any] | None, state_dict: dict[str, Any]
) -> dict[str, Any]:
  """Build actor architecture from agent.yaml with state_dict fallback."""
  inferred = _infer_architecture_from_state_dict(state_dict)
  policy_cfg: dict[str, Any] = {}
  if isinstance(agent_cfg, dict):
    if isinstance(agent_cfg.get("policy"), dict):
      policy_cfg = agent_cfg["policy"]
    elif isinstance(agent_cfg.get("actor"), dict):
      actor_cfg = agent_cfg["actor"]
      policy_cfg = {
        "actor_hidden_dims": actor_cfg.get("hidden_dims"),
        "activation": actor_cfg.get("activation"),
      }
      dist_cfg = actor_cfg.get("distribution_cfg")
      if isinstance(dist_cfg, dict):
        class_name = str(dist_cfg.get("class_name", ""))
        if (
          "HeteroscedasticGaussianDistribution" in class_name
          or "SoftplusGaussianDistribution" in class_name
        ):
          policy_cfg["state_dependent_std"] = True

  state_dependent_std = bool(policy_cfg.get("state_dependent_std", False))
  actor_output_dim = inferred["actor_output_dim"]
  if state_dependent_std:
    if actor_output_dim % 2 != 0:
      raise ValueError(
        "state_dependent_std=True but actor output dim is odd: "
        f"{actor_output_dim}. Expected 2 * num_actions."
      )
    num_actions = actor_output_dim // 2
    output_dim: int | list[int] = [2, num_actions]
  else:
    num_actions = actor_output_dim
    output_dim = num_actions

  hidden_dims = policy_cfg.get("actor_hidden_dims", inferred["hidden_dims"])
  if not isinstance(hidden_dims, (list, tuple)) or not hidden_dims:
    raise ValueError(f"Invalid actor_hidden_dims in agent config: {hidden_dims!r}.")

  num_actor_obs = inferred["num_actor_obs"]
  if "actor_obs_normalizer._mean" in state_dict:
    num_actor_obs = int(state_dict["actor_obs_normalizer._mean"].numel())
  elif "obs_normalizer._mean" in state_dict:
    num_actor_obs = int(state_dict["obs_normalizer._mean"].numel())

  return {
    "num_actor_obs": num_actor_obs,
    "hidden_dims": list(hidden_dims),
    "num_actions": num_actions,
    "actor_output_dim": output_dim,
    "activation": policy_cfg.get("activation", "elu"),
    "state_dependent_std": state_dependent_std,
  }


def _check_unsupported(
  state_dict: dict[str, Any], arch: dict[str, Any], agent_cfg: dict[str, Any] | None
) -> None:
  """Check for unsupported checkpoint configurations."""
  rnn_keys = [k for k in state_dict if "memory_a.rnn" in k or "memory_s.rnn" in k]
  if rnn_keys:
    raise ValueError(
      "Recurrent policies (LSTM/GRU) are not supported by this script. "
      "Use the training pipeline's built-in ONNX export instead."
    )

  # state_dependent_std is supported: we slice out only the mean at export time.

  policy_cfg = agent_cfg.get("policy", {}) if isinstance(agent_cfg, dict) else {}
  class_name = policy_cfg.get("class_name")
  if class_name not in (None, "ActorCritic"):
    raise ValueError(f"Unsupported policy class for standalone export: {class_name!r}.")


def _resolve_export_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
  """Resolve actor state_dict from either legacy or current checkpoint format."""
  if not isinstance(checkpoint, dict):
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
  if isinstance(checkpoint.get("model_state_dict"), dict):
    return checkpoint["model_state_dict"]
  if isinstance(checkpoint.get("actor_state_dict"), dict):
    return checkpoint["actor_state_dict"]
  # Fallback: checkpoint may already be a state_dict
  if any(k.startswith("actor.") or k.startswith("mlp.") for k in checkpoint):
    return checkpoint
  raise KeyError(
    "Could not resolve actor state_dict from checkpoint. "
    f"Available top-level keys: {list(checkpoint.keys())[:20]}"
  )


def _extract_normalizer_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
  if any(k.startswith("actor_obs_normalizer.") for k in state_dict):
    return {
      k.removeprefix("actor_obs_normalizer."): v
      for k, v in state_dict.items()
      if k.startswith("actor_obs_normalizer.")
    }
  if any(k.startswith("obs_normalizer.") for k in state_dict):
    return {
      k.removeprefix("obs_normalizer."): v
      for k, v in state_dict.items()
      if k.startswith("obs_normalizer.")
    }
  return {}


def _extract_actor_mlp_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
  actor_state: dict[str, Any] = {}
  for key, value in state_dict.items():
    k = key
    if k.startswith("actor."):
      k = k.removeprefix("actor.")
    if k.startswith("mlp."):
      k = k.removeprefix("mlp.")
    if k.startswith("obs_normalizer.") or k.startswith("distribution."):
      continue
    if re.match(r"^\d+\.(weight|bias)$", k):
      actor_state[k] = value
  return actor_state


def _get_policy_history_length(policy_obs_cfg: dict[str, Any]) -> int | None:
  group_history = policy_obs_cfg.get("history_length")
  if group_history is not None:
    return int(group_history)

  terms = policy_obs_cfg.get("terms", {})
  if not isinstance(terms, dict) or not terms:
    return None

  term_histories = []
  for term_cfg in terms.values():
    if isinstance(term_cfg, dict):
      term_histories.append(int(term_cfg.get("history_length", 0) or 0))
  return max(term_histories, default=None)


def _infer_control_config(env_cfg: dict[str, Any]) -> dict[str, Any]:
  config: dict[str, Any] = {}

  actions = env_cfg.get("actions", {})
  joint_action_cfg = None
  if isinstance(actions, dict):
    for term_cfg in actions.values():
      if isinstance(term_cfg, dict):
        joint_action_cfg = term_cfg
        break

  if isinstance(joint_action_cfg, dict):
    action_scale = joint_action_cfg.get("action_scale")
    ema_alpha = joint_action_cfg.get("ema_alpha")
    warmup_time_s = joint_action_cfg.get("warmup_time_s")

    if action_scale is not None:
      config["action_scale"] = action_scale
    if ema_alpha is not None:
      config["ema_alpha"] = ema_alpha
    if warmup_time_s is not None:
      config["warmup_time_s"] = warmup_time_s

    if ema_alpha is not None:
      config["control_mode"] = "absolute"
    elif action_scale is not None:
      config["control_mode"] = "absolute"
    else:
      config["control_mode"] = "joint_limits"
  else:
    config["control_mode"] = "unknown"

  return config


def _command_cfg_dict(env_cfg: dict[str, Any]) -> dict[str, Any]:
  commands = env_cfg.get("commands", {})
  if not isinstance(commands, dict):
    return {}
  command_cfg = commands.get("reorient_command", {})
  return command_cfg if isinstance(command_cfg, dict) else {}


def _infer_robot_binding(env_cfg: dict[str, Any]) -> tuple[str, ReorientRobotBinding]:
  task_id = str(env_cfg.get("task_id", ""))
  if task_id.startswith("Revo3RightHand"):
    return task_id, REVO3_RIGHT_HAND_BINDING
  if task_id.startswith("WujiHand"):
    return task_id, WUJI_RIGHT_HAND_BINDING

  command_cfg = _command_cfg_dict(env_cfg)
  palm_body_pattern = str(command_cfg.get("palm_body_pattern", ""))
  if palm_body_pattern == REVO3_RIGHT_HAND_BINDING.palm_body_pattern:
    return "Revo3RightHand_Reorient", REVO3_RIGHT_HAND_BINDING
  return "WujiHand_Reorient", WUJI_RIGHT_HAND_BINDING


def _build_config(run_dir: str, env_cfg: dict[str, Any] | None) -> dict[str, Any]:
  """Build task-aware config.json from params/env.yaml."""
  config: dict[str, Any] = {}

  if not isinstance(env_cfg, dict):
    return config

  config.update(_infer_control_config(env_cfg))
  task_id, robot_binding = _infer_robot_binding(env_cfg)
  config["task_id"] = task_id
  config["joint_names"] = list(robot_binding.joint_names)
  config["palm_body_name"] = robot_binding.viewer_body_name
  config["tag_in_palm_pos"] = list(robot_binding.tag_in_palm_pos)
  config["tag_in_palm_quat"] = list(robot_binding.tag_in_palm_quat)

  decimation = int(env_cfg.get("decimation", 5))
  sim_cfg = env_cfg.get("sim", {})
  timestep = 0.01
  if isinstance(sim_cfg, dict):
    timestep = float(sim_cfg.get("timestep", 0.01))
  config["ctrl_dt"] = decimation * timestep

  observations = env_cfg.get("observations", {})
  if isinstance(observations, dict):
    policy_obs = observations.get("policy", {})
    if isinstance(policy_obs, dict):
      history_len = _get_policy_history_length(policy_obs)
      if history_len is not None:
        config["history_len"] = history_len

  return config


def main() -> None:
  parser = argparse.ArgumentParser(description="Export policy checkpoint to ONNX")
  parser.add_argument("checkpoint", help="Path to model_*.pt checkpoint file")
  parser.add_argument("--filename", default="policy.onnx", help="Output ONNX filename")
  args = parser.parse_args()

  checkpoint_path = args.checkpoint
  if not os.path.exists(checkpoint_path):
    print(f"Error: checkpoint not found: {checkpoint_path}")
    sys.exit(1)

  run_dir = os.path.dirname(checkpoint_path)
  agent_cfg = _load_yaml(os.path.join(run_dir, "params", "agent.yaml"))
  env_cfg = _load_yaml(os.path.join(run_dir, "params", "env.yaml"))

  print(f"Loading checkpoint: {checkpoint_path}")
  ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
  state_dict = _resolve_export_state_dict(ckpt)

  arch = _build_architecture(agent_cfg, state_dict)
  print("Inferred architecture:")
  print(f"  num_actor_obs: {arch['num_actor_obs']}")
  print(f"  hidden_dims:   {arch['hidden_dims']}")
  print(f"  num_actions:   {arch['num_actions']}")
  print(f"  activation:    {arch['activation']}")

  _check_unsupported(state_dict, arch, agent_cfg)

  norm_state = _extract_normalizer_state_dict(state_dict)
  has_normalizer = bool(norm_state)
  if has_normalizer:
    normalizer = EmpiricalNormalization(arch["num_actor_obs"])
    normalizer.load_state_dict(norm_state)
    normalizer.eval()
    print("  normalizer:    EmpiricalNormalization (loaded)")
  else:
    normalizer = nn.Identity()
    print("  normalizer:    Identity (none in checkpoint)")

  actor = MLP(
    input_dim=arch["num_actor_obs"],
    output_dim=arch["actor_output_dim"],
    hidden_dims=arch["hidden_dims"],
    activation=arch["activation"],
  )
  actor_state = _extract_actor_mlp_state_dict(state_dict)
  actor.load_state_dict(actor_state)
  actor.eval()

  if arch["state_dependent_std"]:
    # MLP outputs [batch, 2, action_dim]; slice to keep only the mean.
    actor = nn.Sequential(actor, _MeanSliceDeterministicOutput())
  exporter = _StandaloneExporter(normalizer, actor)
  exporter.eval()

  onnx_path = os.path.join(run_dir, args.filename)
  dummy_input = torch.zeros(1, arch["num_actor_obs"])
  print(f"\nExporting to: {onnx_path}")
  torch.onnx.export(
    exporter,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=_ONNX_OPSET_VERSION,
    input_names=["obs"],
    output_names=["actions"],
    dynamic_axes={},
    dynamo=False,
  )

  data_path = onnx_path + ".data"
  if os.path.exists(data_path):
    model = onnx.load(onnx_path, load_external_data=True)
    onnx.save(model, onnx_path, save_as_external_data=False)
    os.remove(data_path)
    print("  Merged external data into single file")

  print("\nVerification:")
  session = ort.InferenceSession(onnx_path)
  input_name = session.get_inputs()[0].name
  output_name = session.get_outputs()[0].name
  input_shape = session.get_inputs()[0].shape
  output_shape = session.get_outputs()[0].shape
  print(f"  input:  {input_name} {input_shape}")
  print(f"  output: {output_name} {output_shape}")

  zero_input = np.zeros((1, arch["num_actor_obs"]), dtype=np.float32)
  result = session.run([output_name], {input_name: zero_input})[0]
  max_abs = np.max(np.abs(result))
  print(f"  zero-input max|output|: {max_abs:.6f}")

  config = _build_config(run_dir, env_cfg)
  config_path = os.path.join(run_dir, "config.json")
  with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
  print(f"\nGenerated config: {config_path}")
  print(f"  {json.dumps(config)}")

  print("\nDone.")
