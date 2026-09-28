# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Importable core for the reorient ONNX export CLI."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import yaml
from rsl_rl.modules import (
  MLP,
  EmpiricalNormalization,
  HeteroscedasticGaussianDistribution,
)

import wuji_mjlab

_ONNX_OPSET_VERSION = 13

_POLICY_SCHEMA = 1
_POLICY_IDENTITY_KEY = "wuji_policy_identity"
_POLICY_IDENTITY_FIELDS = (
  "policy_schema",
  "hand_gen",
  "robot_mjcf",
  "joint_order",
  "default_joint_pos",
  "default_joint_pos_source",
  "obs_term_names",
)


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
        has_global_std = any(
          key.endswith(("distribution.std_param", "distribution.log_std_param"))
          for key in state_dict
        )
        has_state_dependent_cfg = (
          dist_cfg.get("std_type") in {"scalar", "log"} or "min_std" in dist_cfg
        )
        if has_state_dependent_cfg and not has_global_std:
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
  rnn_keys = [k for k in state_dict if "memory_a.rnn" in k or "memory_s.rnn" in k]
  if rnn_keys:
    raise ValueError(
      "Recurrent policies (LSTM/GRU) are not supported by this script. "
      "Use the training pipeline's built-in ONNX export instead."
    )

  policy_cfg = agent_cfg.get("policy", {}) if isinstance(agent_cfg, dict) else {}
  class_name = policy_cfg.get("class_name")
  if class_name not in (None, "ActorCritic"):
    raise ValueError(f"Unsupported policy class for standalone export: {class_name!r}.")


def _resolve_export_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
  if not isinstance(checkpoint, dict):
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
  if isinstance(checkpoint.get("model_state_dict"), dict):
    return checkpoint["model_state_dict"]
  if isinstance(checkpoint.get("actor_state_dict"), dict):
    return checkpoint["actor_state_dict"]
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


def _build_policy_identity(env_cfg: dict[str, Any]) -> dict[str, Any]:
  try:
    robot_cfg = env_cfg["scene"]["entities"]["robot"]
    encoded_path = robot_cfg["spec_fn"]
  except (KeyError, IndexError, TypeError) as exc:
    raise ValueError(
      "Cannot find the robot MJCF in "
      "scene.entities.robot.spec_fn.state[1] from params/env.yaml"
    ) from exc
  while isinstance(encoded_path, dict):
    state = encoded_path.get("state")
    if (
      not isinstance(state, (list, tuple))
      or len(state) < 2
      or not isinstance(state[1], (list, tuple))
      or not state[1]
    ):
      break
    encoded_path = state[1][0]

  if isinstance(encoded_path, str):
    robot_mjcf = encoded_path
  elif (
    isinstance(encoded_path, (list, tuple))
    and encoded_path
    and all(isinstance(part, str) for part in encoded_path)
  ):
    robot_mjcf = str(Path(*encoded_path))
  else:
    raise ValueError(
      "Cannot decode the robot MJCF path from "
      f"scene.entities.robot.spec_fn.state[1]: {encoded_path!r}"
    )

  mjcf_parts = Path(robot_mjcf).parts
  anchor_index = next(
    (
      index
      for index in range(len(mjcf_parts) - 1)
      if mjcf_parts[index : index + 2] == ("src", "wuji_mjlab")
    ),
    None,
  )
  if anchor_index is None:
    raise FileNotFoundError(
      f"Cannot remap robot MJCF {robot_mjcf!r}: "
      "the recorded path has no src/wuji_mjlab/ anchor"
    )
  local_mjcf = (
    Path(wuji_mjlab.__file__).resolve().parent.joinpath(*mjcf_parts[anchor_index + 2 :])
  )
  if not local_mjcf.is_file():
    raise FileNotFoundError(
      f"Robot MJCF {robot_mjcf!r} remaps to {local_mjcf}, but that file does not exist"
    )

  if "wuji_hand2" in mjcf_parts:
    hand_gen = 2
  elif "wuji_hand" in mjcf_parts:
    hand_gen = 1
  else:
    raise ValueError(
      f"Cannot determine hand generation from robot MJCF path {robot_mjcf!r}"
    )

  if hand_gen == 1:
    from wuji_mjlab.assets.robots.wuji_hand.wuji_hand_cfg import _get_spec
  else:
    from wuji_mjlab.assets.robots.wuji_hand2.wuji_hand2_cfg import _get_spec

  try:
    spec = _get_spec(local_mjcf)
  except ValueError as exc:
    raise ValueError(f"Cannot parse robot MJCF {local_mjcf}: {exc}") from exc
  joint_order = [actuator.target for actuator in spec.actuators]
  if len(joint_order) != 20:
    raise ValueError(
      f"Robot MJCF {local_mjcf} has {len(joint_order)} actuators; expected 20"
    )
  if len(set(joint_order)) != len(joint_order):
    raise ValueError(f"Robot MJCF {local_mjcf} maps multiple actuators to one joint")

  try:
    joint_pos_rules = robot_cfg["init_state"]["joint_pos"]
  except (KeyError, TypeError) as exc:
    raise ValueError(
      "Cannot find scene.entities.robot.init_state.joint_pos in params/env.yaml"
    ) from exc
  if not isinstance(joint_pos_rules, dict):
    raise ValueError(
      "scene.entities.robot.init_state.joint_pos must be a regex-to-value mapping"
    )

  default_joint_pos: list[float] = []
  for joint_name in joint_order:
    matches: list[tuple[str, Any]] = []
    for pattern, value in joint_pos_rules.items():
      if not isinstance(pattern, str):
        raise ValueError(f"Joint position pattern must be a string, got {pattern!r}")
      try:
        matched = re.fullmatch(pattern, joint_name) is not None
      except re.error as exc:
        raise ValueError(f"Invalid joint position regex {pattern!r}: {exc}") from exc
      if matched:
        matches.append((pattern, value))
    if len(matches) != 1:
      matched_patterns = [pattern for pattern, _ in matches]
      raise ValueError(
        f"Joint {joint_name!r} matched {len(matches)} init_state.joint_pos "
        f"patterns {matched_patterns!r}; expected exactly one"
      )
    pattern, value = matches[0]
    try:
      default_joint_pos.append(float(value))
    except (TypeError, ValueError) as exc:
      raise ValueError(
        f"Joint position pattern {pattern!r} has non-numeric value {value!r}"
      ) from exc

  try:
    policy_terms = env_cfg["observations"]["policy"]["terms"]
  except (KeyError, TypeError) as exc:
    raise ValueError(
      "Cannot find observations.policy.terms in params/env.yaml"
    ) from exc
  if (
    not isinstance(policy_terms, dict)
    or not policy_terms
    or not all(isinstance(name, str) for name in policy_terms)
  ):
    raise ValueError("observations.policy.terms must be a non-empty named mapping")

  return {
    "policy_schema": _POLICY_SCHEMA,
    "hand_gen": hand_gen,
    "robot_mjcf": robot_mjcf,
    "joint_order": joint_order,
    "default_joint_pos": default_joint_pos,
    "default_joint_pos_source": "scene.entities.robot.init_state.joint_pos",
    "obs_term_names": list(policy_terms),
  }


def _policy_identity_from_config(config: dict[str, Any]) -> dict[str, Any]:
  missing = [field for field in _POLICY_IDENTITY_FIELDS if field not in config]
  if missing:
    raise ValueError(f"Export config is missing policy identity fields: {missing}")
  return {field: config[field] for field in _POLICY_IDENTITY_FIELDS}


def _set_policy_identity_metadata(
  model: onnx.ModelProto, identity: dict[str, Any]
) -> None:
  encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
  for prop in model.metadata_props:
    if prop.key == _POLICY_IDENTITY_KEY:
      prop.value = encoded
      return
  prop = model.metadata_props.add()
  prop.key = _POLICY_IDENTITY_KEY
  prop.value = encoded


def _build_config(run_dir: str, env_cfg: dict[str, Any] | None) -> dict[str, Any]:
  if not isinstance(env_cfg, dict):
    raise ValueError(
      f"Cannot build policy identity for {run_dir}: "
      "params/env.yaml is missing or could not be parsed"
    )

  config: dict[str, Any] = {}
  config.update(_infer_control_config(env_cfg))

  decimation = int(env_cfg.get("decimation", 5))
  sim_cfg = env_cfg.get("sim", {})
  timestep = 0.01
  if isinstance(sim_cfg, dict):
    mujoco_cfg = sim_cfg.get("mujoco", {})
    if isinstance(mujoco_cfg, dict) and "timestep" in mujoco_cfg:
      timestep = float(mujoco_cfg["timestep"])
    else:
      timestep = float(sim_cfg.get("timestep", 0.01))
  config["ctrl_dt"] = decimation * timestep

  observations = env_cfg.get("observations", {})
  if isinstance(observations, dict):
    policy_obs = observations.get("policy", {})
    if isinstance(policy_obs, dict):
      history_len = _get_policy_history_length(policy_obs)
      if history_len is not None:
        config["history_len"] = history_len

  config.update(_build_policy_identity(env_cfg))
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
  config = _build_config(run_dir, env_cfg)

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
    actor = nn.Sequential(
      actor,
      HeteroscedasticGaussianDistribution(
        arch["actor_output_dim"][1]
      ).as_deterministic_output_module(),
    )
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
  had_external_data = os.path.exists(data_path)
  model = onnx.load(onnx_path, load_external_data=True)
  _set_policy_identity_metadata(model, _policy_identity_from_config(config))
  onnx.save(model, onnx_path, save_as_external_data=False)
  if had_external_data:
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

  config_path = os.path.join(run_dir, "config.json")
  with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
  print(f"\nGenerated config: {config_path}")
  print(f"  {json.dumps(config)}")

  print("\nDone.")
