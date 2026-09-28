# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""ONNX policy runner with model-intrinsic config and robot identity checks."""
from __future__ import annotations

import json
import os
import shlex
from typing import Any

import numpy as np
import onnxruntime as ort

from .constants import Constants, joint_names, load_constants

POLICY_SCHEMA = 1
POLICY_IDENTITY_KEY = "wuji_policy_identity"
POLICY_IDENTITY_FIELDS = (
  "policy_schema",
  "hand_gen",
  "robot_mjcf",
  "joint_order",
  "default_joint_pos",
  "default_joint_pos_source",
  "obs_term_names",
)


class ONNXPolicy:
  """Run a deployed ONNX policy with its model-specific configuration."""

  def __init__(
    self,
    onnx_path: str,
    constants: Constants,
    config_path: str | None = None,
    *,
    hand_side: str = "right",
  ):
    """Load and validate an ONNX policy and its deployment configuration."""
    self.onnx_path = str(onnx_path)
    self.constants = constants
    self.hand_side = hand_side
    self.session = ort.InferenceSession(
      self.onnx_path, providers=["CPUExecutionProvider"]
    )
    self._in = self.session.get_inputs()[0].name
    self._out = self.session.get_outputs()[0].name
    in_dim = self.session.get_inputs()[0].shape[-1]
    out_dim = self.session.get_outputs()[0].shape[-1]
    if in_dim != constants.obs_dim or out_dim != constants.action_dim:
      raise ValueError(
        f"ONNX IO ({in_dim}->{out_dim}) != expected "
        f"({constants.obs_dim}->{constants.action_dim})"
      )

    self.config: dict[str, Any] = self._load_config(config_path)
    self.identity = self._load_identity()
    self._validate_policy_identity(self.identity)

    if self.config.get("control_mode", "absolute") != "absolute":
      raise ValueError("Only control_mode='absolute' is supported")
    if int(self.config.get("history_len", constants.history_length)) != constants.history_length:
      raise ValueError(
        f"history_len {self.config['history_len']} != {constants.history_length}"
      )

  def _load_config(self, config_path: str | None) -> dict[str, Any]:
    for cand in (
      config_path,
      self.onnx_path + ".config.json",
      os.path.join(os.path.dirname(self.onnx_path), "config.json"),
    ):
      if cand and os.path.exists(cand):
        with open(cand) as f:
          return json.load(f)
    raise FileNotFoundError(
      f"No config sidecar for {self.onnx_path}: expected "
      f"{self.onnx_path}.config.json or config.json beside it. "
      "The exporter writes this; without it control params "
      "(action_scale, ctrl_dt, ema_alpha, ...) are unknown."
    )

  def _load_identity(self) -> dict[str, Any]:
    metadata_map = self.session.get_modelmeta().custom_metadata_map
    metadata_identity: dict[str, Any] | None = None
    if POLICY_IDENTITY_KEY in metadata_map:
      raw_identity = metadata_map[POLICY_IDENTITY_KEY]
      try:
        decoded_identity = json.loads(raw_identity)
      except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
          f"Invalid {POLICY_IDENTITY_KEY!r} JSON in ONNX metadata"
        ) from exc
      if not isinstance(decoded_identity, dict):
        raise ValueError(
          f"Invalid {POLICY_IDENTITY_KEY!r} in ONNX metadata: expected an object"
        )
      metadata_identity = self._identity_from_mapping(
        decoded_identity, "ONNX metadata"
      )
      if metadata_identity is None:
        raise self._missing_identity_error("ONNX identity metadata is empty")

    sidecar_identity = self._identity_from_mapping(self.config, "config.json")
    if (
      metadata_identity is not None
      and sidecar_identity is not None
      and metadata_identity != sidecar_identity
    ):
      raise ValueError(
        "Policy identity disagreement: ONNX metadata and config.json contain "
        "different signatures; the sidecar may have been replaced"
      )

    identity = metadata_identity or sidecar_identity
    if identity is None:
      raise self._missing_identity_error(
        "Policy identity is absent from ONNX metadata and config.json"
      )
    return identity

  def _identity_from_mapping(
    self, mapping: dict[str, Any], source: str
  ) -> dict[str, Any] | None:
    present = [field for field in POLICY_IDENTITY_FIELDS if field in mapping]
    if not present:
      return None
    missing = [field for field in POLICY_IDENTITY_FIELDS if field not in mapping]
    if missing:
      raise self._missing_identity_error(
        f"{source} has an incomplete policy identity; missing {missing}"
      )
    identity = {field: mapping[field] for field in POLICY_IDENTITY_FIELDS}
    self._validate_identity_shape(identity, source)
    return identity

  def _validate_identity_shape(
    self, identity: dict[str, Any], source: str
  ) -> None:
    if (
      type(identity["policy_schema"]) is not int
      or identity["policy_schema"] != POLICY_SCHEMA
    ):
      raise ValueError(
        f"Unsupported policy_schema in {source}: "
        f"{identity['policy_schema']!r}; expected {POLICY_SCHEMA}"
      )
    if type(identity["hand_gen"]) is not int or identity["hand_gen"] not in (1, 2):
      raise ValueError(f"Invalid diagnostic hand_gen in {source}: {identity['hand_gen']!r}")
    if not isinstance(identity["robot_mjcf"], str) or not identity["robot_mjcf"]:
      raise ValueError(f"Invalid robot_mjcf in {source}: expected a non-empty string")

    recorded_order = identity["joint_order"]
    if (
      not isinstance(recorded_order, list)
      or len(recorded_order) != self.constants.num_joints
      or not all(isinstance(name, str) for name in recorded_order)
    ):
      raise ValueError(
        f"Invalid joint_order in {source}: expected "
        f"{self.constants.num_joints} joint names"
      )

    recorded_default = identity["default_joint_pos"]
    try:
      default_values = np.asarray(recorded_default, dtype=np.float64)
    except (TypeError, ValueError) as exc:
      raise ValueError(
        f"Invalid default_joint_pos in {source}: expected numeric values"
      ) from exc
    if (
      not isinstance(recorded_default, list)
      or default_values.shape != (self.constants.num_joints,)
      or not np.all(np.isfinite(default_values))
    ):
      raise ValueError(
        f"Invalid default_joint_pos in {source}: expected "
        f"{self.constants.num_joints} finite values"
      )

    if (
      not isinstance(identity["default_joint_pos_source"], str)
      or not identity["default_joint_pos_source"]
    ):
      raise ValueError(
        f"Invalid default_joint_pos_source in {source}: "
        "expected a non-empty string"
      )

    recorded_obs = identity["obs_term_names"]
    if (
      not isinstance(recorded_obs, list)
      or not recorded_obs
      or not all(isinstance(name, str) for name in recorded_obs)
    ):
      raise ValueError(f"Invalid obs_term_names in {source}: expected named terms")

  def _missing_identity_error(self, detail: str) -> ValueError:
    generation = self._constants_generation()
    quoted_path = shlex.quote(self.onnx_path)
    if generation is not None:
      command = (
        "pixi run -e reorient-deploy python "
        "src/wuji_mjlab/tasks/reorient/scripts/stamp_policy_metadata.py "
        f"{quoted_path} --gen {generation} --hand {self.hand_side} "
        f"--i-verified-this-is-gen{generation}"
      )
    else:
      command = (
        "pixi run -e reorient-deploy python "
        "src/wuji_mjlab/tasks/reorient/scripts/stamp_policy_metadata.py "
        f"{quoted_path} --gen 1 --hand {self.hand_side} "
        "--i-verified-this-is-gen1"
      )
    return ValueError(
      f"{detail}. This policy is unsafe to load without a semantic identity. "
      f"Stamp a verified historical artifact with:\n  {command}"
    )

  def _constants_generation(self) -> int | None:
    for generation in (1, 2):
      candidate = load_constants(generation, self.hand_side)
      if (
        candidate.num_joints == self.constants.num_joints
        and candidate.history_length == self.constants.history_length
        and candidate.obs_dim == self.constants.obs_dim
        and candidate.action_dim == self.constants.action_dim
        and list(candidate.obs_terms) == list(self.constants.obs_terms)
        and np.allclose(
          candidate.default_joint_pos,
          self.constants.default_joint_pos,
          rtol=0,
          atol=1e-6,
        )
      ):
        return generation
    return None

  def _identity_matches(
    self, identity: dict[str, Any], constants: Constants
  ) -> bool:
    return (
      identity["joint_order"] == joint_names(self.hand_side)
      and np.allclose(
        np.asarray(identity["default_joint_pos"], dtype=np.float64),
        constants.default_joint_pos,
        rtol=0,
        atol=1e-6,
      )
      and identity["obs_term_names"] == [name for name, _ in constants.obs_terms]
    )

  def _validate_policy_identity(self, identity: dict[str, Any]) -> None:
    if self._identity_matches(identity, self.constants):
      return

    expected_generation = self._constants_generation()
    matching_generations = [
      generation
      for generation in (1, 2)
      if self._identity_matches(
        identity, load_constants(generation, self.hand_side)
      )
    ]
    if (
      expected_generation is not None
      and matching_generations
      and matching_generations[0] != expected_generation
    ):
      recorded_generation = matching_generations[0]
      raise ValueError(
        "Policy generation mismatch: this is semantically a "
        f"gen{recorded_generation} policy, but it was loaded with "
        f"gen{expected_generation} constants. The wrong policy file was "
        "selected; change --gen or use the matching model"
      )

    differences = self._identity_differences(identity)
    raise ValueError(
      "Policy constant drift: the recorded signature matches neither gen1 nor "
      "gen2 constants in this checkout. The training-side configuration "
      "changed; deployment constants must be synchronized. "
      f"Differences: {differences}"
    )

  def _identity_differences(self, identity: dict[str, Any]) -> str:
    differences: list[str] = []
    expected_order = joint_names(self.hand_side)
    recorded_order = identity["joint_order"]
    if recorded_order != expected_order:
      order_differences = [
        f"[{index}] recorded={recorded!r}, deploy={expected!r}"
        for index, (recorded, expected) in enumerate(
          zip(recorded_order, expected_order, strict=True)
        )
        if recorded != expected
      ]
      differences.append("joint_order " + ", ".join(order_differences[:4]))

    recorded_default = np.asarray(identity["default_joint_pos"], dtype=np.float64)
    expected_default = np.asarray(self.constants.default_joint_pos, dtype=np.float64)
    mismatch_indices = np.flatnonzero(
      ~np.isclose(recorded_default, expected_default, rtol=0, atol=1e-6)
    )
    if mismatch_indices.size:
      joint_differences = [
        f"{expected_order[index]}: recorded={recorded_default[index]:.9g}, "
        f"deploy={expected_default[index]:.9g}"
        for index in mismatch_indices[:4]
      ]
      differences.append("default_joint_pos " + ", ".join(joint_differences))

    expected_obs = [name for name, _ in self.constants.obs_terms]
    if identity["obs_term_names"] != expected_obs:
      differences.append(
        f"obs_term_names recorded={identity['obs_term_names']!r}, "
        f"deploy={expected_obs!r}"
      )
    return "; ".join(differences) or "semantic signature differs"

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    """Run the policy on one observation vector."""
    x = np.asarray(obs, dtype=np.float32).reshape(1, self.constants.obs_dim)
    out = self.session.run([self._out], {self._in: x})[0]
    return np.asarray(out, dtype=np.float64).reshape(self.constants.action_dim)
