# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from wuji_mjlab.tasks.reorient.tooling import revo3_play_artifact as artifact


def _manifest() -> dict:
  return {
    "schema_version": artifact.SCHEMA_VERSION,
    "model_input_kind": artifact.MODEL_INPUT_KIND,
    "inputs": [
      {
        "name": artifact.INPUT_NAME,
        "shape": [1, artifact.INPUT_DIM],
        "dtype": "float32",
      }
    ],
    "outputs": [
      {
        "name": artifact.OUTPUT_NAME,
        "shape": [1, artifact.ACTION_DIM],
        "dtype": "float32",
      }
    ],
    "normalization": {
      "applied_inside_model": True,
      "external_preprocessing_required": False,
      "epsilon": artifact.NORMALIZER_EPSILON,
    },
    "compatibility": {
      "matches_model_action_extraction_readme": False,
      "old_expected_schema": artifact.OLD_README_SCHEMA,
      "actual_schema": artifact.ACTUAL_SCHEMA,
    },
  }


def test_manifest_requires_raw_input_boundary_and_incompatibility():
  manifest = _manifest()

  artifact._validate_manifest(manifest)

  assert manifest["model_input_kind"] == "raw_obs_with_normalizer_inside_model"
  assert manifest["normalization"]["external_preprocessing_required"] is False
  assert manifest["compatibility"]["matches_model_action_extraction_readme"] is False


def test_manifest_rejects_missing_external_preprocessing_false():
  manifest = _manifest()
  manifest["normalization"].pop("external_preprocessing_required")

  with pytest.raises(ValueError, match="external_preprocessing_required"):
    artifact._validate_manifest(manifest)


def test_manifest_rejects_old_readme_compatibility_true():
  manifest = _manifest()
  manifest["compatibility"]["matches_model_action_extraction_readme"] = True

  with pytest.raises(ValueError, match="compatibility as false"):
    artifact._validate_manifest(manifest)


def test_normalizer_npz_contract(tmp_path: Path):
  path = tmp_path / "obs_normalizer.npz"
  norm_state = {
    "_mean": torch.zeros(1, artifact.INPUT_DIM),
    "_std": torch.ones(1, artifact.INPUT_DIM),
    "_var": torch.ones(1, artifact.INPUT_DIM),
    "count": torch.tensor(123, dtype=torch.long),
  }

  artifact._write_normalizer_npz(path, norm_state)
  data = artifact._load_npz_required(path)
  mean, std, epsilon = artifact._validate_normalizer(data)

  assert mean.shape == (artifact.INPUT_DIM,)
  assert std.shape == (artifact.INPUT_DIM,)
  assert np.isclose(epsilon, artifact.NORMALIZER_EPSILON)
  assert data["input_var"].shape == (artifact.INPUT_DIM,)
  assert int(data["count"]) == 123


class _RawFirst21(nn.Module):
  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return obs[:, : artifact.ACTION_DIM]


def _write_minimal_artifact(root: Path) -> None:
  root.mkdir(exist_ok=True)
  onnx_path = root / "policy.onnx"
  torch.onnx.export(
    _RawFirst21(),
    torch.zeros(1, artifact.INPUT_DIM, dtype=torch.float32),
    onnx_path,
    export_params=True,
    opset_version=artifact.onnx_export_core._ONNX_OPSET_VERSION,
    input_names=[artifact.INPUT_NAME],
    output_names=[artifact.OUTPUT_NAME],
    dynamic_axes={},
    dynamo=False,
  )

  raw = np.linspace(-0.5, 0.5, artifact.INPUT_DIM, dtype=np.float32).reshape(1, -1)
  mean = np.ones(artifact.INPUT_DIM, dtype=np.float32)
  std = np.full(artifact.INPUT_DIM, 2.0, dtype=np.float32)
  normalized = ((raw - mean) / (std + artifact.NORMALIZER_EPSILON)).astype(np.float32)
  action = raw[:, : artifact.ACTION_DIM].astype(np.float32)
  joint_names = np.array([f"j{i}" for i in range(artifact.ACTION_DIM)], dtype=str)

  np.savez(
    root / "obs_normalizer.npz",
    input_mean=mean,
    input_std=std,
    epsilon=np.array(artifact.NORMALIZER_EPSILON, dtype=np.float32),
  )
  np.savez(
    root / "golden_io.npz",
    policy_input_raw=raw,
    policy_input_normalized=normalized,
    action_raw=action,
    action_clipped=np.clip(action, -1.0, 1.0),
    target_post_clip_policy_order=action,
    joint_pos_policy_order=np.zeros((1, artifact.ACTION_DIM), dtype=np.float32),
    object_state=np.zeros((1, 7), dtype=np.float32),
    goal_state=np.zeros((1, 4), dtype=np.float32),
    joint_names_policy_order=joint_names,
  )
  np.savez(
    root / "replay_dataset.npz",
    frame_index=np.array([0], dtype=np.int64),
    timestamp_sec=np.array([0.0], dtype=np.float64),
    policy_input_raw=raw,
    policy_input_normalized=normalized,
    action_raw=action,
    action_clipped=np.clip(action, -1.0, 1.0),
    target_post_clip_policy_order=action,
    joint_pos_policy_order=np.zeros((1, artifact.ACTION_DIM), dtype=np.float32),
    object_state=np.zeros((1, 7), dtype=np.float32),
    goal_state=np.zeros((1, 4), dtype=np.float32),
    done=np.array([False]),
    reset=np.array([False]),
    episode_id=np.array([0], dtype=np.int64),
    joint_names_policy_order=joint_names,
  )
  with (root / "policy_manifest.yaml").open("w", encoding="utf-8") as f:
    yaml.safe_dump(_manifest(), f)


def test_validation_feeds_policy_input_raw_to_onnx(tmp_path: Path):
  _write_minimal_artifact(tmp_path)

  artifact.validate_artifact_dir(tmp_path)
