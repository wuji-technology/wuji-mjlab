# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml
from deploy.reorient.lib.revo3_hand_driver import MockRevo3HandDriver
from deploy.reorient.lib.revo3_policy_artifact import (
    ACTION_DIM,
    INPUT_DIM,
    MODEL_INPUT_KIND,
    SCHEMA_VERSION,
    Revo3PolicyArtifact,
    validate_revo3_manifest,
)
from deploy.reorient.lib.revo3_profile import Revo3Profile

PROFILE_PATH = Path("deploy/reorient/config/revo3_right.yaml")


class _RawFirst21(nn.Module):
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, :ACTION_DIM]


def _profile_joint_order() -> list[str]:
    return list(Revo3Profile.load(PROFILE_PATH).policy_joint_order)


def _manifest() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": {"file": "policy.onnx"},
        "model_input_kind": MODEL_INPUT_KIND,
        "inputs": [{"name": "obs", "shape": [1, INPUT_DIM], "dtype": "float32"}],
        "outputs": [{"name": "actions", "shape": [1, ACTION_DIM], "dtype": "float32"}],
        "input_field_slices": [
            {"name": "noisy_joint_angles", "start": 0, "end": 63, "shape": [3, 21]},
        ],
        "policy_joint_order": _profile_joint_order(),
        "normalization": {
            "applied_inside_model": True,
            "external_preprocessing_required": False,
            "epsilon": 1.0e-2,
        },
        "action": {
            "action_scale": 0.5,
            "ema_alpha": 0.5,
            "warmup_time_s": 0.4,
        },
        "compatibility": {
            "matches_model_action_extraction_readme": False,
            "old_expected_schema": "obs[126] + proprio_hist[30,42]",
            "actual_schema": "single_input[216]",
        },
    }


def _write_artifact(root: Path) -> tuple[np.ndarray, np.ndarray]:
    torch.onnx.export(
        _RawFirst21(),
        torch.zeros(1, INPUT_DIM, dtype=torch.float32),
        root / "policy.onnx",
        export_params=True,
        opset_version=17,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
        dynamo=False,
    )
    raw = np.linspace(-0.5, 0.5, INPUT_DIM, dtype=np.float32).reshape(1, -1)
    action = raw[:, :ACTION_DIM].copy()
    np.savez(
        root / "golden_io.npz",
        policy_input_raw=raw,
        policy_input_normalized=raw + 1.0,
        action_raw=action,
    )
    with (root / "policy_manifest.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(_manifest(), f, sort_keys=False)
    return raw, action


def _write_profile(tmp_path: Path, updates: dict) -> Path:
    with PROFILE_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.update(updates)
    path = tmp_path / "revo3_profile.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def test_manifest_rejects_old_126_plus_history_schema():
    manifest = _manifest()
    manifest["inputs"] = [
        {"name": "obs", "shape": [1, 126], "dtype": "float32"},
        {"name": "proprio_hist", "shape": [1, 30, 42], "dtype": "float32"},
    ]

    with pytest.raises(ValueError, match="shape"):
        validate_revo3_manifest(manifest)


def test_manifest_rejects_external_normalization_boundary():
    manifest = _manifest()
    manifest["normalization"]["external_preprocessing_required"] = True

    with pytest.raises(ValueError, match="external_preprocessing_required"):
        validate_revo3_manifest(manifest)


def test_manifest_requires_explicit_old_readme_incompatibility():
    manifest = _manifest()
    manifest["compatibility"]["matches_model_action_extraction_readme"] = True

    with pytest.raises(ValueError, match="compatibility"):
        validate_revo3_manifest(manifest)


def test_policy_artifact_feeds_raw_216d_to_onnx(tmp_path: Path):
    raw, expected = _write_artifact(tmp_path)

    artifact = Revo3PolicyArtifact.load(tmp_path)
    artifact.validate_golden_io()

    assert np.allclose(artifact(raw), expected[0])
    assert artifact.env_policy_config["history_len"] == 3.0


def test_revo3_profile_joint_mapping_roundtrips_home_pose():
    profile = Revo3Profile.load(PROFILE_PATH)

    sdk_target = profile.target_policy_to_sdk(profile.home_joint_pos_policy)
    roundtrip = profile.measured_sdk_to_policy(sdk_target)

    assert np.allclose(roundtrip, profile.home_joint_pos_policy, atol=1e-6)
    assert profile.policy_joint_order[0] == "right_thumb_CMP_joint"
    assert profile.sdk_joint_order[0] == "right_little_MPR_joint"


def test_revo3_profile_rejects_duplicate_policy_joint(tmp_path: Path):
    with PROFILE_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["policy_joint_order"][1] = cfg["policy_joint_order"][0]
    path = tmp_path / "bad_profile.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    with pytest.raises(ValueError, match="duplicate"):
        Revo3Profile.load(path)


def test_vision_to_policy_frame_transform(tmp_path: Path):
    half = float(np.sqrt(0.5))
    path = _write_profile(
        tmp_path,
        {
            "vision_to_policy_frame": {
                "calibrated": True,
                "position_xyz": [0.0, 0.0, 0.0],
                "quat_wxyz": [half, 0.0, 0.0, half],
            }
        },
    )
    profile = Revo3Profile.load(path)

    pos, quat = profile.transform_vision_pose_to_policy(
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )

    assert profile.vision_transform_calibrated is True
    assert np.allclose(pos, np.array([0.0, 1.0, 0.0]), atol=1e-6)
    assert np.allclose(quat, np.array([half, 0.0, 0.0, half]), atol=1e-6)


def test_mock_revo3_shadow_driver_records_without_tracking():
    profile = Revo3Profile.load(PROFILE_PATH)
    driver = MockRevo3HandDriver(profile, track_targets=False)
    home = driver.read_encoders()
    target = home.copy()
    target[0] += 0.1

    driver.write_target(target)

    assert driver.write_count == 1
    assert np.allclose(driver.last_target_policy, target)
    assert np.allclose(driver.read_encoders(), home)
