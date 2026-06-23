# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml
from deploy.reorient.lib import revo3_artifact_trajectory as traj
from deploy.reorient.lib.revo3_policy_artifact import (
    ACTION_DIM,
    INPUT_DIM,
    MODEL_INPUT_KIND,
    OUTPUT_NAME,
    SCHEMA_VERSION,
)
from deploy.reorient.lib.revo3_profile import Revo3Profile
from deploy.reorient.scripts import replay_revo3_artifact

PROFILE_PATH = Path("deploy/reorient/config/revo3_right.yaml")
OFFSET_PATH = Path("deploy/reorient/config/revo3_right_offset_tuned.yaml")


class _RawFirst21(nn.Module):
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, :ACTION_DIM]


def _profile() -> Revo3Profile:
    return Revo3Profile.load(PROFILE_PATH)


def _write_policy_onnx(root: Path) -> None:
    torch.onnx.export(
        _RawFirst21(),
        torch.zeros(1, INPUT_DIM, dtype=torch.float32),
        root / "policy.onnx",
        export_params=True,
        opset_version=17,
        input_names=["obs"],
        output_names=[OUTPUT_NAME],
        dynamic_axes={},
        dynamo=False,
    )


def _manifest(policy_order: tuple[str, ...]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": {"file": "policy.onnx"},
        "model_input_kind": MODEL_INPUT_KIND,
        "inputs": [{"name": "obs", "shape": [1, INPUT_DIM], "dtype": "float32"}],
        "outputs": [{"name": "actions", "shape": [1, ACTION_DIM], "dtype": "float32"}],
        "policy_joint_order": list(policy_order),
        "normalization": {
            "applied_inside_model": True,
            "external_preprocessing_required": False,
            "epsilon": 1.0e-2,
        },
        "compatibility": {
            "matches_model_action_extraction_readme": False,
            "old_expected_schema": "obs[126] + proprio_hist[30,42]",
            "actual_schema": "single_input[216]",
        },
    }


def _write_artifact(
    root: Path,
    *,
    targets: np.ndarray | None = None,
    include_target: bool = True,
    include_fallback: bool = True,
    joint_order: tuple[str, ...] | None = None,
) -> Path:
    profile = _profile()
    policy_order = joint_order or profile.policy_joint_order
    root.mkdir(parents=True, exist_ok=True)
    _write_policy_onnx(root)
    with (root / "policy_manifest.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(_manifest(policy_order), f, sort_keys=False)

    if targets is None:
        targets = np.stack(
            [
                profile.home_joint_pos_policy,
                profile.home_joint_pos_policy + 0.01,
            ]
        ).astype(np.float32)

    payload = {
        "frame_index": np.arange(targets.shape[0], dtype=np.int64),
        "timestamp_sec": np.arange(targets.shape[0], dtype=np.float64) * 0.05,
        "joint_names_policy_order": np.asarray(policy_order, dtype=str),
        "action_raw": np.zeros((targets.shape[0], ACTION_DIM), dtype=np.float32),
    }
    if include_target:
        payload["target_post_clip_policy_order"] = targets.astype(np.float32)
    if include_fallback:
        payload["joint_pos_policy_order"] = (targets - 0.01).astype(np.float32)
    np.savez(root / "replay_dataset.npz", **payload)
    return root


def test_load_artifact_trajectory_prefers_target_post_clip(tmp_path: Path):
    artifact_dir = _write_artifact(tmp_path / "artifact")

    loaded = traj.load_artifact_trajectory(artifact_dir)

    assert loaded.target_source == "target_post_clip_policy_order"
    assert loaded.warning is None
    assert loaded.frame_count == 2
    np.testing.assert_allclose(
        loaded.targets_policy_order[0],
        _profile().home_joint_pos_policy,
    )


def test_load_artifact_trajectory_rejects_joint_order_mismatch(tmp_path: Path):
    profile = _profile()
    bad_order = list(profile.policy_joint_order)
    bad_order[0], bad_order[1] = bad_order[1], bad_order[0]
    artifact_dir = _write_artifact(
        tmp_path / "artifact",
        joint_order=profile.policy_joint_order,
    )
    with np.load(artifact_dir / "replay_dataset.npz", allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    payload["joint_names_policy_order"] = np.asarray(bad_order, dtype=str)
    np.savez(artifact_dir / "replay_dataset.npz", **payload)

    with pytest.raises(ValueError, match="joint_names_policy_order"):
        traj.load_artifact_trajectory(artifact_dir)


def test_load_artifact_trajectory_refuses_action_raw_as_target(tmp_path: Path):
    artifact_dir = _write_artifact(
        tmp_path / "artifact",
        include_target=False,
        include_fallback=False,
    )

    with pytest.raises(ValueError, match="refusing to derive"):
        traj.load_artifact_trajectory(artifact_dir)


def test_build_replay_plan_maps_policy_targets_to_sdk_with_offsets(tmp_path: Path):
    profile = _profile()
    artifact_dir = _write_artifact(tmp_path / "artifact")
    loaded = traj.load_artifact_trajectory(artifact_dir)

    plan = traj.build_replay_plan(loaded, profile, max_replay_frames=1)

    expected = profile.home_joint_pos_policy[profile.policy_to_sdk_perm]
    expected = expected + profile.sdk_offset_rad
    np.testing.assert_allclose(plan.frames[0].raw_final_sdk_order, expected)


def test_load_tuned_offset_file_validates_order_and_values():
    profile = _profile()

    offsets = traj.load_sdk_offset_file(OFFSET_PATH, profile.sdk_joint_order)

    assert offsets.shape == (ACTION_DIM,)
    assert np.isclose(offsets[0], -0.02)
    assert np.isclose(offsets[-1], -0.3)


def test_final_output_layer_clips_out_of_limit_target(tmp_path: Path):
    profile = _profile()
    target = profile.home_joint_pos_policy.copy()
    bad_policy_index = int(profile.policy_to_sdk_perm[0])
    target[bad_policy_index] = profile.joint_upper_sdk[0] + 1.0
    artifact_dir = _write_artifact(
        tmp_path / "artifact",
        targets=target.reshape(1, -1),
    )
    loaded = traj.load_artifact_trajectory(artifact_dir)

    plan = traj.build_replay_plan(loaded, profile)

    assert plan.clipped_count == 1
    assert plan.frames[0].clipped[0].reason == "final_output_clip"
    assert np.all(plan.frames[0].command_sdk_order <= profile.joint_upper_sdk)
    assert np.all(plan.frames[0].command_sdk_order >= profile.joint_lower_sdk)


def test_cli_dry_run_without_output_dir_writes_no_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    artifact_dir = _write_artifact(tmp_path / "artifact")

    class _NoSdk:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("dry-run must not instantiate the SDK driver")

    monkeypatch.setattr(replay_revo3_artifact, "Revo3SdkHandDriver", _NoSdk)

    rc = replay_revo3_artifact.main(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--profile",
            str(PROFILE_PATH),
            "--mode",
            "dry-run",
            "--max-replay-frames",
            "1",
            "--offset-file",
            str(OFFSET_PATH),
        ]
    )

    assert rc == 0
    assert not (tmp_path / "out").exists()


def test_cli_dry_run_writes_outputs_only_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    artifact_dir = _write_artifact(tmp_path / "artifact")
    output_dir = tmp_path / "out"

    class _NoSdk:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("dry-run must not instantiate the SDK driver")

    monkeypatch.setattr(replay_revo3_artifact, "Revo3SdkHandDriver", _NoSdk)

    rc = replay_revo3_artifact.main(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--profile",
            str(PROFILE_PATH),
            "--mode",
            "dry-run",
            "--max-replay-frames",
            "1",
            "--offset-file",
            str(OFFSET_PATH),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == 0
    assert (output_dir / "revo3_artifact_replay_log.npz").exists()
    assert (output_dir / "revo3_artifact_replay_summary.yaml").exists()


def test_cli_active_requires_explicit_confirmation(tmp_path: Path):
    rc = replay_revo3_artifact.main(
        [
            "--artifact-dir",
            str(tmp_path / "missing"),
            "--profile",
            str(PROFILE_PATH),
            "--mode",
            "active",
        ]
    )

    assert rc == 2


def test_cli_active_replays_with_confirmation_and_reports_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    artifact_dir = _write_artifact(tmp_path / "artifact")
    profile = _profile()

    class _MockDriver:
        def __init__(self) -> None:
            self.write_count = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def read_encoders(self) -> np.ndarray:
            return profile.home_joint_pos_policy.copy()

        def write_target(self, joint_targets: np.ndarray) -> None:
            np.asarray(joint_targets, dtype=np.float32).reshape(ACTION_DIM)
            self.write_count += 1

    driver = _MockDriver()
    monkeypatch.setattr(
        replay_revo3_artifact,
        "_build_driver",
        lambda args, loaded_profile: driver,
    )
    monkeypatch.setattr(replay_revo3_artifact, "_sleep_until", lambda target_time: None)

    rc = replay_revo3_artifact.main(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--profile",
            str(PROFILE_PATH),
            "--mode",
            "active",
            "--max-replay-frames",
            "2",
            "--offset-file",
            str(OFFSET_PATH),
            "--preposition-duration",
            "0.01",
            "--preposition-rate",
            "100",
            "--hold-before-replay",
            "0",
            "--i-understand-active-revo3-control",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert driver.write_count == 3
    assert "[active] replay complete; replay_frames=2" in out
