# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Manifest-driven Revo3 policy artifact loader.

The current Revo3 policy contract is intentionally narrow:

    raw obs[216] -> policy.onnx -> actions[21]

The empirical normalizer is inside the ONNX graph.  ``obs_normalizer.npz`` is
audit data only and must not be applied before inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import yaml

SCHEMA_VERSION = "revo3_play_single_input_216_v1"
MODEL_INPUT_KIND = "raw_obs_with_normalizer_inside_model"
INPUT_NAME = "obs"
OUTPUT_NAME = "actions"
INPUT_DIM = 216
ACTION_DIM = 21
OLD_README_SCHEMA = "obs[126] + proprio_hist[30,42]"
ACTUAL_SCHEMA = "single_input[216]"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def _shape_last_dim(shape: Any) -> int | None:
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        return None
    try:
        return int(shape[-1])
    except (TypeError, ValueError):
        return None


def _batch_dim_ok(shape: Any) -> bool:
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        return False
    return shape[0] in (1, "1", "B", "batch")


def _find_io_entry(entries: Any, name: str) -> dict[str, Any]:
    if not isinstance(entries, list):
        raise ValueError("manifest inputs/outputs must be lists.")
    for entry in entries:
        if isinstance(entry, dict) and entry.get("name") == name:
            return entry
    raise ValueError(f"manifest missing I/O entry named {name!r}.")


def validate_revo3_manifest(manifest: dict[str, Any]) -> None:
    """Validate the deploy-time Revo3 216-D policy contract."""
    schema = manifest.get("schema_version", manifest.get("schema"))
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Revo3 policy schema {schema!r}; expected {SCHEMA_VERSION!r}."
        )
    if manifest.get("model_input_kind") != MODEL_INPUT_KIND:
        raise ValueError(
            "policy_manifest.yaml must declare model_input_kind="
            f"{MODEL_INPUT_KIND!r}."
        )

    input_entry = _find_io_entry(manifest.get("inputs"), INPUT_NAME)
    input_shape = input_entry.get("shape")
    if not _batch_dim_ok(input_shape) or _shape_last_dim(input_shape) != INPUT_DIM:
        raise ValueError(
            f"manifest input {INPUT_NAME!r} must have shape [1, {INPUT_DIM}], "
            f"got {input_shape!r}."
        )
    if input_entry.get("dtype") != "float32":
        raise ValueError(f"manifest input {INPUT_NAME!r} must be float32.")

    output_entry = _find_io_entry(manifest.get("outputs"), OUTPUT_NAME)
    output_shape = output_entry.get("shape")
    if not _batch_dim_ok(output_shape) or _shape_last_dim(output_shape) != ACTION_DIM:
        raise ValueError(
            f"manifest output {OUTPUT_NAME!r} must have shape [1, {ACTION_DIM}], "
            f"got {output_shape!r}."
        )
    if output_entry.get("dtype") != "float32":
        raise ValueError(f"manifest output {OUTPUT_NAME!r} must be float32.")

    normalization = manifest.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("manifest missing normalization mapping.")
    if normalization.get("applied_inside_model") is not True:
        raise ValueError("normalization.applied_inside_model must be true.")
    if normalization.get("external_preprocessing_required") is not False:
        raise ValueError(
            "normalization.external_preprocessing_required must be false; "
            "external normalization would double-normalize this policy."
        )

    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, dict):
        raise ValueError("manifest missing compatibility mapping.")
    if compatibility.get("matches_model_action_extraction_readme") is not False:
        raise ValueError(
            "manifest must record old README compatibility as false for this policy."
        )
    if compatibility.get("old_expected_schema") != OLD_README_SCHEMA:
        raise ValueError("manifest must record the old 126+history expected schema.")
    if compatibility.get("actual_schema") != ACTUAL_SCHEMA:
        raise ValueError("manifest must record actual_schema=single_input[216].")


@dataclass(frozen=True)
class Revo3PolicyArtifact:
    """Loaded Revo3 policy artifact.

    ``__call__`` accepts only raw 216-D observations.  It never reads
    ``obs_normalizer.npz`` and never normalizes at runtime.
    """

    artifact_dir: Path
    manifest: dict[str, Any]
    session: ort.InferenceSession
    input_name: str
    output_name: str

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        *,
        use_gpu: bool = False,
    ) -> "Revo3PolicyArtifact":
        root = Path(artifact_dir)
        manifest_path = root / "policy_manifest.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(f"policy_manifest.yaml not found: {manifest_path}")
        manifest = _load_yaml(manifest_path)
        validate_revo3_manifest(manifest)

        model_file = str((manifest.get("model") or {}).get("file") or "policy.onnx")
        onnx_path = root / model_file
        if not onnx_path.exists():
            raise FileNotFoundError(f"policy ONNX not found: {onnx_path}")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        obj = cls(
            artifact_dir=root,
            manifest=manifest,
            session=session,
            input_name=session.get_inputs()[0].name,
            output_name=session.get_outputs()[0].name,
        )
        obj.validate_onnx_io()
        return obj

    @property
    def policy_joint_order(self) -> tuple[str, ...]:
        names = self.manifest.get("policy_joint_order") or []
        if len(names) != ACTION_DIM or len(set(names)) != ACTION_DIM:
            raise ValueError("manifest policy_joint_order must contain 21 unique joints.")
        return tuple(str(name) for name in names)

    @property
    def env_policy_config(self) -> dict[str, float]:
        """Model-intrinsic action/history settings for deploy env construction."""
        action = self.manifest.get("action") or {}
        config: dict[str, float] = {}
        for key in ("action_scale", "ema_alpha", "warmup_time_s"):
            if key in action:
                config[key] = float(action[key])

        for entry in self.manifest.get("input_field_slices") or []:
            if isinstance(entry, dict) and entry.get("name") == "noisy_joint_angles":
                shape = entry.get("shape")
                if isinstance(shape, list) and shape:
                    config["history_len"] = float(int(shape[0]))
                break
        return config

    def validate_onnx_io(self) -> None:
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1:
            raise ValueError(f"Revo3 policy must expose one input, got {len(inputs)}.")
        if len(outputs) != 1:
            raise ValueError(f"Revo3 policy must expose one output, got {len(outputs)}.")

        inp = inputs[0]
        out = outputs[0]
        if inp.name != INPUT_NAME:
            raise ValueError(f"ONNX input name {inp.name!r} != {INPUT_NAME!r}.")
        if out.name != OUTPUT_NAME:
            raise ValueError(f"ONNX output name {out.name!r} != {OUTPUT_NAME!r}.")
        if int(inp.shape[-1]) != INPUT_DIM:
            raise ValueError(f"ONNX input dim {inp.shape[-1]!r} != {INPUT_DIM}.")
        if int(out.shape[-1]) != ACTION_DIM:
            raise ValueError(f"ONNX output dim {out.shape[-1]!r} != {ACTION_DIM}.")

    def validate_golden_io(self, *, atol: float = 1e-5, rtol: float = 1e-5) -> None:
        golden_path = self.artifact_dir / "golden_io.npz"
        if not golden_path.exists():
            raise FileNotFoundError(f"golden_io.npz not found: {golden_path}")

        data = np.load(golden_path, allow_pickle=False)
        for key in ("policy_input_raw", "action_raw"):
            if key not in data:
                raise ValueError(f"golden_io.npz missing {key!r}.")
        raw = np.asarray(data["policy_input_raw"], dtype=np.float32)
        expected = np.asarray(data["action_raw"], dtype=np.float32)
        if raw.shape != (1, INPUT_DIM):
            raise ValueError(f"golden policy_input_raw shape {raw.shape}, expected (1, 216).")
        if expected.shape != (1, ACTION_DIM):
            raise ValueError(f"golden action_raw shape {expected.shape}, expected (1, 21).")
        if "policy_input_normalized" in data:
            normalized = np.asarray(data["policy_input_normalized"], dtype=np.float32)
            if np.allclose(raw, normalized, atol=atol, rtol=rtol):
                raise ValueError(
                    "golden_io policy_input_raw unexpectedly equals "
                    "policy_input_normalized; refusing ambiguous normalization boundary."
                )

        actual = self(raw)[None, :]
        if not np.allclose(actual, expected, atol=atol, rtol=rtol):
            max_err = float(np.max(np.abs(actual - expected)))
            raise ValueError(
                "policy.onnx(policy_input_raw) does not match golden action_raw "
                f"(max_abs_err={max_err:.6g})."
            )

    def __call__(self, obs_raw: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs_raw, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        if obs.shape != (1, INPUT_DIM):
            raise ValueError(f"raw obs shape {obs.shape}, expected (1, {INPUT_DIM}).")
        if not np.isfinite(obs).all():
            raise ValueError("raw obs contains NaN/Inf.")
        result = self.session.run([self.output_name], {self.input_name: obs})[0]
        action = np.asarray(result, dtype=np.float32).reshape(1, ACTION_DIM)
        if not np.isfinite(action).all():
            raise RuntimeError("policy emitted NaN/Inf action.")
        return action[0].copy()
