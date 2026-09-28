# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Check that exported control metadata matches training and deployment."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from wuji_mjlab.rl.onnx_export import _build_config, _load_yaml


def test_pen_export_matches_training_and_deployment(tmp_path):
  import wuji_mjlab.tasks  # noqa: F401
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.utils.os import dump_yaml

  env = load_env_cfg("WujiHand2_PenSpin")
  path = tmp_path / "env.yaml"
  dump_yaml(path, asdict(env))
  exported = _build_config(str(tmp_path), _load_yaml(str(path)))
  root = Path(__file__).resolve().parents[5]
  contract = json.loads(
    (root / "deploy/pen_spin/config/pen_spin_policy.json").read_text()
  )
  assert exported["ctrl_dt"] == pytest.approx(env.decimation * env.sim.mujoco.timestep)
  assert exported["control_mode"] == "reference_residual"
  for key in ("ctrl_dt", "action_scale", "action_residual_clip", "action_ema_alpha"):
    assert exported[key] == pytest.approx(contract[key])


def test_legacy_timestep_still_supported():
  config = _build_config(".", {"decimation": 5, "sim": {"timestep": 0.002}})
  assert config["ctrl_dt"] == pytest.approx(0.01)


def test_reorient_export_uses_actual_physics_timestep(tmp_path):
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.utils.os import dump_yaml
  from wuji_mjlab.tasks.reorient.tooling.onnx_export_core import _build_config

  env = load_env_cfg("WujiHand2_Reorient_50Hz")
  env.sim.mujoco.timestep = 0.002
  path = tmp_path / "env.yaml"
  dump_yaml(path, asdict(env))
  config = _build_config(str(tmp_path), _load_yaml(str(path)))
  assert config["ctrl_dt"] == pytest.approx(env.decimation * 0.002)
