# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
from wuji_mjlab.assets.robots.wuji_hand2.wuji_hand2_cfg import wuji_hand2_xml_path
from wuji_mjlab.tasks.reorient.tooling import onnx_export_core as export_onnx
from wuji_mjlab.tasks.reorient.tooling import stamp_policy_metadata as stamp_tool
from wuji_reorient_deploy.constants import joint_names, load_constants
from wuji_reorient_deploy.onnx_policy import ONNXPolicy


@pytest.fixture(scope="session")
def synthetic_policy(tmp_path_factory) -> Path:
  source = tmp_path_factory.mktemp("synthetic-policy")
  onnx_path = source / "policy.onnx"
  with torch.random.fork_rng(devices=[]):
    actor = nn.Linear(207, 20)
  with torch.no_grad():
    actor.weight.zero_()
    actor.weight[torch.arange(20), torch.arange(20)] = 0.5
    actor.bias.copy_(torch.arange(20, dtype=torch.float32) / 10)
  exporter = export_onnx._StandaloneExporter(nn.Identity(), actor)
  torch.onnx.export(
    exporter,
    torch.zeros(1, 207),
    onnx_path,
    export_params=True,
    opset_version=export_onnx._ONNX_OPSET_VERSION,
    input_names=["obs"],
    output_names=["actions"],
    dynamic_axes={},
    dynamo=False,
  )
  (source / "config.json").write_text(
    json.dumps({"control_mode": "absolute", "history_len": 3})
  )
  return onnx_path


def _copy_legacy_policy(tmp_path: Path, synthetic_policy: Path) -> Path:
  onnx_path = tmp_path / synthetic_policy.name
  shutil.copy2(synthetic_policy, onnx_path)
  shutil.copy2(synthetic_policy.parent / "config.json", tmp_path / "config.json")
  return onnx_path


def _stamp_gen2(
  tmp_path: Path, synthetic_policy: Path, identity: dict | None = None
) -> Path:
  onnx_path = _copy_legacy_policy(tmp_path, synthetic_policy)
  if identity is None:
    identity = stamp_tool._identity_from_constants(2, "right")
  stamp_tool.stamp_policy_metadata(onnx_path, identity)
  return onnx_path


def test_legacy_policy_without_identity_is_rejected(tmp_path, synthetic_policy):
  onnx_path = _copy_legacy_policy(tmp_path, synthetic_policy)

  with pytest.raises(ValueError, match="stamp_policy_metadata"):
    ONNXPolicy(str(onnx_path), load_constants(2, "right"))


def test_gen2_identity_loads_and_preserves_inference(tmp_path, synthetic_policy):
  onnx_path = _copy_legacy_policy(tmp_path, synthetic_policy)
  session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  observations = [
    np.linspace(-1, 1, 207, dtype=np.float32),
    np.linspace(2, -2, 207, dtype=np.float32),
  ]
  expected = [
    session.run(["actions"], {"obs": obs.reshape(1, 207)})[0][0] for obs in observations
  ]
  stamp_tool.stamp_policy_metadata(
    onnx_path, stamp_tool._identity_from_constants(2, "right")
  )

  policy = ONNXPolicy(str(onnx_path), load_constants(2, "right"))
  for obs, original in zip(observations, expected, strict=True):
    np.testing.assert_allclose(policy(obs), original, rtol=0, atol=1e-6)


def test_gen2_identity_rejects_gen1_constants_as_generation_mismatch(
  tmp_path, synthetic_policy
):
  onnx_path = _stamp_gen2(tmp_path, synthetic_policy)

  with pytest.raises(ValueError) as exc_info:
    ONNXPolicy(str(onnx_path), load_constants(1, "right"))

  message = str(exc_info.value)
  assert "gen1" in message
  assert "gen2" in message
  assert "generation mismatch" in message.lower()


def test_unknown_default_pose_is_reported_as_constant_drift(tmp_path, synthetic_policy):
  identity = stamp_tool._identity_from_constants(2, "right")
  changed_joint = "right_finger1_joint2"
  changed_index = identity["joint_order"].index(changed_joint)
  identity["default_joint_pos"][changed_index] += 0.1
  onnx_path = _stamp_gen2(tmp_path, synthetic_policy, identity)

  with pytest.raises(ValueError) as exc_info:
    ONNXPolicy(str(onnx_path), load_constants(2, "right"))

  message = str(exc_info.value)
  assert "constant drift" in message.lower()
  assert "generation mismatch" not in message.lower()
  assert changed_joint in message


def test_default_pose_float_representation_noise_is_tolerated(
  tmp_path, synthetic_policy
):
  identity = stamp_tool._identity_from_constants(2, "right")
  identity["default_joint_pos"] = [
    value + 1e-9 for value in identity["default_joint_pos"]
  ]
  onnx_path = _stamp_gen2(tmp_path, synthetic_policy, identity)

  policy = ONNXPolicy(str(onnx_path), load_constants(2, "right"))
  obs = np.linspace(-1, 1, 207, dtype=np.float32)
  expected = policy.session.run(["actions"], {"obs": obs.reshape(1, 207)})[0][0]
  np.testing.assert_allclose(policy(obs), expected, rtol=0, atol=1e-6)


def test_reordered_joints_are_rejected(tmp_path, synthetic_policy):
  identity = stamp_tool._identity_from_constants(2, "right")
  identity["joint_order"][0], identity["joint_order"][1] = (
    identity["joint_order"][1],
    identity["joint_order"][0],
  )
  onnx_path = _stamp_gen2(tmp_path, synthetic_policy, identity)

  with pytest.raises(ValueError, match="joint_order"):
    ONNXPolicy(str(onnx_path), load_constants(2, "right"))


def test_changed_observation_terms_are_rejected(tmp_path, synthetic_policy):
  identity = stamp_tool._identity_from_constants(2, "right")
  identity["obs_term_names"] = identity["obs_term_names"][:-1]
  onnx_path = _stamp_gen2(tmp_path, synthetic_policy, identity)

  with pytest.raises(ValueError, match="obs_term_names"):
    ONNXPolicy(str(onnx_path), load_constants(2, "right"))


def test_onnx_and_sidecar_identity_disagreement_is_rejected(tmp_path, synthetic_policy):
  onnx_path = _stamp_gen2(tmp_path, synthetic_policy)
  config_path = tmp_path / "config.json"
  config = json.loads(config_path.read_text())
  config["default_joint_pos"][0] += 0.1
  config_path.write_text(json.dumps(config, indent=2) + "\n")

  with pytest.raises(ValueError) as exc_info:
    ONNXPolicy(str(onnx_path), load_constants(2, "right"))

  message = str(exc_info.value)
  assert "ONNX metadata" in message
  assert "config.json" in message


def test_exported_identity_matches_deploy_constants(tmp_path, synthetic_policy):
  constants = load_constants(2, "right")
  robot_mjcf = wuji_hand2_xml_path("right")
  env_cfg = {
    "scene": {
      "entities": {
        "robot": {
          "spec_fn": {"state": [None, [str(robot_mjcf)]]},
          "init_state": {
            "joint_pos": dict(
              zip(
                joint_names("right"), constants.default_joint_pos.tolist(), strict=True
              )
            )
          },
        }
      }
    },
    "observations": {
      "policy": {
        "history_length": constants.history_length,
        "terms": {name: {} for name, _ in constants.obs_terms},
      }
    },
    "actions": {"joint_pos": {"action_scale": 0.5}},
  }
  config = export_onnx._build_config(str(tmp_path), env_cfg)
  onnx_path = _copy_legacy_policy(tmp_path, synthetic_policy)
  (tmp_path / "config.json").write_text(json.dumps(config))
  model = onnx.load(onnx_path)
  export_onnx._set_policy_identity_metadata(
    model, export_onnx._policy_identity_from_config(config)
  )
  onnx.save(model, onnx_path)

  policy = ONNXPolicy(str(onnx_path), constants)
  obs = np.linspace(-1, 1, 207, dtype=np.float32)
  np.testing.assert_allclose(
    policy(obs),
    policy.session.run(["actions"], {"obs": obs.reshape(1, 207)})[0][0],
    rtol=0,
    atol=1e-6,
  )


def test_export_identity_unwraps_recorded_nested_hand2_partials(tmp_path):
  constants = load_constants(2, "right")
  robot_mjcf = wuji_hand2_xml_path("right")
  inner = {
    "args": ["wuji_mjlab.assets.robots.wuji_hand2.wuji_hand2_cfg._get_spec"],
    "state": [
      "wuji_mjlab.assets.robots.wuji_hand2.wuji_hand2_cfg._get_spec",
      [list(robot_mjcf.parts)],
      {},
      None,
    ],
  }
  outer = {
    "args": [
      "wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants.get_hand2_reorient_spec"
    ],
    "state": [
      "wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants.get_hand2_reorient_spec",
      [inner],
      {},
      None,
    ],
  }
  env_cfg = {
    "scene": {
      "entities": {
        "robot": {
          "spec_fn": outer,
          "init_state": {
            "joint_pos": dict(
              zip(
                joint_names("right"), constants.default_joint_pos.tolist(), strict=True
              )
            )
          },
        }
      }
    },
    "observations": {
      "policy": {
        "terms": {name: {} for name, _ in constants.obs_terms},
      }
    },
    "actions": {"joint_pos": {"action_scale": 0.5, "ema_alpha": 0.3}},
  }
  config = export_onnx._build_config(str(tmp_path), env_cfg)
  assert config["robot_mjcf"] == str(robot_mjcf)
  assert config["hand_gen"] == 2
  assert config["joint_order"] == list(joint_names("right"))
  np.testing.assert_allclose(config["default_joint_pos"], constants.default_joint_pos)

  inner["state"][1][0] = {"unexpected": "not a path"}
  with pytest.raises(ValueError, match="unexpected.*not a path"):
    export_onnx._build_config(str(tmp_path), env_cfg)


@pytest.mark.parametrize("directory_sidecar", [False, True])
def test_stamp_prefers_policy_specific_sidecar(
  tmp_path, synthetic_policy, directory_sidecar
):
  onnx_path = _copy_legacy_policy(tmp_path, synthetic_policy)
  generic = tmp_path / "config.json"
  specific = Path(f"{onnx_path}.config.json")
  config = json.loads(generic.read_text())
  config["action_scale"] = 0.125
  specific.write_text(json.dumps(config))
  original_generic = generic.read_bytes()
  if not directory_sidecar:
    generic.unlink()
  identity = stamp_tool._identity_from_constants(2, "right")

  _, updated = stamp_tool.stamp_policy_metadata(onnx_path, identity)

  assert updated == specific
  stamped = json.loads(specific.read_text())
  assert all(stamped[field] == value for field, value in identity.items())
  policy = ONNXPolicy(str(onnx_path), load_constants(2, "right"))
  assert policy.config["action_scale"] == 0.125
  if directory_sidecar:
    assert generic.read_bytes() == original_generic
