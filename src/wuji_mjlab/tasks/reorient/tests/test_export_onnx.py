# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import onnxruntime as ort
import torch
import torch.nn as nn
from rsl_rl.modules.distribution import _MeanSliceDeterministicOutput
from wuji_mjlab.tasks.reorient.tooling import onnx_export_core as export_onnx


def test_build_architecture_supports_new_rsl_rl_actor_state_dict():
  state_dict = {
    "obs_normalizer._mean": torch.zeros(69),
    "obs_normalizer._var": torch.ones(69),
    "mlp.0.weight": torch.zeros(512, 69),
    "mlp.0.bias": torch.zeros(512),
    "mlp.2.weight": torch.zeros(256, 512),
    "mlp.2.bias": torch.zeros(256),
    "mlp.4.weight": torch.zeros(20, 256),
    "mlp.4.bias": torch.zeros(20),
    "distribution.std_param": torch.ones(20) * 0.1,
  }
  agent_cfg = {
    "actor": {
      "hidden_dims": [512, 256],
      "activation": "elu",
      "distribution_cfg": {
        "std_type": "scalar",
      },
    }
  }

  arch = export_onnx._build_architecture(agent_cfg, state_dict)

  assert arch == {
    "num_actor_obs": 69,
    "hidden_dims": [512, 256],
    "num_actions": 20,
    "actor_output_dim": 20,
    "activation": "elu",
    "state_dependent_std": False,
  }


def test_resolve_export_state_dict_supports_actor_state_dict_checkpoint():
  actor_state_dict = {
    "obs_normalizer._mean": torch.zeros(3),
    "mlp.0.weight": torch.zeros(4, 3),
    "mlp.0.bias": torch.zeros(4),
    "mlp.2.weight": torch.zeros(2, 4),
    "mlp.2.bias": torch.zeros(2),
  }
  checkpoint = {
    "actor_state_dict": actor_state_dict,
  }

  resolved = export_onnx._resolve_export_state_dict(checkpoint)

  assert resolved is actor_state_dict


def test_build_architecture_detects_softplus_state_dependent_std():
  state_dict = {
    "obs_normalizer._mean": torch.zeros(69),
    "obs_normalizer._var": torch.ones(69),
    "mlp.0.weight": torch.zeros(512, 69),
    "mlp.0.bias": torch.zeros(512),
    "mlp.2.weight": torch.zeros(256, 512),
    "mlp.2.bias": torch.zeros(256),
    "mlp.4.weight": torch.zeros(128, 256),
    "mlp.4.bias": torch.zeros(128),
    "mlp.6.weight": torch.zeros(40, 128),
    "mlp.6.bias": torch.zeros(40),
  }
  agent_cfg = {
    "actor": {
      "hidden_dims": [512, 256, 128],
      "activation": "elu",
      "distribution_cfg": {
        "class_name": "SoftplusGaussianDistribution",
        "init_std": 0.5,
        "min_std": 0.1,
      },
    }
  }

  arch = export_onnx._build_architecture(agent_cfg, state_dict)

  assert arch == {
    "num_actor_obs": 69,
    "hidden_dims": [512, 256, 128],
    "num_actions": 20,
    "actor_output_dim": [2, 20],
    "activation": "elu",
    "state_dependent_std": True,
  }


def test_standalone_exporter_returns_mean_slice_for_state_dependent_std():
  class FakeActor(nn.Module):
    def forward(self, x):
      batch = x.shape[0]
      out = torch.zeros(batch, 2, 3)
      out[:, 0, :] = torch.tensor([1.0, 2.0, 3.0])
      out[:, 1, :] = torch.tensor([4.0, 5.0, 6.0])
      return out

  exporter = export_onnx._StandaloneExporter(
    nn.Identity(),
    nn.Sequential(FakeActor(), _MeanSliceDeterministicOutput()),
  )

  result = exporter(torch.zeros(2, 5))

  assert result.shape == (2, 3)
  assert torch.allclose(result, torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))


def test_standalone_exporter_can_export_state_dependent_std_to_onnx(tmp_path):
  actor = export_onnx.MLP(
    input_dim=5,
    output_dim=[2, 3],
    hidden_dims=[4],
    activation="elu",
  )
  exporter = export_onnx._StandaloneExporter(
    nn.Identity(),
    nn.Sequential(actor, _MeanSliceDeterministicOutput()),
  )
  onnx_path = tmp_path / "hetero_policy.onnx"

  torch.onnx.export(
    exporter,
    torch.zeros(1, 5),
    onnx_path,
    export_params=True,
    opset_version=export_onnx._ONNX_OPSET_VERSION,
    input_names=["obs"],
    output_names=["actions"],
    dynamic_axes={},
    dynamo=False,
  )

  session = ort.InferenceSession(str(onnx_path))
  assert session.get_outputs()[0].shape == [1, 3]


def test_build_config_emits_revo3_scene_metadata():
  env_cfg = {
    "task_id": "Revo3RightHand_Reorient",
    "actions": {
      "joint_pos": {
        "action_scale": 0.5,
        "ema_alpha": 0.5,
        "warmup_time_s": 0.4,
      }
    },
    "observations": {
      "policy": {
        "terms": {
          "joint": {"history_length": 3},
        }
      }
    },
    "sim": {"timestep": 0.01},
    "decimation": 5,
  }

  config = export_onnx._build_config(".", env_cfg)

  assert config["task_id"] == "Revo3RightHand_Reorient"
  assert config["palm_body_name"] == "right_palm"
  assert len(config["joint_names"]) == 21
  assert config["joint_names"][0] == "right_thumb_CMP_joint"
