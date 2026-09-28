# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
from rsl_rl.modules import HeteroscedasticGaussianDistribution
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


@pytest.mark.parametrize(
  "class_name",
  [
    "SoftplusGaussianDistribution",
    "wuji_mjlab.rl.distribution:SoftplusGaussianDistribution",
  ],
)
def test_build_architecture_detects_legacy_softplus_state_dependent_std(class_name):
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
        "class_name": class_name,
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


def test_build_architecture_detects_native_heteroscedastic_std():
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
        "class_name": "rsl_rl.modules:HeteroscedasticGaussianDistribution",
        "init_std": 0.5,
        "std_range": [0.2, 50.0],
        "std_type": "scalar",
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
    nn.Sequential(
      FakeActor(),
      HeteroscedasticGaussianDistribution(3).as_deterministic_output_module(),
    ),
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
    nn.Sequential(
      actor,
      HeteroscedasticGaussianDistribution(3).as_deterministic_output_module(),
    ),
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
