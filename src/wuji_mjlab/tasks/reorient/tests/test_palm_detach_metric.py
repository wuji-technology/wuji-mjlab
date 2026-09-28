# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from wuji_mjlab.tasks.reorient.mdp.metrics import palm_detach_frequency
from wuji_mjlab.tasks.reorient.mdp.rewards import palm_detach_reward


def _fake_env(palm_found: torch.Tensor, distal_found: torch.Tensor) -> SimpleNamespace:
  scene = {
    "palm_object_found": SimpleNamespace(data=SimpleNamespace(found=palm_found)),
    "distal_finger_object_found": SimpleNamespace(
      data=SimpleNamespace(found=distal_found)
    ),
  }
  return SimpleNamespace(scene=scene, num_envs=palm_found.shape[0], device="cpu")


def test_palm_detach_frequency_none_guard():
  env = _fake_env(torch.zeros(2, 1), torch.zeros(2, 1))
  env.scene["palm_object_found"].data.found = None
  out = palm_detach_frequency(env)
  assert torch.equal(out, torch.zeros(2))


@pytest.mark.parametrize("func", [palm_detach_reward, palm_detach_frequency])
def test_palm_detach_uses_every_primary_in_both_groups(func):
  palm = torch.zeros(5, 11)
  distal = torch.zeros(5, 15)
  distal[0, 14] = 1
  palm[1, 10] = 1
  distal[1, 0] = 1
  palm[2, 0] = 1
  distal[2, 14] = 1
  distal[4, 0] = 1
  assert func(_fake_env(palm, distal)).tolist() == [1.0, 0.0, 0.0, 0.0, 1.0]
