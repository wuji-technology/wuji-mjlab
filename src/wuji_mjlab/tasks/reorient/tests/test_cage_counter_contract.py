# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from types import SimpleNamespace

import torch
from wuji_mjlab.tasks.reorient.mdp.cage import (
  _compute_cage_bounds_in_palm,
  cage_drop,
)


def test_cage_bounds_apply_extra_margin_to_configured_up_axis():
  hand_in_palm = torch.tensor(
    [[[-1.0, -2.0, -3.0], [2.0, 4.0, 6.0]]], dtype=torch.float64
  )

  lo, hi = _compute_cage_bounds_in_palm(
    hand_in_palm, margin=0.1, up_margin=0.3, up_axis=0
  )

  assert torch.equal(lo, torch.tensor([[-1.1, -2.1, -3.1]], dtype=torch.float64))
  assert torch.equal(hi, torch.tensor([[2.3, 4.1, 6.1]], dtype=torch.float64))


def test_cage_drop_uses_shared_counter_behavior():
  env = SimpleNamespace(
    _cage_penalty_counter=torch.tensor([9.0, 10.0]),
    num_envs=2,
    device="cpu",
  )

  result = cage_drop(env, max_outside_steps=10)

  assert torch.equal(result, torch.tensor([False, True]))
