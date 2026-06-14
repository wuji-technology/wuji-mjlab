# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

from types import SimpleNamespace

import torch
from wuji_mjlab.tasks.reorient.mdp.actions import (
  JointPositionOffsetEMAAction,
  JointPositionOffsetEMAActionCfg,
)


def _make_action_term(
  *,
  num_envs: int = 2,
  action_dim: int = 3,
  hold_prob: float = 0.0,
) -> JointPositionOffsetEMAAction:
  action = JointPositionOffsetEMAAction.__new__(JointPositionOffsetEMAAction)
  action.cfg = JointPositionOffsetEMAActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    action_scale=1.0,
    ema_alpha=1.0,
    warmup_time_s=0.0,
    action_hold_max_prob=hold_prob,
  )
  action._env = SimpleNamespace(
    episode_length_buf=torch.ones(num_envs, dtype=torch.long),
    step_dt=0.02,
    curriculum_manager=SimpleNamespace(_curriculum_state={}),
  )
  action._raw_actions = torch.zeros(num_envs, action_dim)
  action._processed_actions = torch.zeros_like(action._raw_actions)
  action._action_scale = 1.0
  action._ema_alpha = 1.0
  action._warmup_time_s = 0.0
  action._action_hold_max_prob = hold_prob
  action._action_hold_curriculum_term = ""
  action._default_joint_pos = torch.zeros_like(action._raw_actions)
  action._lower_limits = torch.full_like(action._raw_actions, -10.0)
  action._upper_limits = torch.full_like(action._raw_actions, 10.0)
  action._prev_target = torch.zeros_like(action._raw_actions)
  action._prev_effective_action = torch.zeros_like(action._raw_actions)
  return action


def test_action_hold_defaults_to_old_behavior():
  action = _make_action_term(hold_prob=0.0)
  raw = torch.tensor([[0.2, -0.3, 0.4], [-0.5, 0.6, -0.7]])

  action.process_actions(raw)

  assert torch.equal(action.raw_action, raw)
  assert torch.equal(action.processed_action, raw)


def test_action_hold_reuses_previous_effective_action_and_resets_cache():
  action = _make_action_term(hold_prob=0.0)
  first = torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
  second = -first

  action.process_actions(first)
  action._action_hold_max_prob = 1.0
  action.process_actions(second)

  assert torch.equal(action.raw_action, second)
  assert torch.equal(action.processed_action, first)

  action.reset(torch.tensor([0], dtype=torch.long))
  action.process_actions(second)

  assert torch.equal(action.processed_action[0], torch.zeros(3))
  assert torch.equal(action.processed_action[1], first[1])
