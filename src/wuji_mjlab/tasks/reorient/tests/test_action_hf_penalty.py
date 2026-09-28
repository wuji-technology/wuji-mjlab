# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import torch
from wuji_mjlab.tasks.reorient.mdp.rewards import ActionHighFreqPenalty


class _FakeTerm:
  def __init__(self):
    self.processed_action = None


class _FakeActionManager:
  def __init__(self):
    self._term = _FakeTerm()

  def get_term(self, name):
    return self._term


class _FakeEnv:
  def __init__(self, n_envs=3):
    self.num_envs = n_envs
    self.action_manager = _FakeActionManager()

  def set_cmd(self, c):
    self.action_manager._term.processed_action = c


def _make():
  env = _FakeEnv()
  term = ActionHighFreqPenalty(cfg=None, env=env)
  return env, term


def _step(env, term, c):
  env.set_cmd(c)
  return term(env)


def test_constant_velocity_is_free():
  env, term = _make()
  base = torch.tensor([[0.1, 0.2, 0.3]] * 3)
  vel = torch.tensor([[0.05, -0.02, 0.01]] * 3)
  vals = []
  for t in range(6):
    vals.append(_step(env, term, base + vel * t).clone())
  assert torch.allclose(vals[2], torch.zeros(3), atol=1e-6)
  assert torch.allclose(vals[3], torch.zeros(3), atol=1e-6)
  assert torch.allclose(vals[5], torch.zeros(3), atol=1e-6)


def test_jitter_is_penalised():
  env, term = _make()
  seq = [
    torch.zeros(3, 3),
    torch.ones(3, 3),
    torch.zeros(3, 3),
    torch.ones(3, 3),
  ]
  vals = [_step(env, term, c).clone() for c in seq]
  assert torch.allclose(vals[2], torch.full((3,), 12.0), atol=1e-5)
  assert (vals[3] > 0).all()


def test_first_two_steps_masked():
  env, term = _make()
  vals = [
    _step(env, term, torch.full((3, 3), float(v))).clone() for v in (5.0, 9.0, 9.0)
  ]
  assert torch.allclose(vals[2], torch.full((3,), 48.0), atol=1e-5)


def test_reset_remasks():
  env, term = _make()
  for v in (0.0, 1.0, 2.0):
    _step(env, term, torch.full((3, 3), v))
  term.reset(torch.tensor([0, 1]))
  out = _step(env, term, torch.full((3, 3), 5.0))
  assert torch.allclose(out[:2], torch.zeros(2))
