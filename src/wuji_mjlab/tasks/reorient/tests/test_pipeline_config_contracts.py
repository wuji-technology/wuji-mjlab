# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch
from wuji_mjlab.tasks.reorient.config.wuji_hand.env_cfgs import (
  wuji_hand_reorient_env_cfg,
)
from wuji_mjlab.tasks.reorient.config.wuji_hand2 import _reorient_hand2_50hz_env_cfg
from wuji_mjlab.tasks.reorient.config.wuji_hand2.env_cfgs import (
  wuji_hand2_reorient_env_cfg,
)
from wuji_mjlab.tasks.reorient.tests.fakes import make_command_manager


class _FakeEntity:
  def __init__(self, site_names, site_pose_w):
    self._site_names = list(site_names)
    self.data = SimpleNamespace(site_pose_w=site_pose_w)

  def find_sites(self, pattern, preserve_order: bool = False):
    patterns = pattern if isinstance(pattern, (tuple, list)) else (pattern,)
    regexes = [re.compile(p) for p in patterns]
    ids = [
      i for i, n in enumerate(self._site_names) if any(r.fullmatch(n) for r in regexes)
    ]
    return ids, [self._site_names[i] for i in ids]


def _identity_quat(num_envs: int) -> torch.Tensor:
  q = torch.zeros(num_envs, 4)
  q[:, 0] = 1.0
  return q


class _FakeScene(dict):
  def __init__(self, items, device: str = "cpu") -> None:
    super().__init__(items)
    self.device = device


def _make_obs_env(num_envs: int) -> SimpleNamespace:
  site_pose_w = torch.zeros(num_envs, 1, 7)
  site_pose_w[:, :, 3] = 1.0
  robot = _FakeEntity(
    site_names=["robot_wrist_tag"],
    site_pose_w=site_pose_w,
  )

  obj = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.zeros(num_envs, 3),
      root_link_quat_w=_identity_quat(num_envs),
    )
  )

  command = SimpleNamespace(goal_quat=_identity_quat(num_envs))
  scene = _FakeScene({"robot": robot, "object": obj})
  return SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    scene=scene,
    command_manager=make_command_manager(command),
  )


@pytest.mark.parametrize(
  "factory",
  [
    wuji_hand_reorient_env_cfg,
    wuji_hand2_reorient_env_cfg,
    _reorient_hand2_50hz_env_cfg,
  ],
)
@pytest.mark.parametrize("term_name", ["cube_pos_in_tag", "cube_ori_error"])
def test_play_observations_have_no_internal_injection(factory, term_name):
  env = _make_obs_env(num_envs=4096)
  clean = (
    torch.zeros(4096, 3)
    if term_name == "cube_pos_in_tag"
    else torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]).repeat(4096, 1)
  )
  with torch.random.fork_rng():
    torch.manual_seed(20260927)
    play_term = factory(play=True).observations["policy"].terms[term_name]
    assert torch.equal(play_term.func(env, **play_term.params), clean)
    train_term = factory().observations["policy"].terms[term_name]
    assert torch.any(train_term.func(env, **train_term.params) != clean)
