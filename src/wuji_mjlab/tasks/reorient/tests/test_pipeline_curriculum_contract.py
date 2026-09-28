# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from wuji_mjlab.tasks.reorient.reorient_env_cfg import make_reorient_env_cfg
from wuji_mjlab.tasks.reorient.tests.fakes import make_command_manager


def _train_cfg():
  return make_reorient_env_cfg()


class TestCurriculumDataSource:
  def test_success_curriculum_reads_from_command_metrics(self):
    from wuji_mjlab.tasks.reorient.mdp.curriculums import (
      reorient_success_curriculum,
    )

    num_envs = 4
    env = _make_curriculum_env(
      num_envs=num_envs,
      goal_reach_count_metric=torch.full((num_envs,), 5.0),
    )
    env_ids = torch.arange(num_envs)

    out = reorient_success_curriculum(
      env, env_ids, count_threshold=3, delta_per_loop=0.1
    )

    assert out["success_env_frac"] == pytest.approx(1.0), (
      "All envs above threshold should yield success_env_frac == 1.0"
    )
    assert out["delta"] > 0, "When all envs are above threshold, delta must be positive"

  def test_success_curriculum_ignores_stale_env_attribute(self):
    from wuji_mjlab.tasks.reorient.mdp.curriculums import (
      reorient_success_curriculum,
    )

    num_envs = 4
    env = _make_curriculum_env(
      num_envs=num_envs,
      goal_reach_count_metric=torch.zeros(num_envs),
    )
    env.goal_reach_count = torch.full((num_envs,), 999.0)
    env_ids = torch.arange(num_envs)

    out = reorient_success_curriculum(
      env, env_ids, count_threshold=3, delta_per_loop=0.1
    )

    assert out["success_env_frac"] == 0.0
    assert out["delta"] < 0, (
      "Delta must be negative (reads from metric, ignores stale env attr)"
    )

  def test_success_curriculum_config_references_command_name(self):
    cfg = _train_cfg()
    cur = cfg.curriculum["success_curriculum"]
    assert cur.params["command_name"] == "reorient_command", (
      f"success_curriculum must reference 'reorient_command', "
      f"got '{cur.params['command_name']}'"
    )

  def test_adaptive_episode_curriculum_responds_to_episode_length_buf(self):
    from wuji_mjlab.tasks.reorient.mdp.curriculums import (
      adaptive_episode_curriculum,
    )

    num_envs = 4
    short_env = _make_curriculum_env(num_envs=num_envs)
    short_env.episode_length_buf = torch.full((num_envs,), 100, dtype=torch.long)

    long_env = _make_curriculum_env(num_envs=num_envs)
    long_env.episode_length_buf = torch.full((num_envs,), 900, dtype=torch.long)

    env_ids = torch.arange(num_envs)
    short_out = adaptive_episode_curriculum(short_env, env_ids, target_steps=800)
    long_out = adaptive_episode_curriculum(long_env, env_ids, target_steps=800)

    assert long_out["mean_episode_steps"] > short_out["mean_episode_steps"]
    assert long_out["reached_target_frac"] == 1.0
    assert short_out["reached_target_frac"] == 0.0
    assert long_out["value"] > short_out["value"]


def _make_curriculum_env(
  num_envs: int = 4,
  goal_reach_count_metric: torch.Tensor | None = None,
) -> SimpleNamespace:
  if goal_reach_count_metric is None:
    goal_reach_count_metric = torch.zeros(num_envs)
  command = SimpleNamespace(metrics={"goal_reach_count": goal_reach_count_metric})
  return SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
    termination_manager=SimpleNamespace(
      terminated=torch.zeros(num_envs, dtype=torch.bool)
    ),
    command_manager=make_command_manager(command),
  )
