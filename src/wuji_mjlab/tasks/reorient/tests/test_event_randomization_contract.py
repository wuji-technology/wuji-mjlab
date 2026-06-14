# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import torch
from wuji_mjlab.tasks.reorient.mdp.event_impl.curriculum import (
  apply_friction_curriculum,
)
from wuji_mjlab.tasks.reorient.mdp.event_impl.randomization import (
  randomize_field,
  randomize_object_friction_scale,
)
from wuji_mjlab.tasks.reorient.tests.fakes import make_fake_event_env


def test_randomize_field_is_noop_for_empty_env_ids():
  env = make_fake_event_env(num_envs=2)
  snapshot = env.sim.model.body_mass.clone()

  randomize_field(
    env,
    env_ids=torch.tensor([], dtype=torch.long),
    field="body_mass",
    ranges=(0.5, 1.5),
    operation="scale",
  )

  assert torch.equal(env.sim.model.body_mass, snapshot)


def test_apply_friction_curriculum_reuses_startup_snapshot():
  env = make_fake_event_env(num_envs=2)
  env.curriculum_manager._curriculum_state["friction_curriculum"] = {"value": 0.5}
  env_ids = torch.tensor([0, 1], dtype=torch.long)

  apply_friction_curriculum(env, env_ids)
  initial_robot_friction = env.sim.model.geom_friction[:, :3, 0].clone()

  env.sim.model.geom_friction[:, :, 0] = 7.0
  apply_friction_curriculum(env, env_ids)

  assert torch.equal(env.sim.model.geom_friction[:, :3, 0], initial_robot_friction)


def test_randomize_object_friction_scale_respects_curriculum_gate():
  env = make_fake_event_env(num_envs=3)
  env_ids = torch.arange(env.num_envs, dtype=torch.long)
  object_geom_ids = env.scene["object"].indexing.geom_ids.long()

  env.sim.model.geom_friction[:, object_geom_ids, 0] = 5.0
  env.curriculum_manager._curriculum_state["adaptive_episode"] = {"value": 0.0}
  randomize_object_friction_scale(env, env_ids)

  assert torch.allclose(
    env.sim.model.geom_friction[:, object_geom_ids, 0],
    torch.ones((env.num_envs, object_geom_ids.numel()), device=env.device),
  )

  env.curriculum_manager._curriculum_state["adaptive_episode"] = {"value": 1.0}
  randomize_object_friction_scale(env, env_ids)
  sampled = env.sim.model.geom_friction[:, object_geom_ids, 0]

  assert torch.all(sampled >= 0.7)
  assert torch.all(sampled <= 1.3)
