# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from wuji_mjlab.tasks.reorient.mdp.event_impl.episode import (
  apply_velocity_disturbance,
  reset_object_pose_in_palm,
)
from wuji_mjlab.tasks.reorient.mdp.event_impl.state import get_reorient_event_state
from wuji_mjlab.tasks.reorient.mdp.events import reset_disturbance_caches
from wuji_mjlab.tasks.reorient.mdp.observations import (
  perturbation_direction,
  perturbation_velocity,
)
from wuji_mjlab.tasks.reorient.tests.fakes import make_fake_event_env


def test_reset_disturbance_caches_only_clears_selected_envs():
  env = make_fake_event_env(num_envs=4)
  state = get_reorient_event_state(env)
  state.pert_force_dir = torch.arange(
    12, device=env.device, dtype=torch.float32
  ).reshape(4, 3)
  state.pert_velocity_cache = torch.arange(
    24, device=env.device, dtype=torch.float32
  ).reshape(4, 6)

  reset_disturbance_caches(
    env, torch.tensor([1, 3], dtype=torch.long, device=env.device)
  )

  assert torch.equal(
    state.pert_force_dir[0],  # type: ignore[index]
    torch.tensor([0.0, 1.0, 2.0], device=env.device),
  )
  assert torch.equal(state.pert_force_dir[1], torch.zeros(3, device=env.device))  # type: ignore[index]
  assert torch.equal(
    state.pert_force_dir[2],  # type: ignore[index]
    torch.tensor([6.0, 7.0, 8.0], device=env.device),
  )
  assert torch.equal(state.pert_force_dir[3], torch.zeros(3, device=env.device))  # type: ignore[index]
  assert torch.equal(
    state.pert_velocity_cache[0],  # type: ignore[index]
    torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], device=env.device),
  )
  assert torch.equal(state.pert_velocity_cache[1], torch.zeros(6, device=env.device))  # type: ignore[index]
  assert torch.equal(
    state.pert_velocity_cache[2],  # type: ignore[index]
    torch.tensor([12.0, 13.0, 14.0, 15.0, 16.0, 17.0], device=env.device),
  )
  assert torch.equal(state.pert_velocity_cache[3], torch.zeros(6, device=env.device))  # type: ignore[index]


def test_perturbation_observations_read_shared_event_state():
  env = make_fake_event_env(num_envs=2)
  state = get_reorient_event_state(env)
  state.pert_force_dir[0] = torch.tensor([0.1, 0.2, 0.3], device=env.device)  # type: ignore[index]
  state.pert_velocity_cache[1] = torch.tensor(  # type: ignore[index]
    [1.0, 2.0, 3.0, 0.0, 0.0, 0.0], device=env.device
  )

  direction = perturbation_direction(env)
  velocity = perturbation_velocity(env)

  assert direction is state.pert_force_dir
  assert velocity is state.pert_velocity_cache
  assert torch.equal(direction[0], torch.tensor([0.1, 0.2, 0.3], device=env.device))
  assert torch.equal(
    velocity[1],
    torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], device=env.device),
  )


def test_reset_object_pose_in_palm_scales_pos_noise_by_curriculum():
  env = make_fake_event_env(num_envs=4)
  env_ids = torch.arange(env.num_envs, dtype=torch.long)
  cube_offset = (0.1, 0.2, 0.3)
  object_cfg = SceneEntityCfg("object")
  robot_cfg = SceneEntityCfg("robot", body_names=("body_0",))
  expected_pos = torch.tensor(cube_offset, device=env.device).expand(env.num_envs, 3)

  env.curriculum_manager._curriculum_state["adaptive_episode"] = {"value": 0.0}
  reset_object_pose_in_palm(
    env,
    env_ids,
    cube_offset_in_palm=cube_offset,
    pos_noise=0.003,
    pos_noise_curriculum_term="adaptive_episode",
    asset_cfg=object_cfg,
    robot_cfg=robot_cfg,
  )
  pose, _ = env.scene["object"].last_written_root_pose
  assert torch.allclose(pose[:, :3], expected_pos)

  env.curriculum_manager._curriculum_state["adaptive_episode"] = {"value": 1.0}
  reset_object_pose_in_palm(
    env,
    env_ids,
    cube_offset_in_palm=cube_offset,
    pos_noise=0.003,
    pos_noise_curriculum_term="adaptive_episode",
    asset_cfg=object_cfg,
    robot_cfg=robot_cfg,
  )
  pose, _ = env.scene["object"].last_written_root_pose
  noise = pose[:, :3] - expected_pos

  assert torch.all(noise.abs() <= 0.003 + 1.0e-6)
  assert torch.count_nonzero(noise) > 0


def test_velocity_disturbance_can_add_angular_velocity_cache():
  env = make_fake_event_env(num_envs=3)
  env_ids = torch.arange(env.num_envs, dtype=torch.long)
  env.curriculum_manager._curriculum_state["adaptive_episode"] = {"value": 1.0}
  env.common_step_counter = 1
  env.max_common_steps = 1

  apply_velocity_disturbance(
    env,
    env_ids,
    min_speed=0.0,
    max_speed=0.0,
    min_ang_speed=2.0,
    max_ang_speed=2.0,
    warmup_time_s=0.0,
    warmup_frac=0.0,
    rampup_frac=0.0,
    adaptive_curriculum_term="adaptive_episode",
  )

  state = get_reorient_event_state(env)
  root_vel = env.scene["object"].data.root_link_vel_w
  assert torch.allclose(root_vel[:, :3], torch.zeros_like(root_vel[:, :3]))
  assert torch.allclose(state.pert_velocity_cache[:, :3], root_vel[:, :3])  # type: ignore[index]
  assert torch.allclose(state.pert_velocity_cache[:, 3:], root_vel[:, 3:])  # type: ignore[index]
  assert torch.allclose(
    torch.linalg.norm(root_vel[:, 3:], dim=-1),
    torch.full((env.num_envs,), 2.0, device=env.device),
    atol=1.0e-5,
  )
