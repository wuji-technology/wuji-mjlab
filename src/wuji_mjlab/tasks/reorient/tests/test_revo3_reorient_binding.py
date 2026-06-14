# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import re

import mujoco
import numpy as np
import wuji_mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg
from wuji_mjlab.assets.robots.revo3_hand.revo3_hand_cfg import get_revo3_hand_cfg
from wuji_mjlab.tasks.reorient.config.revo3_hand.env_cfgs import (
  REVO3_CUBE_OFFSET_IN_PALM,
  REVO3_ROBOT_ROOT_ROT,
  revo3_right_hand_reorient_repose_reward_env_cfg,
  revo3_right_hand_reorient_repose_reward_finetune_env_cfg,
)
from wuji_mjlab.tasks.reorient.robot_bindings import REVO3_RIGHT_HAND_BINDING
from wuji_mjlab.tasks.reorient.tooling.scene_builder import build_reorient_scene


def _matches(patterns: tuple[str, ...], names: list[str]) -> bool:
  return all(any(re.fullmatch(pattern, name) for name in names) for pattern in patterns)


def test_revo3_mjcf_compiles_with_actuators_and_tip_sites():
  model = get_revo3_hand_cfg().spec_fn().compile()

  assert model.nq == 21
  assert model.nv == 21
  assert model.nu == 21
  assert model.nsite == 5
  assert [model.site(i).name for i in range(model.nsite)] == [
    "right_thumb_tip",
    "right_index_tip",
    "right_middle_tip",
    "right_ring_tip",
    "right_little_tip",
  ]


def test_revo3_tasks_are_registered_and_play_overrides_match_wuji_contract():
  tasks = set(list_tasks())
  assert "Revo3RightHand_Reorient" in tasks
  assert "Revo3RightHand_Reorient_Light" in tasks
  assert "Revo3RightHand_Reorient_ReposeReward" in tasks
  assert "Revo3RightHand_Reorient_ReposeReward_FineTune" in tasks

  cfg = load_env_cfg("Revo3RightHand_Reorient", play=True)
  assert cfg.scene.num_envs == 4
  assert cfg.viewer.body_name == "right_palm"
  assert cfg.observations["policy"].enable_corruption is False
  assert cfg.curriculum == {}
  assert "robot_geom_size" not in cfg.events
  assert cfg.commands["reorient_command"].palm_body_pattern == "right_palm"


def _assert_same_non_reward_contract(candidate_cfg, reference_cfg) -> None:
  assert set(candidate_cfg.observations) == set(reference_cfg.observations)
  for group_name, group_cfg in reference_cfg.observations.items():
    assert set(candidate_cfg.observations[group_name].terms) == set(group_cfg.terms)
    assert (
      candidate_cfg.observations[group_name].enable_corruption
      == group_cfg.enable_corruption
    )
  assert set(candidate_cfg.actions) == set(reference_cfg.actions)
  assert set(candidate_cfg.commands) == set(reference_cfg.commands)
  assert set(candidate_cfg.events) == set(reference_cfg.events)
  assert set(candidate_cfg.terminations) == set(reference_cfg.terminations)
  assert candidate_cfg.curriculum == reference_cfg.curriculum
  assert candidate_cfg.viewer.body_name == reference_cfg.viewer.body_name
  assert candidate_cfg.scene.num_envs == reference_cfg.scene.num_envs


def test_revo3_repose_reward_task_only_replaces_rewards():
  base_cfg = load_env_cfg("Revo3RightHand_Reorient", play=True)
  repose_cfg = load_env_cfg("Revo3RightHand_Reorient_ReposeReward", play=True)

  assert set(base_cfg.rewards) == {
    "orientation_alignment",
    "hand_pose",
    "action_rate",
    "torque",
    "tip_slide",
    "cage_escape",
    "finger_collision",
    "hold_escalation",
    "palm_detach",
  }
  assert set(repose_cfg.rewards) == {
    "repose_position_distance",
    "repose_inverse_orientation",
    "repose_action_l2",
    "cage_escape",
    "hold_escalation",
  }

  _assert_same_non_reward_contract(repose_cfg, base_cfg)

  assert repose_cfg.rewards["repose_position_distance"].weight == -200.0
  assert repose_cfg.rewards["repose_inverse_orientation"].weight == 20.0
  assert repose_cfg.rewards["repose_action_l2"].weight == -0.004
  assert repose_cfg.rewards["cage_escape"].weight == -1.0e-6
  assert repose_cfg.rewards["hold_escalation"].weight == 11.4
  assert (
    repose_cfg.rewards["hold_escalation"].func
    is base_cfg.rewards["hold_escalation"].func
  )


def test_revo3_repose_reward_finetune_task_adds_conservative_penalties():
  repose_cfg = load_env_cfg("Revo3RightHand_Reorient_ReposeReward", play=True)
  finetune_cfg = load_env_cfg(
    "Revo3RightHand_Reorient_ReposeReward_FineTune", play=True
  )

  assert set(finetune_cfg.rewards) == {
    "repose_position_distance",
    "repose_inverse_orientation",
    "repose_action_l2",
    "hand_pose",
    "action_rate",
    "torque",
    "tip_slide",
    "cage_escape",
    "finger_collision",
    "hold_escalation",
  }
  assert "palm_detach" not in finetune_cfg.rewards
  assert "orientation_alignment" not in finetune_cfg.rewards

  _assert_same_non_reward_contract(finetune_cfg, repose_cfg)

  assert finetune_cfg.rewards["repose_position_distance"].weight == -200.0
  assert finetune_cfg.rewards["repose_inverse_orientation"].weight == 20.0
  assert finetune_cfg.rewards["repose_action_l2"].weight == -0.004
  assert finetune_cfg.rewards["hand_pose"].weight == -0.2
  assert finetune_cfg.rewards["action_rate"].weight == -0.02
  assert finetune_cfg.rewards["torque"].weight == -24.0
  assert finetune_cfg.rewards["tip_slide"].weight == -0.3
  assert finetune_cfg.rewards["cage_escape"].weight == -500.0
  assert finetune_cfg.rewards["finger_collision"].weight == -1.0
  assert finetune_cfg.rewards["hold_escalation"].weight == 11.4
  assert (
    finetune_cfg.rewards["hold_escalation"].func
    is repose_cfg.rewards["hold_escalation"].func
  )


def test_revo3_finetune_train_enables_robust_randomization_only():
  repose_cfg = revo3_right_hand_reorient_repose_reward_env_cfg(play=False, num_envs=8)
  finetune_cfg = revo3_right_hand_reorient_repose_reward_finetune_env_cfg(
    play=False, num_envs=8
  )
  finetune_play_cfg = revo3_right_hand_reorient_repose_reward_finetune_env_cfg(
    play=True
  )
  finetune_no_robust_cfg = revo3_right_hand_reorient_repose_reward_finetune_env_cfg(
    play=False,
    num_envs=8,
    robust_randomization=None,
  )

  assert "object_friction" not in repose_cfg.events
  assert "object_friction" in finetune_cfg.events
  assert "object_friction" not in finetune_play_cfg.events
  assert "object_friction" not in finetune_no_robust_cfg.events

  reset_params = finetune_cfg.events["reset_object_pose"].params
  assert reset_params["pos_noise"] == 0.003
  assert reset_params["pos_noise_curriculum_term"] == "adaptive_episode"
  assert repose_cfg.events["reset_object_pose"].params["pos_noise"] == 0.0
  assert finetune_play_cfg.events["reset_object_pose"].params["pos_noise"] == 0.0

  disturbance_params = finetune_cfg.events["object_disturbance_force"].params
  assert disturbance_params["min_ang_speed"] == 0.0
  assert disturbance_params["max_ang_speed"] == 2.0
  assert "max_ang_speed" not in repose_cfg.events["object_disturbance_force"].params

  action_cfg = finetune_cfg.actions["joint_pos"]
  assert action_cfg.action_hold_max_prob == 0.10
  assert action_cfg.action_hold_curriculum_term == "adaptive_episode"
  assert repose_cfg.actions["joint_pos"].action_hold_max_prob == 0.0
  assert finetune_play_cfg.actions["joint_pos"].action_hold_max_prob == 0.0


def test_revo3_policy_observation_dimension_contract_from_binding():
  cfg = load_env_cfg("Revo3RightHand_Reorient", play=True)
  history = max(term.history_length or 0 for term in cfg.observations["policy"].terms.values())
  action_dim = len(REVO3_RIGHT_HAND_BINDING.joint_names)

  assert history == 3
  assert action_dim == 21
  assert history * (action_dim + action_dim + 3 + 6 + action_dim) == 216


def test_revo3_scene_is_palm_up_and_cube_keeps_palm_local_offset():
  scene = build_reorient_scene(task_id="Revo3RightHand_Reorient")
  mujoco.mj_forward(scene.model, scene.data)

  base_body_id = scene.model.body("robot/right_hand_base_link").id
  base_quat = scene.data.xquat[base_body_id]
  assert np.allclose(
    base_quat,
    np.array(REVO3_ROBOT_ROOT_ROT, dtype=np.float64),
    atol=1e-6,
  )

  palm_mat = scene.data.xmat[scene.palm_body_id].reshape(3, 3)
  palm_up_axis_world = palm_mat[:, 0]
  assert np.allclose(palm_up_axis_world, np.array([0.0, 0.0, 1.0]), atol=1e-6)

  cube_offset_in_palm = palm_mat.T @ (scene.cube_pos - scene.palm_pos)

  assert np.allclose(
    cube_offset_in_palm,
    np.array(REVO3_CUBE_OFFSET_IN_PALM, dtype=np.float64),
    atol=1e-6,
  )


def test_revo3_binding_patterns_match_asset_names():
  model = get_revo3_hand_cfg().spec_fn().compile()
  body_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
    for i in range(model.nbody)
  ]
  geom_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    for i in range(model.ngeom)
  ]
  site_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or ""
    for i in range(model.nsite)
  ]

  binding = REVO3_RIGHT_HAND_BINDING
  assert _matches(binding.palm_body_names, body_names)
  assert _matches(binding.tip_body_names, body_names)
  assert _matches(binding.palm_object_found_bodies, body_names)
  assert _matches(binding.distal_finger_object_found_bodies, body_names)
  assert _matches(binding.cage_body_names, body_names)
  assert _matches(binding.tip_site_names, site_names)
  assert _matches(binding.tip_collision_geoms, geom_names)
  assert _matches(binding.finger_collision_primary_geoms, geom_names)
  assert _matches(binding.dr_robot_geoms, geom_names)
  assert _matches(binding.contact_params_palm_thumb_geoms, geom_names)
  assert _matches(binding.contact_params_fingers_geoms, geom_names)
