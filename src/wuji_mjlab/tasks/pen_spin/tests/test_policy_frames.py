# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Physical hand2 wrist coordinates shared by references and simulation."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import combine_frame_transforms, quat_apply, quat_unique
from wuji_mjlab.tasks.pen_spin.mdp import observations as obs
from wuji_mjlab.tasks.pen_spin.mdp.events import apply_reference_state
from wuji_mjlab.tasks.pen_spin.mdp.visualization import _hand_ghost_pose


@pytest.fixture
def state():
  rng = np.random.default_rng(147)
  wrist = rng.normal(size=(8, 7))
  wrist[:, 3:] /= np.linalg.norm(wrist[:, 3:], axis=-1, keepdims=True)
  wrist = torch.tensor(wrist, dtype=torch.float32)
  local = torch.tensor(rng.normal(0, 0.15, (8, 7)), dtype=torch.float32)
  local[:, 3:] /= local[:, 3:].norm(dim=-1, keepdim=True)
  local[:, 3:] = quat_unique(local[:, 3:])
  obj_pos, obj_quat = combine_frame_transforms(
    wrist[:, :3], wrist[:, 3:], local[:, :3], local[:, 3:]
  )
  links = torch.tensor(rng.normal(0, 0.1, (8, 21, 3)), dtype=torch.float32)
  links[:, 0] = 0
  world_links = quat_apply(wrist[:, None, 3:].expand(8, 21, 4), links)
  world_links += wrist[:, None, :3]
  robot = SimpleNamespace(
    body_names=["right_palm_link"] + [f"finger_link_{i}" for i in range(20)],
    data=SimpleNamespace(
      root_link_pos_w=wrist[:, :3],
      root_link_quat_w=wrist[:, 3:],
      body_link_pos_w=world_links,
      body_link_quat_w=wrist[:, None, 3:].expand(8, 21, 4),
    ),
    write_mocap_pose_to_sim=Mock(),
    write_joint_state_to_sim=Mock(),
    find_joints=lambda names, **kwargs: (list(range(20)), names),
    indexing=SimpleNamespace(joint_q_adr=torch.arange(20), mocap_id=0),
  )
  obj = SimpleNamespace(
    data=SimpleNamespace(root_link_pos_w=obj_pos, root_link_quat_w=obj_quat),
    write_root_state_to_sim=Mock(),
  )
  motion = SimpleNamespace(
    axis_points=obs.make_axis_points(),
    cfg=SimpleNamespace(object_position_tanh_scale_m=0.15),
    wrist_pose_w=wrist,
    object_pose=local,
    joint_names=tuple(f"joint_{i}" for i in range(20)),
    joint_pos=torch.zeros(8, 20),
    joint_vel=torch.zeros(8, 20),
    object_linvel_w=torch.zeros(8, 3),
    object_angvel_w=torch.zeros(8, 3),
  )
  env = SimpleNamespace(
    scene={"robot": robot, "object": obj},
    num_envs=8,
    device="cpu",
    cfg=SimpleNamespace(auto_reset=True),
    command_manager=SimpleNamespace(get_term=lambda _: motion),
  )
  return env, motion, local, links, wrist


@pytest.mark.parametrize("select_body", [False, True])
def test_actual_pose_and_bounded_points_match_hand2(state, select_body):
  env, motion, expected, _, _ = state
  cfg = (
    SceneEntityCfg("robot", body_names=("right_palm_link",), body_ids=[0])
    if select_body
    else obs.ROBOT
  )
  actual = obs.object_pose_wrist_local(env, robot_cfg=cfg)
  torch.testing.assert_close(actual[:, :3], expected[:, :3], atol=3e-7, rtol=0)
  torch.testing.assert_close(
    quat_unique(actual[:, 3:]), expected[:, 3:], atol=3e-7, rtol=0
  )
  points = obs.bounded_object_pose_to_points_wrist_local(
    expected, motion.axis_points, position_tanh_scale_m=0.15
  )
  torch.testing.assert_close(
    obs.object_axis_points_wrist_local_bounded(env, robot_cfg=cfg),
    points.flatten(1),
    atol=3e-7,
    rtol=0,
  )


@pytest.mark.parametrize("ids", [slice(None), [3, 0, 7], [4, 5]])
def test_actual_links_use_physical_wrist_origin_in_any_order(state, ids):
  env, _, _, expected, _ = state
  cfg = SceneEntityCfg("robot", body_ids=ids)
  actual = obs.link_pos_wrist_local(env, cfg).reshape(8, -1, 3)
  torch.testing.assert_close(actual, expected[:, ids], atol=4e-7, rtol=0)


def test_camera_noise_lag_and_dropout_follow_hand2_frame(state, monkeypatch):
  env, _, expected, _, _ = state
  cfg = SimpleNamespace(
    params=dict(
      robot_cfg=obs.ROBOT,
      object_cfg=obs.OBJECT,
      command_name="motion",
      bias_position_m=(-0.01, 0.01),
      bias_orientation_rad=(-0.1, 0.1),
      noise_position_m=(-0.005, 0.005),
      noise_orientation_rad=(-0.05, 0.05),
      lag_ticks=(1, 3),
      dropout_enter_prob=0.3,
      dropout_continue_prob=0.8,
      dropout_max_ticks=5,
    )
  )

  def run():
    torch.manual_seed(501)
    camera = obs.ObjectPoseCameraModel(cfg, env)
    outputs = []
    for tick in range(24):
      if tick == 12:
        camera.reset(torch.tensor([0, 3]))
      outputs.append(camera(env))
    return torch.stack(outputs)

  actual = run()
  monkeypatch.setattr(obs, "object_pose_wrist_local", lambda *args: expected.clone())
  torch.testing.assert_close(actual, run(), atol=3e-7, rtol=0)


def test_reset_and_ghost_use_reference_without_conversion(state):
  env, motion, _, _, wrist = state
  ids = torch.tensor([1, 4])
  apply_reference_state(env, ids, motion)
  robot, obj = env.scene["robot"], env.scene["object"]
  actual_wrist = robot.write_mocap_pose_to_sim.call_args.args[0]
  torch.testing.assert_close(actual_wrist, wrist[ids], atol=3e-7, rtol=0)
  root_state = obj.write_root_state_to_sim.call_args.args[0]
  torch.testing.assert_close(root_state[:, :3], obj.data.root_link_pos_w[ids])
  torch.testing.assert_close(
    root_state[:, 3:7], quat_unique(obj.data.root_link_quat_w[ids])
  )
  reference = dict(
    joint_pos=motion.joint_pos,
    wrist_pos=motion.wrist_pose_w[:, :3],
    wrist_quat=motion.wrist_pose_w[:, 3:],
  )
  _, pos, quat = _hand_ghost_pose(
    SimpleNamespace(_env=env), SimpleNamespace(qpos0=np.zeros(20), nmocap=1), reference
  )
  np.testing.assert_allclose(pos[0], wrist[0, :3], atol=3e-7)
  np.testing.assert_allclose(quat[0], wrist[0, 3:], atol=3e-7)
