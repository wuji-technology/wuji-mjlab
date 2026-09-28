# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import warnings
from contextlib import nullcontext
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer
from wuji_mjlab.tasks.reorient import mdp as reorient_mdp
from wuji_mjlab.tasks.reorient.config.wuji_hand.env_cfgs import (
  wuji_hand_reorient_env_cfg,
)
from wuji_mjlab.tasks.reorient.mdp.cage import CageEscapePenalty
from wuji_mjlab.tasks.reorient.mdp.command_visualization import (
  ReorientCommandVisualization,
  draw_reorient_frame_triads,
  format_reorient_status_markdown,
  goal_vis_env_indices,
  goal_vis_position_above_object,
  reorient_status_color_rgba,
  update_reorient_status_markdown,
)
from wuji_mjlab.tasks.reorient.tooling.scene_builder import set_goal_mocap


class _StubScene:
  def __init__(self, cube_pos: np.ndarray):
    self._cube_pos = np.array(cube_pos, dtype=np.float64)
    self.goal_mocap_id = 0
    self.data = SimpleNamespace(
      mocap_pos=np.zeros((1, 3), dtype=np.float64),
      mocap_quat=np.zeros((1, 4), dtype=np.float64),
    )

  @property
  def cube_pos(self) -> np.ndarray:
    return self._cube_pos.copy()


class _StubVisualizer:
  def __init__(self) -> None:
    self.frames = []

  def add_frame(
    self,
    *,
    position,
    rotation_matrix,
    scale,
    axis_radius,
    alpha,
    label,
    axis_colors=None,
  ) -> None:
    self.frames.append(
      {
        "position": position,
        "rotation_matrix": rotation_matrix,
        "scale": scale,
        "axis_radius": axis_radius,
        "alpha": alpha,
        "label": label,
        "axis_colors": axis_colors,
      }
    )


def test_set_goal_mocap_places_goal_visualization_above_current_cube():
  scene = _StubScene(cube_pos=np.array([0.12, -0.07, 0.63], dtype=np.float64))
  goal_quat = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)

  set_goal_mocap(scene, goal_quat)  # type: ignore[arg-type]

  np.testing.assert_allclose(
    scene.data.mocap_pos[scene.goal_mocap_id],
    np.array([0.12, -0.07, 0.78], dtype=np.float64),
  )
  np.testing.assert_allclose(scene.data.mocap_quat[scene.goal_mocap_id], goal_quat)


def test_goal_vis_position_above_object_uses_live_object_position():
  object_pos = np.array([0.01, 0.02, 0.73], dtype=np.float32)

  got = goal_vis_position_above_object(object_pos)

  np.testing.assert_allclose(got, np.array([0.01, 0.02, 0.88], dtype=np.float32))


def test_wuji_hand_reorient_play_env_cfg_enables_debug_vis_only_for_play():
  train_cfg = wuji_hand_reorient_env_cfg(play=False)
  play_cfg = wuji_hand_reorient_env_cfg(play=True)

  assert train_cfg.commands["reorient_command"].debug_vis is False
  assert play_cfg.commands["reorient_command"].debug_vis is True


def test_format_reorient_status_markdown_shows_ori_error_and_success():
  got = format_reorient_status_markdown(ori_error_rad=np.pi / 6, success=True)

  assert "ori_error" in got
  assert "30.0 deg" in got
  assert "success" in got
  assert "True" in got


def test_command_gui_accepts_viewer_callbacks_and_displays_status():
  from wuji_mjlab.tasks.reorient.mdp.commands import InHandReorientCommand

  markdown = SimpleNamespace(content="")
  server = SimpleNamespace(
    gui=SimpleNamespace(
      add_folder=lambda name: nullcontext(),
      add_markdown=lambda text: markdown,
    )
  )
  command = SimpleNamespace(
    _visualization=ReorientCommandVisualization(entity_name="object"),
    num_envs=1,
    _policy_status_for_env=lambda idx: (np.pi / 6, True),
  )
  InHandReorientCommand.create_gui(
    command,
    "reorient",
    server,
    lambda: 0,
    on_change=lambda: None,
    request_action=lambda name, payload: None,
  )
  assert "30.0 deg" in markdown.content
  assert "True" in markdown.content


def test_cage_debug_visualization_can_be_disabled_by_viewer():
  from wuji_mjlab.tasks.reorient.mdp.cage import CageEscapePenalty

  term = CageEscapePenalty(
    SimpleNamespace(weight=-1.0, params={}),
    SimpleNamespace(num_envs=1, device="cpu"),
  )
  assert term._debug_vis_enabled is True
  term._debug_vis_enabled = False
  term.debug_vis(None)  # Disabled visualization must not access the scene.


def test_reorient_status_color_rgba_maps_success_to_green_and_failure_to_red():
  assert reorient_status_color_rgba(True) == (0.2, 0.9, 0.2, 0.95)
  assert reorient_status_color_rgba(False) == (0.95, 0.2, 0.2, 0.95)


def test_goal_vis_env_indices_draws_all_envs_not_just_selected_env():
  assert goal_vis_env_indices(num_envs=4) == [0, 1, 2, 3]


def test_visualization_helpers_remain_reachable_via_mdp_reexports():
  assert reorient_mdp.goal_vis_env_indices(3) == [0, 1, 2]
  assert reorient_mdp.format_reorient_status_markdown is format_reorient_status_markdown


def test_update_reorient_status_markdown_clamps_env_selection():
  markdown = SimpleNamespace(content="")

  update_reorient_status_markdown(
    status_markdown=markdown,
    get_env_idx=lambda: 99,
    num_envs=2,
    policy_status_for_env=lambda env_idx: (np.pi / 3, env_idx == 1),
  )

  assert "60.0 deg" in markdown.content
  assert "True" in markdown.content


def test_update_reorient_status_markdown_fast_returns_for_empty_envs():
  markdown = SimpleNamespace(content="unchanged")

  update_reorient_status_markdown(
    status_markdown=markdown,
    get_env_idx=lambda: 0,
    num_envs=0,
    policy_status_for_env=lambda env_idx: (_ for _ in ()).throw(
      AssertionError("should not run")
    ),
  )

  assert markdown.content == "unchanged"


def test_draw_reorient_frame_triads_adds_palm_and_tag_frames():
  visualizer = _StubVisualizer()
  palm_pose_w = torch.tensor(
    [[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]],
    dtype=torch.float32,
  )
  tag_pose_w = torch.tensor(
    [[0.11, 0.19, 0.32, 1.0, 0.0, 0.0, 0.0]],
    dtype=torch.float32,
  )

  draw_reorient_frame_triads(
    visualizer=visualizer,
    num_envs=1,
    palm_pose_w=palm_pose_w,
    tag_pose_w=tag_pose_w,
  )

  assert len(visualizer.frames) == 2
  assert visualizer.frames[0]["label"] == "palm_frame_env0"
  assert visualizer.frames[1]["label"] == "tag_frame_env0"


def test_ghost_goal_resolves_live_cube_arucocube_box():
  import mujoco
  from wuji_mjlab.tasks.reorient.tooling.scene_builder import build_reorient_scene

  model = build_reorient_scene(hand="hand2", hand_side="right").model
  viz = ReorientCommandVisualization(entity_name="object")

  assert viz._ensure_ghost_material(model) is True
  expected_mat = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_MATERIAL, "object/arucocube"
  )
  assert expected_mat >= 0
  assert viz.ghost_mat_id == expected_mat
  assert np.allclose(viz.ghost_box_size, [0.027, 0.027, 0.027], atol=1e-4)


def test_visualization_adapter_owns_gui_refresh_state():
  visualization = ReorientCommandVisualization(entity_name="object")
  markdown = SimpleNamespace(content="")
  visualization.attach_status_targets(
    status_markdown=markdown,
    get_env_idx=lambda: 99,
  )

  visualization.update_status_gui(
    num_envs=2,
    policy_status_for_env=lambda env_idx: (np.pi / 4, env_idx == 1),
  )

  assert "45.0 deg" in markdown.content
  assert "True" in markdown.content


@pytest.fixture(scope="module")
def native_play_env():
  cfg = load_env_cfg("WujiHand2_Reorient_50Hz", play=True)
  cfg.seed = 0
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
  try:
    env.reset()
    yield env
  finally:
    env.close()


@pytest.mark.parametrize("decoration", ["ghost", "cage"])
def test_native_play_debug_visuals(native_play_env, decoration):
  env = native_play_env
  model = env.sim.mj_model
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  scene = mujoco.MjvScene(model, maxgeom=2048)
  mujoco.mjv_updateScene(
    model,
    data,
    mujoco.MjvOption(),
    mujoco.MjvPerturb(),
    mujoco.MjvCamera(),
    mujoco.mjtCatBit.mjCAT_ALL,
    scene,
  )
  initial_ngeom = scene.ngeom
  visualizer = MujocoNativeDebugVisualizer(scene, model, env_idx=0)

  env.update_visualizers(visualizer)

  boxes = [
    geom
    for geom in scene.geoms[initial_ngeom : scene.ngeom]
    if geom.type == mujoco.mjtGeom.mjGEOM_BOX
    and geom.category == mujoco.mjtCatBit.mjCAT_DECOR
  ]
  if decoration == "ghost":
    command = env.command_manager.get_term("reorient_command")
    expected_mat = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_MATERIAL, "object/arucocube"
    )
    assert command._visualization.ghost_mat_id == expected_mat >= 0
    ghosts = [geom for geom in boxes if geom.matid == expected_mat]
    assert len(ghosts) == 1
    expected_pos = command.object.data.root_link_pos_w[0].cpu().numpy().copy()
    expected_pos[2] += 0.15
    np.testing.assert_allclose(ghosts[0].pos, expected_pos, atol=1e-6)
    expected_rotation = np.empty(9)
    mujoco.mju_quat2Mat(expected_rotation, command.goal_quat_w[0].cpu().numpy())
    np.testing.assert_allclose(
      ghosts[0].mat, expected_rotation.reshape(3, 3), atol=1e-6
    )
    np.testing.assert_allclose(ghosts[0].size, [0.027, 0.027, 0.027], atol=1e-4)
  else:
    cages = [geom for geom in boxes if np.isclose(geom.rgba[3], 0.25)]
    assert len(cages) == 1
    assert np.all(cages[0].size > 0)


def test_unsupported_visualizer_warns_once(native_play_env):
  class UnsupportedVisualizer(_StubVisualizer):
    pass

  command = native_play_env.command_manager.get_term("reorient_command")
  visualizer = UnsupportedVisualizer()
  with pytest.warns(RuntimeWarning, match="UnsupportedVisualizer"):
    command._debug_vis_impl(visualizer)
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    command._debug_vis_impl(visualizer)
  assert not caught


def test_native_goal_without_material_warns_once():
  model = mujoco.MjModel.from_xml_string("<mujoco/>")
  scene = mujoco.MjvScene(model, maxgeom=32)
  visualizer = MujocoNativeDebugVisualizer(scene, model, env_idx=0)
  visualization = ReorientCommandVisualization(entity_name="object")
  pose = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
  kwargs = dict(
    visualizer=visualizer,
    num_envs=1,
    palm_pose_w=pose,
    tag_pose_w=pose,
    object_pos_w=pose[:, :3],
    goal_quat_w=pose[:, 3:],
    policy_status_for_env=lambda _: (0.0, True),
  )
  with pytest.warns(RuntimeWarning, match="material"):
    visualization.draw_debug_visuals(**kwargs)
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    visualization.draw_debug_visuals(**kwargs)
  assert not caught


def test_viser_command_gui_allows_simulation_steps(native_play_env):
  env = native_play_env
  server = viser.ViserServer(host="localhost", port=0)
  viewer = ViserPlayViewer(
    RslRlVecEnvWrapper(env),
    lambda _: torch.zeros((1, env.action_manager.total_action_dim)),
    viser_server=server,
  )
  start_time = env.sim.data.time[0].item()
  try:
    viewer.run(num_steps=2)
  except Exception:
    viewer.close()
    raise
  finally:
    server.stop()
  assert env.sim.data.time[0].item() >= start_time + 2 * env.step_dt - 1e-6
  assert torch.isfinite(env.scene["object"].data.root_link_pos_w).all()


def test_cage_debug_switch_changes_geometry_without_reward_state(native_play_env):
  env = native_play_env
  cage = next(
    func
    for _, func in env.reward_manager.get_visualizable_terms()
    if isinstance(func, CageEscapePenalty)
  )
  model = env.sim.mj_model
  scene = mujoco.MjvScene(model, maxgeom=32)
  visualizer = MujocoNativeDebugVisualizer(scene, model, env_idx=0)
  counter = cage._penalty_counter.clone()
  try:
    for enabled, expected_boxes in ((False, 0), (True, 1), (False, 0)):
      cage._debug_vis_enabled = enabled
      scene.ngeom = 0
      cage.debug_vis(visualizer)
      assert scene.ngeom == expected_boxes
      torch.testing.assert_close(cage._penalty_counter, counter, rtol=0, atol=0)
  finally:
    cage._debug_vis_enabled = True
