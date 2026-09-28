# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from wuji_reorient_deploy.constants import joint_names, load_constants
from wuji_reorient_deploy.repo_assets import REPO_ASSETS_DIR, load_hand_spec

_RENDER_VIEWER = Path(__file__).resolve().parents[1] / "scripts" / "render_viewer.py"


def _load_render_viewer():
  spec = importlib.util.spec_from_file_location("render_viewer", _RENDER_VIEWER)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


rv = _load_render_viewer()


@pytest.mark.parametrize("gen,side", [(1, "right"), (2, "right")])
def test_build_scene_compiles(gen, side):
  const = load_constants(gen, side)
  root_quat = rv._root_quat(const, 10.0)
  model = rv._build_scene(const, gen, side, root_quat)
  assert model.nu == 2 * const.num_joints
  assert model.nexclude == 2 * len(load_hand_spec(gen, side).excludes)
  names = joint_names(side)
  real_qadr = np.array([model.joint(name).qposadr[0] for name in names])
  ghost_qadr = np.array([model.joint(f"target/{name}").qposadr[0] for name in names])
  cube = model.body("obs_cube")
  assert model.jnt_type[model.body_jntadr[cube.id]] == mujoco.mjtJoint.mjJNT_FREE
  assert model.body("goal_cube").mocapid[0] >= 0

  data = mujoco.MjData(model)
  data.qpos[real_qadr] = const.default_joint_pos
  data.qpos[ghost_qadr] = const.default_joint_pos
  mujoco.mj_forward(model, data)
  assert np.all(np.isfinite(data.xpos))


@pytest.mark.parametrize("gen,mesh", [(1, "right_palm_link"), (2, "r_wrist")])
def test_load_hand_spec_resolves_each_generation(gen, mesh):
  assert (REPO_ASSETS_DIR / "objects/inhand_object/textures").is_dir()
  spec = load_hand_spec(gen, "right")
  assert spec.mesh(mesh) is not None
  model = spec.compile()
  for name in joint_names("right"):
    assert model.joint(name) is not None


def test_viewer_frame_uses_measured_hand_and_target_ghost(monkeypatch):
  target = np.linspace(-0.1, 0.1, 20)
  measured = target + 0.04
  rendered = []

  class SingleFrameViewer:
    def __init__(self, model, data):
      rendered.append((model, data))
      self.cam = SimpleNamespace(
        lookat=np.zeros(3), distance=0.0, azimuth=0.0, elevation=0.0
      )
      self.frames = 0

    def __enter__(self):
      return self

    def __exit__(self, *args):
      pass

    def is_running(self):
      return self.frames == 0

    def sync(self):
      self.frames += 1

  monkeypatch.setattr(sys, "argv", ["render_viewer.py", "--gen", "2", "--hand", "right"])
  monkeypatch.setattr(rv.mujoco.viewer, "launch_passive", SingleFrameViewer)
  monkeypatch.setattr(
    rv, "JointReceiver",
    lambda: SimpleNamespace(latest=lambda: target, latest_measured=lambda: measured),
  )
  monkeypatch.setattr(
    rv, "CubeReceiver",
    lambda **kwargs: SimpleNamespace(
      latest=lambda: (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
      count=0,
    ),
  )
  monkeypatch.setattr(
    rv, "GoalReceiver", lambda: SimpleNamespace(latest=lambda: np.array([1.0, 0, 0, 0]))
  )
  monkeypatch.setattr(rv.time, "sleep", lambda seconds: None)

  rv.main()

  model, data = rendered[0]
  names = joint_names("right")
  np.testing.assert_allclose(
    data.qpos[[model.joint(name).qposadr[0] for name in names]], measured
  )
  np.testing.assert_allclose(
    data.qpos[[model.joint(f"target/{name}").qposadr[0] for name in names]], target
  )
