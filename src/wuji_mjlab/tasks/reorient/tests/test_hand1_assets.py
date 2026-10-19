# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import hashlib

import mujoco
import numpy as np
from wuji_mjlab.assets.robots.wuji_hand import wuji_hand_cfg
from wuji_reorient_deploy.repo_assets import load_hand_spec


def _model_digest(model: mujoco.MjModel) -> bytes:
  buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
  mujoco.mj_saveModel(model, None, buffer)
  return hashlib.sha256(buffer.tobytes()).digest()


def test_hand1_loads_wuji_description_body_with_soft() -> None:
  xml_path = wuji_hand_cfg.wuji_hand_xml_path("right")
  assert xml_path.is_relative_to(wuji_hand_cfg.WUJI_HAND_DESCRIPTION_DIR)
  assert mujoco.MjSpec.from_file(str(xml_path)).modelname == "wujihand-right"


def test_hand1_loader_applies_the_task_edits() -> None:
  upstream = mujoco.MjModel.from_xml_path(str(wuji_hand_cfg.wuji_hand_xml_path()))
  model = wuji_hand_cfg._get_spec(wuji_hand_cfg.wuji_hand_xml_path()).compile()
  for field in ("body_pos", "body_quat", "body_mass", "body_inertia", "jnt_range"):
    np.testing.assert_array_equal(getattr(model, field), getattr(upstream, field))
  np.testing.assert_array_equal(model.actuator_gainprm, upstream.actuator_gainprm)
  assert model.opt.timestep == 0.01
  assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
  np.testing.assert_allclose(model.geom_friction, [[0.7, 0.005, 0.0001]] * model.ngeom)
  np.testing.assert_allclose(model.geom_solref, [[0.02, 1.5]] * model.ngeom)
  assert {model.site(i).name for i in range(model.nsite)} == {
    f"right_finger{finger}_tip" for finger in range(1, 6)
  }
  assert model.nexclude == 6
  collidable = {
    model.geom(g).name
    for g in range(model.ngeom)
    if model.geom_contype[g] or model.geom_conaffinity[g]
  }
  assert collidable == set(wuji_hand_cfg._COLLISION_GEOMS)


def test_hand1_deploy_loader_matches_training_loader() -> None:
  training = wuji_hand_cfg._get_spec(
    wuji_hand_cfg.wuji_hand_xml_path("right")
  ).compile()
  deploy = load_hand_spec(1, "right").compile()
  assert _model_digest(training) == _model_digest(deploy)
