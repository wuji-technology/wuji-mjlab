# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest
from mjlab.scene import Scene
from wuji_mjlab.assets.robots.wuji_hand2 import wuji_hand2_cfg
from wuji_mjlab.tasks.reorient.config.wuji_hand2 import (
  _reorient_hand2_50hz_env_cfg,
)
from wuji_mjlab.tasks.reorient.config.wuji_hand2.env_cfgs import (
  HAND2_TIP_COLLISION_GEOMS,
  HAND2_TIP_CONTACT_BODY_NAMES,
  wuji_hand2_reorient_env_cfg,
)
from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
  HAND2_REORIENT_ACTUATOR_FORCE_LIMITS,
  HAND2_REORIENT_ACTUATOR_GAINS,
  REORIENT_HAND2_CUBE_INIT_POS,
  REORIENT_HAND2_JOINT_POS,
  REORIENT_HAND2_ROOT_POS,
  REORIENT_HAND2_ROOT_ROT,
)
from wuji_mjlab.tasks.reorient.reorient_constants import (
  REORIENT_JOINT_POS,
  REORIENT_ROBOT_ROOT_POS,
  REORIENT_ROBOT_ROOT_ROT,
)
from wuji_mjlab.tasks.reorient.reorient_env_cfg import (
  TIP_BODY_NAMES,
  TIP_SITE_NAMES,
  UNDESIRED_OBJECT_CONTACT_BODIES,
)
from wuji_mjlab.tasks.reorient.tooling import cage_pose_check
from wuji_reorient_deploy import constants as deploy_constants
from wuji_reorient_deploy.repo_assets import load_hand_spec

SIDES = ("right",)

_EXPECTED_TIP_SITES = {
  "right_finger1_tip",
  "right_finger2_tip",
  "right_finger3_tip",
  "right_finger4_tip",
  "right_finger5_tip",
}
_EXPECTED_TIP_BODIES = {
  "right_finger1_link4",
  "right_finger2_link4",
  "right_finger3_link4",
  "right_finger4_link4",
  "right_finger5_link4",
}
_EXPECTED_TIP_CONTACT_BODIES = {
  "right_finger1_link4",
  "right_finger1_tip_sensor",
  "right_finger2_link4",
  "right_finger2_tip_sensor",
  "right_finger3_link4",
  "right_finger3_tip_sensor",
  "right_finger4_link4",
  "right_finger4_tip_sensor",
  "right_finger5_link4",
  "right_finger5_tip_sensor",
}
_EXPECTED_TIP_COLLISION_GEOMS = {
  "right_finger1_link4_col",
  "right_finger1_tip_sensor_col",
  "right_finger2_link4_col",
  "right_finger2_tip_sensor_col",
  "right_finger3_link4_col",
  "right_finger3_tip_sensor_col",
  "right_finger4_link4_col",
  "right_finger4_tip_sensor_col",
  "right_finger5_link4_col",
  "right_finger5_tip_sensor_col",
}
_EXPECTED_DISTAL_CONTACT_BODIES = {
  "right_finger1_link3",
  "right_finger1_link4",
  "right_finger1_tip_sensor",
  "right_finger2_link3",
  "right_finger2_link4",
  "right_finger2_tip_sensor",
  "right_finger3_link3",
  "right_finger3_link4",
  "right_finger3_tip_sensor",
  "right_finger4_link3",
  "right_finger4_link4",
  "right_finger4_tip_sensor",
  "right_finger5_link3",
  "right_finger5_link4",
  "right_finger5_tip_sensor",
}
_EXPECTED_FINGER_COLLISION_GEOMS = {
  "right_finger1_link1_col",
  "right_finger1_link2_col",
  "right_finger1_link3_col",
  "right_finger1_link4_col",
  "right_finger1_tip_sensor_col",
  "right_finger2_link1_col",
  "right_finger2_link2_col",
  "right_finger2_link3_col",
  "right_finger2_link4_col",
  "right_finger2_tip_sensor_col",
  "right_finger3_link1_col",
  "right_finger3_link2_col",
  "right_finger3_link3_col",
  "right_finger3_link4_col",
  "right_finger3_tip_sensor_col",
  "right_finger4_link1_col",
  "right_finger4_link2_col",
  "right_finger4_link3_col",
  "right_finger4_link4_col",
  "right_finger4_tip_sensor_col",
  "right_finger5_link1_col",
  "right_finger5_link2_col",
  "right_finger5_link3_col",
  "right_finger5_link4_col",
  "right_finger5_tip_sensor_col",
}
_EXPECTED_GEOM_SIZE_DR_GEOMS = {
  f"right_finger{finger}_link{link}_col" for finger in range(1, 6) for link in (2, 3)
}
_EXPECTED_ACTIVE_COLLISION_GEOMS = {
  "right_palm_collision",
  *{
    f"right_finger{finger}_{suffix}"
    for finger in range(1, 6)
    for suffix in (
      "link2_col",
      "link3_col",
      "link4_col",
      "tip_sensor_col",
    )
  },
}
_EXPECTED_DISABLED_L1_CAPSULES = {
  f"right_finger{finger}_link1_col" for finger in range(1, 6)
}
_EXPECTED_DISABLED_COLLISION_MESHES = {
  f"right_finger{finger}_link{link}_col_mesh_disabled"
  for finger in range(1, 6)
  for link in (1, 2, 3)
}
_EXPECTED_FINGER_CONTACT_PARAM_GEOMS = {
  "right_finger2_link2_col",
  "right_finger2_link3_col",
  "right_finger2_link4_col",
  "right_finger2_tip_sensor_col",
  "right_finger3_link2_col",
  "right_finger3_link3_col",
  "right_finger3_link4_col",
  "right_finger3_tip_sensor_col",
  "right_finger4_link2_col",
  "right_finger4_link3_col",
  "right_finger4_link4_col",
  "right_finger4_tip_sensor_col",
  "right_finger5_link2_col",
  "right_finger5_link3_col",
  "right_finger5_link4_col",
  "right_finger5_tip_sensor_col",
}
_EXPECTED_UNDESIRED_OBJECT_CONTACT_BODIES = {
  "right_palm_link",
  "right_finger1_link1",
  "right_finger2_link1",
  "right_finger2_link2",
  "right_finger2_link3",
  "right_finger3_link1",
  "right_finger3_link2",
  "right_finger3_link3",
  "right_finger4_link1",
  "right_finger4_link2",
  "right_finger4_link3",
  "right_finger5_link1",
  "right_finger5_link2",
}


def _matching_names(names: set[str], patterns: str | tuple[str, ...]) -> set[str]:
  if isinstance(patterns, str):
    patterns = (patterns,)
  return {
    name for name in names if any(re.fullmatch(pattern, name) for pattern in patterns)
  }


def _names(model: mujoco.MjModel, obj_type: mujoco.mjtObj, count: int) -> set[str]:
  return {
    mujoco.mj_id2name(model, obj_type, i)
    for i in range(count)
    if mujoco.mj_id2name(model, obj_type, i)
  }


def test_hand2_public_xml_preserves_upstream_content_with_public_name() -> None:
  xml = wuji_hand2_cfg.wuji_hand2_xml_path("right").read_bytes()
  name_line, contents = xml.split(b"\n", 1)
  assert name_line == b'<mujoco model="wujihand2-right">'
  # Only the public model name differs from the original asset snapshot.
  assert hashlib.sha256(contents).hexdigest() == (
    "9cede782aaed3e9aa955baf36dbf4ae0e6cb1057eef087cb5cd174e473025cec"
  )


def test_hand2_reorient_rig_retains_task_geom_and_site_settings() -> None:
  model = (
    wuji_hand2_reorient_env_cfg(num_envs=1).scene.entities["robot"].spec_fn().compile()
  )
  assert model.ngeom == 73
  np.testing.assert_allclose(model.geom_friction, [[0.7, 0.005, 0.0001]] * model.ngeom)
  np.testing.assert_allclose(model.geom_solref, [[0.02, 1.5]] * model.ngeom)
  np.testing.assert_allclose(
    model.geom_solimp, [[0.9, 0.95, 0.001, 0.5, 2.0]] * model.ngeom
  )
  np.testing.assert_array_equal(model.geom_priority, [1] * model.ngeom)
  assert model.geom_group[model.geom("right_palm_collision").id] == 2
  assert model.geom_group[0] == 1
  assert model.geom_group[model.geom("right_finger1_link1_col").id] == 3
  for finger in range(1, 6):
    assert model.site_group[model.site(f"right_finger{finger}_tip").id] == 4
  assert model.opt.timestep == 0.01
  assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
  assert model.opt.iterations == 5
  assert model.opt.ls_iterations == 8
  assert model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP


def test_hand2_reorient_preserves_explicit_geom_physics() -> None:
  from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
    get_hand2_reorient_spec,
  )

  base = wuji_hand2_cfg._get_spec(wuji_hand2_cfg.wuji_hand2_xml_path("right"))
  geom = base.geoms[0]
  geom.friction = (1.3, 0.006, 0.0002)
  geom.solref = (0.03, 1.2)
  geom.solimp = (0.8, 0.9, 0.002, 0.5, 2)
  geom.priority = 2
  model = get_hand2_reorient_spec(lambda: base).compile()
  np.testing.assert_allclose(model.geom_friction[0], [1.3, 0.006, 0.0002])
  np.testing.assert_allclose(model.geom_solref[0], [0.03, 1.2])
  np.testing.assert_allclose(model.geom_solimp[0], [0.8, 0.9, 0.002, 0.5, 2])
  assert model.geom_priority[0] == 2
  assert model.geom_group[0] == 1


@pytest.mark.parametrize("side", SIDES)
def test_hand2_loader_exposes_task_naming_convention(side: str) -> None:
  cfg = wuji_hand2_reorient_env_cfg(hand_side=side, num_envs=1)
  model = cfg.scene.entities["robot"].spec_fn().compile()

  joints = _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
  actuators = _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
  sites = _names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite)
  bodies = _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
  geoms = _names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
  assert model.njnt == 20
  assert model.nu == 20

  for n in range(1, 6):
    for m in range(1, 5):
      assert f"{side}_finger{n}_joint{m}" in joints
      assert f"{side}_finger{n}_joint{m}_actuator" in actuators
      assert f"{side}_finger{n}_link{m}" in bodies
    assert f"{side}_finger{n}_tip" in sites

  assert f"{side}_palm_link" in bodies
  assert f"{side}_palm_collision" in geoms
  assert f"{side}_wrist_tag" in sites


def test_hand2_loader_rejects_extra_site(tmp_path: Path) -> None:
  original_xml = wuji_hand2_cfg.wuji_hand2_xml_path("right")
  tmp_mjcf = tmp_path / "mjcf"
  tmp_mjcf.mkdir()
  tmp_xml = tmp_mjcf / original_xml.name
  xml = original_xml.read_text()
  first_body = xml.index("<body name=")
  body_open_end = xml.index(">", first_body) + 1
  xml = (
    xml[:body_open_end] + '<site name="r_extra" size="0.001"/>' + xml[body_open_end:]
  )
  tmp_xml.write_text(xml)
  mesh_dir = tmp_path / "meshes"
  mesh_dir.mkdir()
  (mesh_dir / "right").symlink_to(
    wuji_hand2_cfg.wuji_hand2_mesh_dir("right"), target_is_directory=True
  )

  with pytest.raises(ValueError, match="r_extra"):
    wuji_hand2_cfg._get_spec(tmp_xml)


def test_hand2_loader_rejects_missing_joint_entry(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delitem(wuji_hand2_cfg.WUJI_HAND2_TASK_NAMES["joint"], "r_thumb_mcp")
  with pytest.raises(ValueError, match="r_thumb_mcp"):
    wuji_hand2_cfg._get_spec(wuji_hand2_cfg.wuji_hand2_xml_path("right"))


def test_hand2_deploy_loader_matches_training_loader() -> None:
  assert deploy_constants.WUJI_HAND2_TASK_NAMES == wuji_hand2_cfg.WUJI_HAND2_TASK_NAMES
  training = wuji_hand2_cfg._get_spec(
    wuji_hand2_cfg.wuji_hand2_xml_path("right")
  ).compile()
  deploy = load_hand_spec(2, "right").compile()
  training_bytes = np.empty(mujoco.mj_sizeModel(training), dtype=np.uint8)
  deploy_bytes = np.empty(mujoco.mj_sizeModel(deploy), dtype=np.uint8)
  mujoco.mj_saveModel(training, None, training_bytes)
  mujoco.mj_saveModel(deploy, None, deploy_bytes)
  assert (
    hashlib.sha256(training_bytes.tobytes()).digest()
    == hashlib.sha256(deploy_bytes.tobytes()).digest()
  )


def test_hand2_reorient_name_patterns_resolve_expected_assets() -> None:
  cfg = wuji_hand2_reorient_env_cfg(hand_side="right", num_envs=1)
  model = cfg.scene.entities["robot"].spec_fn().compile()
  sites = _names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite)
  bodies = _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
  geoms = _names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
  assert model.njnt == len(HAND2_REORIENT_ACTUATOR_GAINS) == 20
  assert model.nu == 20
  assert {
    f"{joint_name}_actuator" for joint_name in HAND2_REORIENT_ACTUATOR_GAINS
  } == _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
  for joint_name, (kp, kv) in HAND2_REORIENT_ACTUATOR_GAINS.items():
    actuator_id = model.actuator(f"{joint_name}_actuator").id
    np.testing.assert_allclose(model.actuator_gainprm[actuator_id, 0], kp, atol=1e-6)
    np.testing.assert_allclose(model.actuator_biasprm[actuator_id, 1], -kp, atol=1e-6)
    np.testing.assert_allclose(model.actuator_biasprm[actuator_id, 2], -kv, atol=1e-6)
  assert len(HAND2_REORIENT_ACTUATOR_FORCE_LIMITS) == 20
  for joint_name, limit in HAND2_REORIENT_ACTUATOR_FORCE_LIMITS.items():
    actuator_id = model.actuator(f"{joint_name}_actuator").id
    joint_id = model.joint(joint_name).id
    expected = [-limit, limit]
    np.testing.assert_allclose(
      model.actuator_forcerange[actuator_id], expected, atol=1e-6
    )
    np.testing.assert_allclose(model.jnt_actfrcrange[joint_id], expected, atol=1e-6)
    assert model.actuator_forcelimited[actuator_id] == mujoco.mjtLimited.mjLIMITED_TRUE
    assert model.jnt_actfrclimited[joint_id] == mujoco.mjtLimited.mjLIMITED_TRUE
  active_collision_geoms = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    for geom_id in range(model.ngeom)
    if model.geom_contype[geom_id] != 0 and model.geom_conaffinity[geom_id] != 0
  }
  assert active_collision_geoms == _EXPECTED_ACTIVE_COLLISION_GEOMS
  assert len(active_collision_geoms) == 21

  for geom_name in _EXPECTED_GEOM_SIZE_DR_GEOMS:
    geom_id = model.geom(geom_name).id
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CAPSULE
  for geom_name in _EXPECTED_DISABLED_L1_CAPSULES:
    geom_id = model.geom(geom_name).id
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CAPSULE
    assert model.geom_contype[geom_id] == 0
    assert model.geom_conaffinity[geom_id] == 0
  for geom_name in _EXPECTED_DISABLED_COLLISION_MESHES:
    geom_id = model.geom(geom_name).id
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
    assert model.geom_contype[geom_id] == 0
    assert model.geom_conaffinity[geom_id] == 0
  for geom_name in {
    "right_palm_collision",
    *_EXPECTED_TIP_COLLISION_GEOMS,
  }:
    assert model.geom_type[model.geom(geom_name).id] == mujoco.mjtGeom.mjGEOM_MESH

  sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}

  assert _matching_names(sites, TIP_SITE_NAMES) == _EXPECTED_TIP_SITES
  assert _matching_names(bodies, TIP_BODY_NAMES) == _EXPECTED_TIP_BODIES
  assert (
    _matching_names(bodies, HAND2_TIP_CONTACT_BODY_NAMES)
    == _EXPECTED_TIP_CONTACT_BODIES
  )
  assert (
    _matching_names(geoms, HAND2_TIP_COLLISION_GEOMS) == _EXPECTED_TIP_COLLISION_GEOMS
  )
  assert (
    _matching_names(bodies, UNDESIRED_OBJECT_CONTACT_BODIES)
    == _EXPECTED_UNDESIRED_OBJECT_CONTACT_BODIES
  )

  fingertip_position_patterns = (
    cfg.observations["critic"]
    .terms["fingertip_positions"]
    .params["asset_cfg"]
    .body_names
  )
  assert _matching_names(bodies, fingertip_position_patterns) == _EXPECTED_TIP_BODIES
  assert (
    _matching_names(geoms, sensors["tip_object_contact"].primary.pattern)
    == _EXPECTED_TIP_COLLISION_GEOMS
  )
  assert (
    _matching_names(bodies, sensors["robot_contact"].primary.pattern)
    == _EXPECTED_TIP_CONTACT_BODIES
  )
  assert (
    _matching_names(bodies, sensors["distal_finger_object_found"].primary.pattern)
    == _EXPECTED_DISTAL_CONTACT_BODIES
  )
  assert (
    _matching_names(bodies, sensors["undesired_object_contact"].primary.pattern)
    == _EXPECTED_UNDESIRED_OBJECT_CONTACT_BODIES
  )
  assert (
    _matching_names(geoms, sensors["finger_collision"].primary.pattern)
    == _EXPECTED_FINGER_COLLISION_GEOMS
  )

  friction_patterns = cfg.events["robot_friction"].params["asset_cfg"].geom_names
  assert _matching_names(geoms, friction_patterns) == {
    "right_palm_collision",
    *_EXPECTED_FINGER_COLLISION_GEOMS,
  }
  finger_contact_patterns = (
    cfg.events["contact_params_fingers"].params["robot_cfg"].geom_names
  )
  assert (
    _matching_names(geoms, finger_contact_patterns)
    == _EXPECTED_FINGER_CONTACT_PARAM_GEOMS
  )
  geom_size_patterns = cfg.events["robot_geom_size"].params["asset_cfg"].geom_names
  assert _matching_names(geoms, geom_size_patterns) == _EXPECTED_GEOM_SIZE_DR_GEOMS


def test_hand2_training_contact_softness_covers_each_colliding_fingertip() -> None:
  cfg = _reorient_hand2_50hz_env_cfg(num_envs=1)
  scene = Scene(cfg.scene, device="cpu")
  model = scene.compile()
  robot_indexing = scene.entities["robot"]._compute_indexing(model, "cpu")

  selected = {}
  for event_name in ("contact_params_palm_thumb", "contact_params_fingers"):
    robot_cfg = cfg.events[event_name].params["robot_cfg"]
    robot_cfg.resolve(scene)
    geom_ids = robot_indexing.geom_ids[robot_cfg.geom_ids]
    selected[event_name] = {
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
      for geom_id in geom_ids
    }

  assert selected["contact_params_palm_thumb"] == {
    "robot/right_palm_collision",
    "robot/right_finger1_link2_col",
    "robot/right_finger1_link3_col",
    "robot/right_finger1_link4_col",
    "robot/right_finger1_tip_sensor_col",
  }
  assert selected["contact_params_fingers"] == {
    f"robot/right_finger{finger}_{part}_col"
    for finger in range(2, 6)
    for part in ("link2", "link3", "link4", "tip_sensor")
  }


def test_hand2_deploy_gains_match_task_override() -> None:
  gains = np.array(list(HAND2_REORIENT_ACTUATOR_GAINS.values()))
  np.testing.assert_allclose(deploy_constants._GEN2_SIM_KP, gains[:, 0], atol=1e-6)
  np.testing.assert_allclose(deploy_constants._GEN2_SIM_KD, gains[:, 1], atol=1e-6)


@pytest.mark.parametrize("side", SIDES)
def test_hand2_init_pose_cages_cube(side: str) -> None:
  joint_pos = {
    key.removeprefix(".*_"): value for key, value in REORIENT_HAND2_JOINT_POS.items()
  }
  result = cage_pose_check.run_check(
    hand="wuji_hand2",
    side=side,
    root_pos=REORIENT_HAND2_ROOT_POS,
    root_quat=REORIENT_HAND2_ROOT_ROT,
    cube_pos=REORIENT_HAND2_CUBE_INIT_POS[side],
    joint_pos=joint_pos,
    sim_seconds=3.0,
    verbose=False,
  )
  assert not result["dropped"], f"cube fell out of the {side} hand cage"
  assert result["drift"] < 0.05, f"cube drifted {result['drift']:.3f} m"


@pytest.mark.parametrize(
  (
    "hand_args",
    "expected_hand",
    "expected_root_pos",
    "expected_root_rot",
    "expected_joint_patterns",
  ),
  (
    pytest.param(
      [],
      "wuji_hand2",
      REORIENT_HAND2_ROOT_POS,
      REORIENT_HAND2_ROOT_ROT,
      REORIENT_HAND2_JOINT_POS,
      id="default-hand2",
    ),
    pytest.param(
      ["--hand", "wuji_hand"],
      "wuji_hand",
      REORIENT_ROBOT_ROOT_POS,
      REORIENT_ROBOT_ROOT_ROT,
      REORIENT_JOINT_POS,
      id="hand1",
    ),
  ),
)
def test_cage_pose_cli_uses_each_hands_task_pose(
  hand_args: list[str],
  expected_hand: str,
  expected_root_pos: tuple[float, ...],
  expected_root_rot: tuple[float, ...],
  expected_joint_patterns: dict[str, float],
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  run_check = Mock(return_value={})
  monkeypatch.setattr(cage_pose_check, "run_check", run_check)

  cage_pose_check.main([*hand_args, "--cube-pos", "0", "0", "1", "--sim-seconds", "0"])

  run_check.assert_called_once()
  args = run_check.call_args.args
  assert args[0] == expected_hand
  assert args[2] == expected_root_pos
  assert args[3] == expected_root_rot
  assert args[5] == {
    pattern.removeprefix(".*_"): value
    for pattern, value in expected_joint_patterns.items()
  }


@pytest.mark.parametrize("side", SIDES)
def test_hand2_env_cfg_binds_side_specific_names(side: str) -> None:
  cfg = wuji_hand2_reorient_env_cfg(hand_side=side, num_envs=16)

  assert cfg.viewer.body_name == f"{side}_palm_link"

  finger_collision = next(s for s in cfg.scene.sensors if s.name == "finger_collision")
  assert finger_collision.secondary.pattern == f"{side}_palm_link"

  soft_pad = cfg.events["contact_params_palm_thumb"].params["robot_cfg"].geom_names
  assert f"{side}_palm_collision" in soft_pad
  assert all(g.startswith(side) for g in soft_pad)
