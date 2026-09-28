# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Hand2 model, contact, and training/deployment frame contracts."""

from __future__ import annotations

import importlib
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from mjlab.entity import Entity
from mjlab.scene import Scene
from wuji_mjlab.assets.robots.wuji_hand2 import get_wuji_hand2_cfg
from wuji_mjlab.assets.robots.wuji_hand2.wuji_hand2_cfg import wuji_hand2_xml_path
from wuji_mjlab.tasks.pen_spin.config.wuji_hand2.env_cfgs import (
  wuji_hand2_pen_spin_env_cfg,
)
from wuji_mjlab.tasks.pen_spin.config.wuji_hand2.hand2_constants import (
  HAND2_PEN_SPIN_CAPSULES,
  PEN_SPIN_HAND2_TORQUE_LIMIT,
  apply_hand2_pen_spin_collision_overlay,
  get_pen_spin_hand2_cfg,
)
from wuji_mjlab.tasks.pen_spin.config.wuji_hand2.rsl_rl.ppo import (
  wuji_hand2_pen_spin_ppo_runner_cfg,
)
from wuji_mjlab.tasks.pen_spin.mdp.commands import SingleHandTrackingReader
from wuji_mjlab.tasks.pen_spin.mdp.rewards import finger_self_collision_penalty
from wuji_mjlab.tasks.pen_spin.tooling.clip_store import ResolvedClipSet, TrackingClip

_REPO = Path(__file__).resolve().parents[5]

# Joint caps from the pre-hand2 right_mjlab.xml, in finger/joint order.
_BASELINE_JOINT_TORQUE_LIMITS = np.array(
  [
    [1.02, 0.83, 0.35, 0.21],
    [1.05, 0.38, 0.31, 0.30],
    [1.06, 0.39, 0.32, 0.29],
    [1.02, 0.41, 0.28, 0.30],
    [0.87, 0.36, 0.28, 0.30],
  ]
).ravel()


@pytest.fixture
def deploy(monkeypatch):
  # Deployment deliberately has no training dependency. Keep its generic "lib"
  # package isolated from any other deploy tests sharing this pytest process.
  saved = {
    k: v
    for k, v in sys.modules.items()
    if k in ("lib", "runtime") or k.startswith(("lib.", "runtime."))
  }
  for name in saved:
    del sys.modules[name]
  monkeypatch.syspath_prepend(str(_REPO / "deploy/pen_spin"))
  yield SimpleNamespace(
    model=importlib.import_module("lib.hand2_model"),
    clips=importlib.import_module("lib.motion_clip"),
    tag=importlib.import_module("lib.tag_frame"),
    scene=importlib.import_module("runtime.scene"),
  )
  for name in tuple(sys.modules):
    if name in ("lib", "runtime") or name.startswith(("lib.", "runtime.")):
      del sys.modules[name]
  sys.modules.update(saved)


def test_capsule_overlay_keeps_hand2_meshes_mass_and_joints():
  base = get_wuji_hand2_cfg().spec_fn().compile()
  pen = get_pen_spin_hand2_cfg().spec_fn().compile()
  assert pen.nmesh == base.nmesh  # No copied legacy meshes.
  for field in (
    "body_parentid",
    "body_pos",
    "body_quat",
    "body_mass",
    "body_inertia",
    "jnt_axis",
    "jnt_pos",
    "jnt_range",
    "body_ipos",
    "body_iquat",
  ):
    np.testing.assert_array_equal(getattr(pen, field), getattr(base, field))
  # The global names buffer also contains the deliberately changed excludes.
  for kind, count in (
    ("body", base.nbody),
    ("joint", base.njnt),
    ("site", base.nsite),
    ("actuator", base.nu),
  ):
    assert [getattr(pen, kind)(i).name for i in range(count)] == [
      getattr(base, kind)(i).name for i in range(count)
    ]
  # Added collision proxies must not alter the original visual surface.
  for bid in range(base.nbody):
    old_geoms = np.flatnonzero(base.geom_bodyid == bid)
    kept_geoms = np.flatnonzero(pen.geom_bodyid == bid)[: len(old_geoms)]
    for field in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_rgba"):
      np.testing.assert_array_equal(
        getattr(pen, field)[kept_geoms], getattr(base, field)[old_geoms]
      )
  for mid in range(base.nmesh):
    other = pen.mesh(base.mesh(mid).name).id
    a, n = base.mesh_vertadr[mid], base.mesh_vertnum[mid]
    b = pen.mesh_vertadr[other]
    np.testing.assert_array_equal(pen.mesh_vert[b : b + n], base.mesh_vert[a : a + n])
  np.testing.assert_allclose(pen.actuator_forcerange[:, 1], PEN_SPIN_HAND2_TORQUE_LIMIT)
  np.testing.assert_array_equal(pen.jnt_actfrclimited, 1)
  np.testing.assert_allclose(
    pen.jnt_actfrcrange,
    np.column_stack((-_BASELINE_JOINT_TORQUE_LIMITS, _BASELINE_JOINT_TORQUE_LIMITS)),
  )


def test_collision_surface_matches_cube_overlay():
  from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
    apply_hand2_reorient_collision_overlay,
  )

  base_spec_fn = get_wuji_hand2_cfg().spec_fn
  pen = apply_hand2_pen_spin_collision_overlay(base_spec_fn()).compile()
  cube = apply_hand2_reorient_collision_overlay(base_spec_fn()).compile()
  assert len(HAND2_PEN_SPIN_CAPSULES) == 15
  assert pen.names == cube.names
  for field in (
    "geom_type",
    "geom_pos",
    "geom_quat",
    "geom_size",
    "geom_dataid",
    "geom_contype",
    "geom_conaffinity",
    "exclude_signature",
  ):
    np.testing.assert_array_equal(getattr(pen, field), getattr(cube, field))
  capsule = pen.geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE
  active = (pen.geom_contype | pen.geom_conaffinity) != 0
  assert capsule.sum() == 15
  assert (capsule & active).sum() == 10
  assert ((pen.geom_type == mujoco.mjtGeom.mjGEOM_MESH) & active).sum() == 11


@pytest.mark.parametrize("effort_scale", [0.7, 1.0, 1.05, 2.0])
@pytest.mark.parametrize("direction", [-1, 1])
def test_effort_dr_preserves_baseline_saturated_joint_torque(effort_scale, direction):
  model = Entity(get_pen_spin_hand2_cfg()).spec.compile()
  # XmlActuator effort DR scales actuator_forcerange only. Include a scale
  # beyond the training band to verify the original joint caps still apply.
  model.actuator_forcerange[:] *= effort_scale
  data = mujoco.MjData(model)
  data.ctrl[:] = direction * 10.0
  mujoco.mj_forward(model, data)

  actuator_limits = np.asarray(PEN_SPIN_HAND2_TORQUE_LIMIT) * effort_scale
  joint_ids = model.actuator_trnid[:, 0]
  dof_ids = model.jnt_dofadr[joint_ids]
  np.testing.assert_allclose(data.actuator_force, direction * actuator_limits)
  np.testing.assert_allclose(
    data.qfrc_actuator[dof_ids],
    direction * np.minimum(actuator_limits, _BASELINE_JOINT_TORQUE_LIMITS),
  )


def test_deploy_name_adapter_matches_main_loader(deploy):
  training = get_wuji_hand2_cfg().spec_fn().compile()
  deployment = deploy.model.load_hand_spec(wuji_hand2_xml_path()).compile()
  assert training.names == deployment.names
  for field in (
    "body_parentid",
    "body_pos",
    "body_quat",
    "geom_bodyid",
    "mesh_vert",
    "exclude_signature",
    "actuator_trnid",
  ):
    np.testing.assert_array_equal(getattr(training, field), getattr(deployment, field))


def test_active_pen_collision_regions_receive_contact_tuning():
  model = Entity(get_pen_spin_hand2_cfg()).spec.compile()
  active = np.flatnonzero(model.geom_contype | model.geom_conaffinity)
  expected = {"right_palm_collision"} | {
    f"right_finger{finger}_{part}_col"
    for finger in range(1, 6)
    for part in ("link2", "link3", "link4", "tip_sensor")
  }
  assert {model.geom(int(i)).name for i in active} == expected
  np.testing.assert_array_equal(model.geom_condim[active], 4)
  np.testing.assert_array_equal(model.geom_priority[active], 1)
  np.testing.assert_allclose(
    model.geom_friction[active], np.tile([0.7, 0.005, 0.0001], (len(active), 1))
  )
  np.testing.assert_allclose(
    model.geom_solref[active], np.tile([0.02, 1.5], (len(active), 1))
  )


def test_pen_disables_only_link1_and_restores_palm_exclusion():
  pen = get_pen_spin_hand2_cfg().spec_fn().compile()
  for finger in range(1, 6):
    proximal = pen.geom(f"right_finger{finger}_link1_col").id
    assert pen.geom_contype[proximal] == pen.geom_conaffinity[proximal] == 0
    for part in ("link2", "link3", "link4", "tip_sensor"):
      geom = pen.geom(f"right_finger{finger}_{part}_col").id
      assert pen.geom_contype[geom] == pen.geom_conaffinity[geom] == 1
  pairs = {
    frozenset((pen.body(int(sig) >> 16).name, pen.body(int(sig) & 65535).name))
    for sig in pen.exclude_signature
  }
  assert pairs == {frozenset(("right_palm_link", "right_finger1_link2"))}


@pytest.fixture(scope="module")
def scene():
  cfg = wuji_hand2_pen_spin_env_cfg(play=True)
  return Scene(cfg.scene, device="cpu")


def test_self_contact_sensor_covers_nail_and_flesh(scene):
  assert len(scene["finger_collision"].primary_names) == 25
  for finger in range(1, 6):
    assert f"right_finger{finger}_link4_col" in scene["finger_collision"].primary_names
    assert (
      f"right_finger{finger}_tip_sensor_col" in scene["finger_collision"].primary_names
    )


@pytest.mark.parametrize("finger", range(1, 6))
@pytest.mark.parametrize("part", ["link4", "tip_sensor"])
def test_each_hand2_nail_and_flesh_mesh_has_physical_contact(scene, finger, part):
  model = scene.compile()
  # Leave only this hand2 piece active: neither nail nor flesh may rely on its
  # sibling collider to produce the contact force.
  tip = model.geom(f"robot/right_finger{finger}_{part}_col").id
  for gid in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
    if name.startswith("robot/") and gid != tip:
      model.geom_contype[gid] = model.geom_conaffinity[gid] = 0
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  adr = model.joint("object/cube_freejoint").qposadr[0]
  center = data.geom_xpos[tip].copy()
  # Probe near the pad surface; placing a cylinder exactly at a convex mesh's
  # center is a degenerate deep overlap and need not produce a contact.
  for offset in itertools.product((-0.015, 0, 0.015), repeat=3):
    data.qpos[adr : adr + 3] = center + offset
    data.qpos[adr + 3 : adr + 7] = (1, 0, 0, 0)
    mujoco.mj_forward(model, data)
    contacts = [
      i
      for i, c in enumerate(data.contact)
      if tip in (c.geom1, c.geom2) and c.dist < -1e-6
    ]
    wrench = np.zeros(6)
    normal_forces = []
    for cid in contacts:
      mujoco.mj_contactForce(model, data, cid, wrench)
      normal_forces.append(wrench[0])
    # A detected constraint can carry zero force at a particular pose. Search
    # for a loaded contact to verify solver response as well as detection.
    if normal_forces and max(normal_forces) > 0:
      break
  else:
    pytest.fail(f"pen never contacted hand2 finger {finger} {part}")
  assert max(normal_forces) > 0
  for name in scene["finger_collision"].primary_names:
    sensor = model.sensor(f"finger_collision_{name}_found")
    assert data.sensordata[sensor.adr[0]] == 0  # Pen contact is not self-collision.


def test_nail_and_flesh_are_one_rigid_distal_link(scene):
  model = scene.compile()
  for finger in range(1, 6):
    nail = model.body(f"robot/right_finger{finger}_link4").id
    flesh = model.body(f"robot/right_finger{finger}_tip_sensor").id
    assert model.body_parentid[flesh] == nail
    assert model.body_jntnum[flesh] == 0
    assert model.body_weldid[flesh] == model.body_weldid[nail]


def test_self_collision_merges_nail_and_flesh_once(scene):
  names = scene["finger_collision"].primary_names
  found = torch.zeros((6, len(names)))
  found[0, names.index("right_finger2_link4_col")] = 3
  found[1, names.index("right_finger2_link3_col")] = 1
  found[1, names.index("right_finger2_link4_col")] = 1
  found[2, names.index("right_finger1_link4_col")] = 1
  found[2, names.index("right_finger2_link4_col")] = 1
  found[3, names.index("right_finger2_tip_sensor_col")] = 1
  found[4, names.index("right_finger2_link4_col")] = 1
  found[4, names.index("right_finger2_tip_sensor_col")] = 1
  sensor = SimpleNamespace(primary_names=names, data=SimpleNamespace(found=found))
  env = SimpleNamespace(scene={"finger_collision": sensor})
  penalty = finger_self_collision_penalty(None, env)
  assert penalty(env).tolist() == [1, 2, 2, 1, 1, 0]


def test_contact_dr_includes_hand2_nail_and_flesh():
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  cfg = wuji_hand2_pen_spin_env_cfg()
  scene = Scene(cfg.scene, device="cpu")
  selection = SceneEntityCfg(
    "robot", geom_names=cfg.events["robot_friction"].params["asset_cfg"].geom_names
  )
  selection.resolve(scene)
  expected = ["right_palm_collision"] + [
    f"right_finger{finger}_{part}_col"
    for finger in range(1, 6)
    for part in ("link1", "link2", "link3", "link4", "tip_sensor")
  ]
  assert [scene["robot"].geom_names[i] for i in selection.geom_ids] == expected


def test_clip_readers_preserve_hand2_policy_frame(tmp_path, deploy):
  frames = 3
  wrist = np.tile([0.1, -0.2, 0.5, 1, 0, 0, 0], (frames, 1))
  objects = np.tile([0.04, 0.01, 0.62, 1, 0, 0, 0], (frames, 1))
  links = np.random.default_rng(7).normal(0, 0.02, (frames, 21, 3)) + wrist[:, None, :3]
  path = tmp_path / "motion.npz"
  np.savez(
    path,
    reference_qpos=np.zeros((frames, 20)),
    wrist_pose_w=wrist,
    object_pose_w=objects,
    link_pos_w=links,
    mediapipe_landmarks_w=np.zeros((frames, 21, 3)),
  )
  training = SingleHandTrackingReader(
    ResolvedClipSet(frame_dt=0.02, clips=(TrackingClip("group/clip", "group", path),))
  ).read()
  deployment = deploy.clips.load_motion_clip(path)
  np.testing.assert_allclose(training["wrist_pose_w"], wrist, atol=1e-7)
  np.testing.assert_allclose(
    training["object_pose"], deployment.object_pose_local, atol=1e-6
  )
  np.testing.assert_allclose(training["link_pos"], deployment.link_pos_local, atol=1e-6)
  # The identity hand2 wrist makes the policy coordinates unambiguous.
  # In particular the first link must not be overwritten by either reader.
  expected_object = objects[:, :3] - wrist[:, :3]
  np.testing.assert_allclose(
    deployment.object_pose_local[:, :3], expected_object, atol=1e-6
  )
  np.testing.assert_allclose(
    deployment.link_pos_local,
    links - wrist[:, None, :3],
    atol=1e-6,
  )


def test_deploy_camera_uses_hand2_policy_frame_and_viewer(deploy):
  tag_pos = np.array([0.05, -0.03, 0.12])
  local_pos, local_quat = deploy.tag.pen_in_palm(tag_pos, np.array([1, 0, 0, 0]))
  # The calibrated tag axes map (x, y, z) to (y, z, x) in the wrist.
  expected_pos = np.array([-0.03300004, 0.157157842, 0.1132369])
  expected_quat = np.array([0.5, -0.5, -0.5, -0.5])
  np.testing.assert_allclose(local_pos, expected_pos, atol=1e-8)
  assert abs(np.dot(local_quat, expected_quat)) == pytest.approx(1)

  wrist_world = np.array([0.1, -0.3, 0.6, 0.5, 0.5, -0.5, 0.5])
  actual_pos, actual_quat = deploy.tag.lift_wrist_local_to_scene(
    local_pos, local_quat, wrist_world[:3], wrist_world[3:]
  )
  expected_pos, expected_quat = deploy.tag.compose(
    wrist_world[:3], wrist_world[3:], expected_pos, expected_quat
  )
  np.testing.assert_allclose(actual_pos, expected_pos, atol=1e-8)
  assert abs(np.dot(actual_quat, expected_quat)) == pytest.approx(1)


@pytest.mark.parametrize("ghost", [False, True])
@pytest.mark.parametrize("jmode", ["default", "neg90"])
def test_deploy_mount_and_ghost_keep_physical_pose(deploy, ghost, jmode):
  scene = deploy.scene.build(ghost=ghost, jmode=jmode)
  expected = (
    (
      -0.08441049950924943,
      0.00300004,
      -0.0309017758746512,
      0.5416685725611728,
      0.45452740016366616,
      0.45452740016366616,
      0.5416685725611728,
    )
    if jmode == "default"
    else (-0.0847999, 0.00300004, -0.025949842, 0.5, 0.5, 0.5, 0.5)
  )
  np.testing.assert_allclose(scene.palm_pos, expected[:3], atol=1e-8)
  np.testing.assert_allclose(scene.palm_quat, expected[3:], atol=1e-8)
  assert scene.model.nu == 20
  assert scene.model.nq == (47 if ghost else 27)
  if ghost:
    np.testing.assert_allclose(
      scene.data.body("ghost_right_palm_link").xpos, scene.palm_pos
    )
    np.testing.assert_allclose(scene.data.qpos[20:40], scene.data.qpos[:20])


def test_pen_ppo_distribution_resolves_without_restoring_removed_shared_module():
  cfg = dict(wuji_hand2_pen_spin_ppo_runner_cfg().actor.distribution_cfg)
  module, name = cfg.pop("class_name").split(":")
  distribution = getattr(importlib.import_module(module), name)(20, **cfg)
  distribution.update(torch.zeros((2, 2, 20)))
  assert torch.isfinite(distribution._distribution.stddev).all()
