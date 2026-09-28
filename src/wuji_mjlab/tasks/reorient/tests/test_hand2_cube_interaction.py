# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from mjlab.managers.metrics_manager import MetricsManager, MetricsTermCfg
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from wuji_mjlab.tasks.reorient.config.wuji_hand2.env_cfgs import (
  wuji_hand2_reorient_env_cfg,
)
from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
  REORIENT_HAND2_CUBE_INIT_POS,
  REORIENT_HAND2_JOINT_POS,
  REORIENT_HAND2_ROOT_POS,
  REORIENT_HAND2_ROOT_ROT,
)
from wuji_mjlab.tasks.reorient.mdp.metrics import FingertipContactCount
from wuji_mjlab.tasks.reorient.mdp.rewards import TipSlidePenalty
from wuji_mjlab.tasks.reorient.tooling.cage_pose_check import (
  build_model,
  set_hand_pose,
)

SIDES = ("right",)

_STRUCT_SIDES = ["right"]

_CAGE_JOINT_POS = {
  key.removeprefix(".*_"): value for key, value in REORIENT_HAND2_JOINT_POS.items()
}


def _build(side: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
  model = build_model(
    "wuji_hand2",
    side,
    REORIENT_HAND2_ROOT_POS,
    REORIENT_HAND2_ROOT_ROT,
    REORIENT_HAND2_CUBE_INIT_POS[side],
  )
  data = mujoco.MjData(model)
  set_hand_pose(model, data, side, _CAGE_JOINT_POS)
  cube_adr = model.joint("cube_free").qposadr[0]
  data.qpos[cube_adr : cube_adr + 3] = REORIENT_HAND2_CUBE_INIT_POS[side]
  mujoco.mj_forward(model, data)
  return model, data


def _gid(model: mujoco.MjModel, name: str) -> int:
  gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
  assert gid >= 0, f"geom {name!r} not found"
  return gid


def _can_collide(model: mujoco.MjModel, a: int, b: int) -> bool:
  return bool(
    (model.geom_contype[a] & model.geom_conaffinity[b])
    or (model.geom_contype[b] & model.geom_conaffinity[a])
  )


def _geom_body(model: mujoco.MjModel, gid: int) -> str:
  return model.body(model.geom_bodyid[gid]).name


def _excluded_body_pairs(model: mujoco.MjModel) -> set[frozenset[str]]:
  pairs: set[frozenset[str]] = set()
  for i in range(model.nexclude):
    sig = int(model.exclude_signature[i])
    b1, b2 = sig >> 16, sig & 0xFFFF
    pairs.add(frozenset({model.body(b1).name, model.body(b2).name}))
  return pairs


def _step_collecting_contacts(
  model: mujoco.MjModel, data: mujoco.MjData, n_steps: int
) -> set[frozenset[str]]:
  seen: set[frozenset[str]] = set()
  for _ in range(n_steps):
    mujoco.mj_step(model, data)
    for i in range(data.ncon):
      c = data.contact[i]
      seen.add(frozenset({_geom_body(model, c.geom1), _geom_body(model, c.geom2)}))
  return seen


def test_reorient_home_pose_has_no_self_contacts() -> None:
  cfg = wuji_hand2_reorient_env_cfg(num_envs=1)
  model = cfg.scene.entities["robot"].spec_fn().compile()
  data = mujoco.MjData(model)
  set_hand_pose(model, data, "right", _CAGE_JOINT_POS)
  mujoco.mj_forward(model, data)
  assert data.ncon == 0


@pytest.mark.parametrize("side", SIDES)
def test_cube_can_collide_with_hand(side: str) -> None:
  model, _ = _build(side)
  cube = _gid(model, "cube")

  assert model.geom_contype[cube] != 0 and model.geom_conaffinity[cube] != 0

  assert _can_collide(model, cube, _gid(model, f"{side}_palm_collision")), (
    "cube cannot collide with the palm"
  )
  for n in range(1, 6):
    for link in (2, 3, 4):
      g = _gid(model, f"{side}_finger{n}_link{link}_col")
      assert _can_collide(model, cube, g), (
        f"cube cannot collide with {side}_finger{n}_link{link}_col"
      )


@pytest.mark.parametrize("side", _STRUCT_SIDES)
def test_mount_collision_is_disabled(side: str) -> None:
  model, _ = _build(side)
  mount = _gid(model, "r_mount_visual")
  assert model.geom_contype[mount] == 0 and model.geom_conaffinity[mount] == 0
  assert not _can_collide(model, _gid(model, "cube"), mount)


@pytest.mark.parametrize("side", _STRUCT_SIDES)
def test_exclusion_set_is_minimal(side: str) -> None:
  model, _ = _build(side)
  excluded = _excluded_body_pairs(model)

  intended = frozenset({f"{side}_palm_link", f"{side}_finger1_link2"})
  assert excluded == {intended}, f"unexpected exclusion set for {side}: {excluded}"
  assert all("cube" not in pair for pair in excluded)

  for link in (3, 4):
    assert (
      frozenset({f"{side}_palm_link", f"{side}_finger1_link{link}"}) not in excluded
    )


@pytest.mark.parametrize("side", SIDES)
def test_cube_is_supported_by_hand_contact(side: str) -> None:
  model, data = _build(side)
  palm_z = data.body(f"{side}_palm_link").xpos[2]
  cube_adr = model.joint("cube_free").qposadr[0]

  n_steps = int(1.5 / model.opt.timestep)
  seen = _step_collecting_contacts(model, data, n_steps)

  cube_hand_contacts = [p for p in seen if "cube" in p and any("cube" != b for b in p)]
  assert cube_hand_contacts, "cube never contacted the hand while settling"

  cube_z = float(data.qpos[cube_adr + 2])
  assert cube_z >= palm_z - 0.05, "cube dropped out of the cage"


@pytest.mark.parametrize("side", SIDES)
def test_cube_never_contacts_the_mount(side: str) -> None:
  model, data = _build(side)
  n_steps = int(1.5 / model.opt.timestep)
  seen = _step_collecting_contacts(model, data, n_steps)
  offending = [pair for pair in seen if "cube" in pair and "r_mount" in pair]
  assert not offending, f"cube contacted the mount: {offending}"


@pytest.mark.parametrize("side", SIDES)
def test_thumb_cannot_tunnel_through_palm(side: str) -> None:
  model, data = _build(side)
  set_hand_pose(model, data, side, _CAGE_JOINT_POS)
  for joint, target in ((1, 1.29), (2, 1.29), (3, 1.6), (4, 1.6)):
    jnt = model.joint(f"{side}_finger1_joint{joint}")
    lo, hi = jnt.range
    value = float(np.clip(target, lo, hi))
    data.qpos[jnt.qposadr[0]] = value
    data.ctrl[model.actuator(f"{side}_finger1_joint{joint}_actuator").id] = value
  mujoco.mj_forward(model, data)

  seen = _step_collecting_contacts(model, data, 150)
  palm = f"{side}_palm_link"
  thumb_palm = [
    p
    for p in seen
    if palm in p
    and any(b in (f"{side}_finger1_link3", f"{side}_finger1_link4") for b in p)
  ]
  assert thumb_palm, (
    "thumb driven into the palm produced no distal-link<->palm contact "
    "— it would tunnel through the palm"
  )


def _tip_contact_env(found, force=None, num_sites=5):
  num_envs = (found if found is not None else force).shape[0]
  velocities = torch.zeros(num_envs, num_sites, 3)
  velocities[:, :, 0] = 2.0 ** torch.arange(num_sites) + 0.5
  num_geoms = (found if found is not None else force).shape[1]
  geom_names = (
    [f"right_finger{finger}_link4_col" for finger in range(1, 6)]
    if num_geoms == 5
    else [
      f"right_finger{finger}_{part}"
      for finger in range(1, 6)
      for part in ("link4_col", "tip_sensor_col")
    ][:num_geoms]
  )
  return SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    scene={
      "robot": SimpleNamespace(
        site_names=[f"right_finger{finger}_tip" for finger in range(1, num_sites + 1)],
        data=SimpleNamespace(site_lin_vel_w=velocities),
      ),
      "object": SimpleNamespace(
        data=SimpleNamespace(
          root_link_lin_vel_w=torch.tensor([[0.5, 0.0, 0.0]]).repeat(num_envs, 1)
        )
      ),
      "tip_object_contact": SimpleNamespace(
        primary_names=geom_names,
        cfg=SimpleNamespace(num_slots=1),
        data=SimpleNamespace(found=found, force=force),
      ),
    },
  )


def _tip_reward_manager(env, threshold=0.0):
  cfg = RewardTermCfg(
    func=TipSlidePenalty,
    weight=1.0,
    params={
      "robot_cfg": SceneEntityCfg("robot"),
      "contact_threshold": threshold,
    },
  )
  return RewardManager({"tip_slide": cfg}, env)


def _tip_count(env):
  manager = MetricsManager(
    {"tip_count": MetricsTermCfg(func=FingertipContactCount)}, env
  )
  manager.compute()
  return manager._step_values[:, 0]


@pytest.mark.parametrize("field", ["found", "force"])
@pytest.mark.parametrize("geoms_per_finger", [1, 2])
def test_tip_slide_pairs_each_contact_with_its_own_finger(field, geoms_per_finger):
  contacts = torch.eye(5 * geoms_per_finger)
  force = torch.zeros(*contacts.shape, 3)
  force[:, :, 1] = contacts
  env = _tip_contact_env(contacts if field == "found" else None, force)
  actual = _tip_reward_manager(env, threshold=0.5).compute(1.0)
  expected = (
    [1.0, 2.0, 4.0, 8.0, 16.0]
    if geoms_per_finger == 1
    else [1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 8.0, 8.0, 16.0, 16.0]
  )
  assert actual.tolist() == expected


@pytest.mark.parametrize("num_contacts,num_sites", [(6, 5), (9, 5), (10, 4), (10, 6)])
def test_tip_slide_rejects_unmatched_contact_or_site_shape(num_contacts, num_sites):
  env = _tip_contact_env(torch.ones(1, 10), num_sites=num_sites)
  env.scene["tip_object_contact"].data.found = torch.ones(1, num_contacts)
  with pytest.raises(ValueError):
    _tip_reward_manager(env).compute(1.0)


@pytest.mark.parametrize(
  "contacts,expected",
  [
    ([[1, 1, 0, 1, 0, 0, 1, 0, 1, 1]], [4.0]),
    ([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], [5.0]),
    ([[1, 0, 1, 0, 1]], [3.0]),
    ([[0, 0, 0, 0, 0]], [0.0]),
  ],
)
def test_fingertip_count_counts_fingers_not_geoms(contacts, expected):
  env = _tip_contact_env(torch.tensor(contacts))
  assert _tip_count(env).tolist() == expected


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_contact_binding_survives_permuted_geom_and_tip_site_order(device):
  if device.startswith("cuda") and not torch.cuda.is_available():
    pytest.skip("CUDA is unavailable")
  env = _tip_contact_env(torch.tensor([[1, 1, 0, 0, 1, 0, 0, 0, 1, 1]]))
  env.device = device
  sensor = env.scene["tip_object_contact"]
  geom_order = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
  sensor.primary_names = [sensor.primary_names[i] for i in geom_order]
  sensor.data.found = sensor.data.found[:, geom_order].to(device)
  robot = env.scene["robot"]
  site_order = [4, 0, 3, 1, 2]
  robot.site_names = [robot.site_names[i] for i in site_order]
  robot.data.site_lin_vel_w = robot.data.site_lin_vel_w[:, site_order].to(device)
  object_data = env.scene["object"].data
  object_data.root_link_lin_vel_w = object_data.root_link_lin_vel_w.to(device)
  assert _tip_reward_manager(env).compute(1.0).tolist() == [21.0]
  assert _tip_count(env).tolist() == [3.0]


@pytest.mark.parametrize(
  "invalid", ["unmatched_geom", "missing_finger", "duplicate_site"]
)
def test_contact_binding_rejects_invalid_names_at_initialization(invalid):
  env = _tip_contact_env(torch.ones(1, 5))
  sensor = env.scene["tip_object_contact"]
  robot = env.scene["robot"]
  if invalid == "unmatched_geom":
    sensor.primary_names[0] = "right_palm_collision"
  elif invalid == "missing_finger":
    sensor.primary_names[4] = "right_finger4_tip_sensor_col"
  else:
    robot.site_names[4] = "left_finger1_tip"
  with pytest.raises(ValueError) as exc:
    _tip_reward_manager(env)
  assert sensor.primary_names[0] in str(exc.value)
  assert robot.site_names[0] in str(exc.value)
