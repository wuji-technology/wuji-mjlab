# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Disturbance-only compensation; model mass and mass DR stay unchanged."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from mjlab.entity import Entity
from wuji_mjlab.tasks.pen_spin.config.wuji_hand2.env_cfgs import (
  wuji_hand2_pen_spin_env_cfg,
)
from wuji_mjlab.tasks.pen_spin.config.wuji_hand2.hand2_constants import (
  get_pen_spin_hand2_cfg,
)
from wuji_mjlab.tasks.pen_spin.mdp.events import apply_link_random_force

# Compiled values from the sole reference, e8f142b (including visual geoms in R).
LEGACY_MASS = (0.0159, 0.0078, 0.0078, 0.0078, 0.0078)
LEGACY_RADIUS = (
  0.024492116024825473,
  0.021780775788495535,
  0.021780775788495535,
  0.021780775788495542,
  0.021780776075749575,
)


def make_env(*, legacy=False, n=256, order=(4, 1, 3, 0, 2)):
  cfg = copy.deepcopy(wuji_hand2_pen_spin_env_cfg().events["fingertip_random_force"])
  model = Entity(get_pen_spin_hand2_cfg()).spec.compile()
  names = [f"right_finger{i}_link4" for i in range(1, 6)]
  ids = [model.body(name).id for name in names]
  mass = np.array(model.body_mass)
  if legacy:
    mass[ids] = LEGACY_MASS
    for bid, radius in zip(ids, LEGACY_RADIUS, strict=True):
      model.geom_rbound[model.geom_bodyid == bid] = radius
    cfg.params.pop("force_scale")
    cfg.params.pop("offset_radius_scale")
  # Same per-body density multipliers, including both ends of the existing DR.
  density = torch.exp(2 * torch.linspace(-0.458, 0.203, n)).reshape(n, 1)
  masses = torch.tensor(mass, dtype=torch.float32).expand(n, -1).clone()
  masses[:, ids] *= density
  body_quat = torch.zeros(n, 5, 4)
  body_quat[..., 0] = 1.0
  robot = SimpleNamespace(
    num_bodies=5,
    body_names=names,
    indexing=SimpleNamespace(body_ids=torch.tensor(ids)),
    data=SimpleNamespace(body_com_quat_w=body_quat),
    write_external_wrench_to_sim=Mock(),
  )
  cfg.params["asset_cfg"].body_ids = list(order)
  env = SimpleNamespace(
    scene={"robot": robot},
    num_envs=n,
    device="cpu",
    step_dt=0.02,
    sim=SimpleNamespace(
      mj_model=model,
      model=SimpleNamespace(
        body_mass=masses, opt=SimpleNamespace(gravity=(0, 0, -9.81))
      ),
    ),
    curriculum_manager=SimpleNamespace(
      _curriculum_state={"disturbance_success_rate": 0.6}
    ),
  )
  return cfg, env


def run_wrench(cfg, env, event_cls=apply_link_random_force):
  torch.manual_seed(142)
  event = event_cls(cfg, env)
  forces, torques, active = [], [], []
  # Include expiry, cooldown, reset and curriculum changes.
  for step in range(100):
    if step == 20:
      env.curriculum_manager._curriculum_state["disturbance_success_rate"] = 1.0
    if step == 60:
      event.reset(torch.arange(8))
    event(env, None, **cfg.params)
    args = env.scene["robot"].write_external_wrench_to_sim.call_args.args
    forces.append(args[0].clone())
    torques.append(args[1].clone())
    active.append(event._active.clone())
  return torch.stack(forces), torch.stack(torques), torch.stack(active)


def test_rounded_wrenches_stay_near_legacy_without_mass_writes():
  old_cfg, old_env = make_env(legacy=True)
  cfg, env = make_env()
  original_mass = env.sim.model.body_mass.clone()
  before = {
    name: np.array(getattr(env.sim.mj_model, name))
    for name in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "geom_rbound")
  }
  expected = run_wrench(old_cfg, old_env)
  actual = run_wrench(cfg, env)
  # Rounded coefficients deliberately trade exact equality for simple tuning:
  # <=4% force and <=6% additional torque deviation from the legacy baseline.
  torch.testing.assert_close(actual[0], expected[0], atol=2e-7, rtol=0.04)
  torch.testing.assert_close(actual[1], expected[1], atol=1e-8, rtol=0.06)
  assert torch.equal(actual[2], expected[2])
  assert torch.equal(env.sim.model.body_mass, original_mass)
  for name, value in before.items():
    np.testing.assert_array_equal(getattr(env.sim.mj_model, name), value)


@pytest.mark.parametrize("field", ["force_scale", "offset_radius_scale"])
@pytest.mark.parametrize("bad_value", [-1.0, float("nan"), float("inf")])
def test_invalid_coefficients_fail_before_training(field, bad_value):
  cfg, env = make_env(n=8)
  cfg.params[field]["right_finger1_link4"] = bad_value
  with pytest.raises(ValueError, match="finite nonnegative"):
    apply_link_random_force(cfg, env)


def test_misspelled_body_coefficient_is_rejected():
  cfg, env = make_env(n=8)
  cfg.params["force_scale"]["right_finger1_link3"] = 1.2
  with pytest.raises(ValueError, match="unselected bodies"):
    apply_link_random_force(cfg, env)


def test_mass_dr_and_play_configuration_remain_unchanged():
  cfg = wuji_hand2_pen_spin_env_cfg()
  mass_dr = cfg.events["robot_link_inertial"].params
  assert mass_dr["alpha_range"] == (-0.458, 0.203)
  assert mass_dr["d_range"] == (-0.2, 0.2)
  assert len(mass_dr["asset_cfg"].body_names) == 21
  assert "fingertip_random_force" not in wuji_hand2_pen_spin_env_cfg(play=True).events
