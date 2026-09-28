# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import math

import pytest
import torch
from wuji_mjlab.tasks.reorient.mdp.actions import warmup_steps as sim_warmup_steps
from wuji_reorient_deploy.obs_builder import (
  warmup_steps as deploy_warmup_steps,
)

WARMUP_S = 0.4
DT_20, DT_50 = 0.05, 0.02


def _float_compare_disagrees(warmup_s: float, dt: float, n: int) -> bool:
  sim = bool(((torch.tensor([n], dtype=torch.int64) * dt) < warmup_s).item())
  deploy = (n * dt) < warmup_s
  return sim != deploy


class TestTheOldFloatComparison:
  def test_twenty_hz_agreed_only_by_luck(self):
    assert not _float_compare_disagrees(WARMUP_S, DT_20, 8)

  def test_fifty_hz_disagreed_at_the_boundary_step(self):
    assert _float_compare_disagrees(WARMUP_S, DT_50, 20)

  def test_the_disagreement_is_exactly_one_step_wide(self):
    off = [n for n in range(40) if _float_compare_disagrees(WARMUP_S, DT_50, n)]
    assert off == [20]


class TestIntegerStepCount:
  def test_sim_and_deploy_agree_at_the_two_rates_we_ship(self):
    assert (
      sim_warmup_steps(WARMUP_S, DT_20) == deploy_warmup_steps(WARMUP_S, DT_20) == 8
    )
    assert (
      sim_warmup_steps(WARMUP_S, DT_50) == deploy_warmup_steps(WARMUP_S, DT_50) == 20
    )

  @pytest.mark.parametrize("warmup_s", [0.0, 0.2, 0.4, 0.41, 1.0, 3.0])
  @pytest.mark.parametrize("dt", [0.05, 0.04, 0.02, 0.01, 0.005, 0.003])
  def test_sim_and_deploy_agree_everywhere(self, warmup_s, dt):
    assert sim_warmup_steps(warmup_s, dt) == deploy_warmup_steps(warmup_s, dt)

  @pytest.mark.parametrize("dt", [0.05, 0.02, 0.01, 0.003])
  def test_hold_covers_at_least_the_requested_time(self, dt):
    n = sim_warmup_steps(WARMUP_S, dt)
    assert n * dt >= WARMUP_S - 1e-9
    assert (n - 1) * dt < WARMUP_S

  def test_a_non_integer_ratio_rounds_up(self):
    assert sim_warmup_steps(0.4, 0.03) == math.ceil(0.4 / 0.03)

  def test_zero_warmup_holds_nothing(self):
    assert sim_warmup_steps(0.0, 0.02) == deploy_warmup_steps(0.0, 0.02) == 0

  @pytest.mark.parametrize("dt", [0.0, -0.01])
  def test_nonpositive_dt_is_rejected(self, dt):
    with pytest.raises(ValueError):
      sim_warmup_steps(WARMUP_S, dt)
    with pytest.raises(ValueError):
      deploy_warmup_steps(WARMUP_S, dt)


class TestTwentyHzBehaviourUnchanged:
  @pytest.mark.parametrize("n", list(range(12)))
  def test_step_by_step_match_against_the_old_float_rule(self, n):
    old = bool(((torch.tensor([n], dtype=torch.int64) * DT_20) < WARMUP_S).item())
    new = n < sim_warmup_steps(WARMUP_S, DT_20)
    assert new == old
