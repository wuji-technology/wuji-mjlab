# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import math
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.play import PlayConfig
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from wuji_mjlab.rl.runner import WujiOnPolicyRunner
from wuji_mjlab.tasks.reorient.config.wuji_hand2 import _reorient_hand2_50hz_env_cfg
from wuji_mjlab.tasks.reorient.mdp.cage import update_cage_penalty_counter
from wuji_mjlab.tasks.reorient.mdp.curriculums import get_linear_progress
from wuji_mjlab.tasks.reorient.mdp.runtime_state import get_reorient_runtime_state
from wuji_mjlab.tasks.reorient.scripts import view_task
from wuji_mjlab.utils import play_runner


def test_50hz_cage_drop_is_reachable():
  cfg = _reorient_hand2_50hz_env_cfg(num_envs=8)
  drop_threshold = cfg.terminations["cage_drop"].params["max_outside_steps"]
  counter_ceiling = cfg.rewards["cage_escape"].params["max_count"]
  assert counter_ceiling >= drop_threshold


def test_cage_counter_respects_configurable_max_count():
  counter = torch.full((1,), 20.0)
  outside = torch.ones(1, dtype=torch.bool)
  out = update_cage_penalty_counter(counter, outside, decay_rate=0.5, max_count=37.5)
  assert out.item() == 21.0
  capped = update_cage_penalty_counter(
    torch.full((1,), 37.5), outside, decay_rate=0.5, max_count=37.5
  )
  assert capped.item() == 37.5


def test_cage_penalty_rejects_ceiling_below_threshold():
  import pytest
  from wuji_mjlab.tasks.reorient.mdp.cage import CageEscapePenalty

  class _Cfg:
    weight = -500.0
    params = {"max_outside_steps": 25, "max_count": 15.0}

  class _Env:
    num_envs = 4
    device = "cpu"

  with pytest.raises(ValueError, match="max_count"):
    CageEscapePenalty(_Cfg(), _Env())


@pytest.mark.parametrize(
  "task",
  ["WujiHand_Reorient", "WujiHand2_Reorient", "WujiHand2_Reorient_50Hz"],
)
def test_disturbance_ramp_reaches_full_amplitude_at_runner_budget(monkeypatch, task):
  for name in ("WANDB_PROJECT", "WANDB_API_KEY"):
    monkeypatch.delenv(name, raising=False)
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 2
  cfg.commands["reorient_command"].debug_vis = False
  rl_cfg = load_rl_cfg(task)
  event = load_env_cfg(task).events["object_disturbance_force"]
  rampup = event.params["rampup_frac"]
  warmup = event.params["warmup_frac"]
  full_iteration = math.ceil(rl_cfg.max_iterations * rampup)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
  try:
    runner = WujiOnPolicyRunner(
      RslRlVecEnvWrapper(env), asdict(rl_cfg), log_dir=None, device="cpu"
    )
    assert runner.logger.writer is None
    policy = runner.get_inference_policy(device="cpu")
    with torch.inference_mode():
      assert torch.isfinite(policy(runner.env.get_observations())).all()
    env.common_step_counter = (full_iteration - 1) * rl_cfg.num_steps_per_env
    assert get_linear_progress(env, warmup, rampup) < 1.0
    env.common_step_counter = full_iteration * rl_cfg.num_steps_per_env
    assert get_linear_progress(env, warmup, rampup) == 1.0
  finally:
    env.close()


@pytest.mark.parametrize("entry", ["view", "play"])
@pytest.mark.parametrize(
  "task",
  ["WujiHand_Reorient", "WujiHand2_Reorient", "WujiHand2_Reorient_50Hz"],
)
@pytest.mark.parametrize("budget", [None, (1200, 80)], ids=["default", "custom"])
def test_standalone_viewers_execute_disturbance_with_task_budget(
  monkeypatch, entry, task, budget
):
  def load_test_rl_cfg(task_id):
    cfg = load_rl_cfg(task_id)
    if budget is not None:
      cfg.max_iterations, cfg.num_steps_per_env = budget
    return cfg

  rl_cfg = load_test_rl_cfg(task)
  params = load_env_cfg(task).events["object_disturbance_force"].params
  warmup = params["warmup_frac"]
  rampup = params["rampup_frac"]
  total_steps = rl_cfg.max_iterations * rl_cfg.num_steps_per_env
  half_ramp_step = round(total_steps * (warmup + rampup) / 2)
  min_speed = params["min_speed"]
  half_ramp_max_speed = min_speed + (params["max_speed"] - min_speed) / 2
  monkeypatch.setattr(view_task, "load_rl_cfg", load_test_rl_cfg)
  environments = []

  def viewer(env, policy):
    environments.append(env)

    def run():
      raw = env.unwrapped
      raw.event_manager.apply(mode="interval", dt=2.0)
      raw.common_step_counter = half_ramp_step
      assert get_linear_progress(raw, warmup, rampup) == pytest.approx(0.5)
      raw.episode_length_buf.fill_(math.ceil(params["warmup_time_s"] / raw.step_dt))
      raw.event_manager.apply(mode="interval", dt=2.0)
      state = get_reorient_runtime_state(raw)
      speed = state.pert_velocity_cache[:, :3].norm(dim=-1)
      assert torch.all(speed >= min_speed - 1e-6)
      assert torch.all(speed <= half_ramp_max_speed + 1e-6)
      obs, reward, _, _ = env.step(policy(env.get_observations()))
      assert torch.isfinite(obs["policy"]).all()
      assert torch.isfinite(reward).all()

    return SimpleNamespace(run=run)

  monkeypatch.setattr(view_task, "ViserPlayViewer", viewer)
  monkeypatch.setattr(play_runner, "ViserPlayViewer", viewer)
  try:
    if entry == "view":
      view_task.run_view(
        task, view_task.ViewTaskConfig(device="cpu", viewer="viser", num_envs=1)
      )
    else:
      cfg = load_env_cfg(task, play=False)
      play_runner.run_play_with_cfg(
        task,
        PlayConfig(agent="zero", device="cpu", viewer="viser", num_envs=1),
        cfg,
        load_test_rl_cfg(task),
      )
  finally:
    for env in environments:
      env.close()
