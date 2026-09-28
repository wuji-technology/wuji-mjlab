# SPDX-License-Identifier: Apache-2.0
"""Exercise the real wrapper, action manager, residual filter, and reward chain."""

import json
from types import SimpleNamespace

import torch
from mjlab.envs.mdp.observations import last_action
from mjlab.managers.action_manager import ActionManager
from mjlab.rl import RslRlVecEnvWrapper
from wuji_mjlab.rl.action_diagnostics import ActionDiagnosticsWrapper
from wuji_mjlab.tasks.pen_spin.mdp.actions import SingleHandResidualEMAAction
from wuji_mjlab.tasks.pen_spin.mdp.rewards import action_rate_combined


class BoundaryEnv:
  """Omit physics; retain the production action/reward/observation interface."""

  def __init__(self):
    self.unwrapped = self
    self.num_envs, self.device, self.max_episode_length = 4, "cpu", 100
    self.cfg = SimpleNamespace(is_finite_horizon=False)
    self.motion = SimpleNamespace(
      joint_pos=torch.zeros(4, 20),
      trajectory_ids=torch.arange(4),
      time_steps=torch.zeros(4, dtype=torch.long),
    )
    self.command_manager = SimpleNamespace(get_term=lambda _: self.motion)
    self.observation_manager = SimpleNamespace(compute=self.observe)
    term = SingleHandResidualEMAAction.__new__(SingleHandResidualEMAAction)
    term._env, term._action_dim, term._target_ids = self, 20, slice(None)
    term._entity = SimpleNamespace(
      data=SimpleNamespace(encoder_bias=torch.zeros(4, 20))
    )
    term.cfg = SimpleNamespace(
      alpha=0.5, residual_clip=(-0.6, 0.6), command_name="motion"
    )
    term._scale, term._mask_enabled = 0.5, False
    for name in ("_raw_actions", "_processed_actions", "_targets"):
      setattr(term, name, torch.zeros(4, 20))
    term._targets_valid = torch.zeros(4, dtype=torch.bool)
    manager = ActionManager.__new__(ActionManager)
    manager._env, manager._terms = self, {"joint_pos": term}
    for name in ("_action", "_prev_action", "_prev_prev_action"):
      setattr(manager, name, torch.zeros(4, 20))
    self.action_manager = manager

  def observe(self):
    return {"policy": last_action(self, "joint_pos").clone()}

  def reset(self):
    return self.observe(), {}

  def step(self, action):
    self.action_manager.process_action(action)
    self.motion.time_steps += 1
    reward = -0.1 * 0.02 * action_rate_combined(self)
    done = torch.zeros(4, dtype=torch.bool)
    return self.observe(), reward, done, done, {"log": {"original": torch.tensor(3.0)}}


def test_large_policy_outputs_keep_targets_but_bound_history_and_reward():
  old, fixed = BoundaryEnv(), BoundaryEnv()
  baseline = RslRlVecEnvWrapper(old, clip_actions=None)
  wrapped = RslRlVecEnvWrapper(fixed, clip_actions=1.2)
  torch.manual_seed(12)
  actions = torch.randn(12, 4, 20) * 5
  actions[3:6] = torch.tensor([1e6, -1e6, 1e6])[:, None, None]
  old_peak = 0.0
  for raw in actions:
    original = raw.clone()
    _, old_reward, _, _ = baseline.step(raw)
    obs, reward, _, _ = wrapped.step(raw)
    # PPO's stored sample/log probability must still refer to the original sample.
    torch.testing.assert_close(raw, original, rtol=0, atol=0)
    torch.testing.assert_close(
      old.action_manager.get_term("joint_pos")._targets,
      fixed.action_manager.get_term("joint_pos")._targets,
      rtol=0,
      atol=0,
    )
    assert obs["policy"].abs().max() <= 1.2
    assert fixed.action_manager.prev_action.abs().max() <= 1.2
    assert fixed.action_manager.prev_prev_action.abs().max() <= 1.2
    assert reward.min() >= -1.15201
    old_peak = max(old_peak, float(old_reward.abs().max()))
  assert old_peak > 1e10


def test_diagnostics_retain_rollout_peak_and_outlier_input(tmp_path):
  env = BoundaryEnv()
  wrapped = ActionDiagnosticsWrapper(env, 1.2, rollout_steps=3, output_dir=tmp_path)
  wrapped.get_observations()
  for value in (0.0, 1e6, -0.3):
    _, _, _, extras = wrapped.step(torch.full((4, 20), value))
  assert extras["log"]["original"] == 3
  assert extras["log"]["ActionBoundary/raw_action_max"] == 1e6
  assert extras["log"]["ActionBoundary/applied_action_max"] <= 1.2
  assert extras["log"]["ActionBoundary/action_rate_combined_max"] <= 576.001
  row = json.loads((tmp_path / "rank0.jsonl").read_text())
  assert row["raw_action_max"] == 1e6
  outlier = torch.load(tmp_path / "outlier_rank0_iter0.pt", weights_only=True)
  torch.testing.assert_close(outlier["inputs"][1]["raw_action"], torch.full((20,), 1e6))
  assert "policy_obs" in outlier["inputs"][1]


def test_rank_is_available_before_runner_initializes_nccl(tmp_path, monkeypatch):
  monkeypatch.setenv("RANK", "1")
  wrapped = ActionDiagnosticsWrapper(
    BoundaryEnv(), 1.2, rollout_steps=1, output_dir=tmp_path
  )
  wrapped.step(torch.zeros(4, 20))
  assert json.loads((tmp_path / "rank1.jsonl").read_text())["rank"] == 1
  assert not (tmp_path / "rank0.jsonl").exists()
