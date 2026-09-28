# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import re
from types import SimpleNamespace

import torch
from wuji_mjlab.tasks.reorient.mdp.commands import (
  InHandReorientCommand,
  InHandReorientCommandCfg,
)


class TestCommandSuccessCounting:
  def test_within_threshold_for_hold_steps_increments_goal_reach_count(self):
    cmd = _make_reorient_command(num_envs=1)
    _align_object_with_goal(cmd)

    success_hold_steps = cmd.cfg.success_hold_steps
    assert cmd.goal_reach_count.item() == 0

    for tick in range(success_hold_steps):
      cmd._update_command()
      if tick < success_hold_steps - 1:
        assert cmd.goal_reach_count.item() == 0, (
          f"goal_reach_count incremented prematurely at tick {tick}"
        )
        assert not cmd.success_achieved.any()

    assert cmd.goal_reach_count.item() == 1, (
      "goal_reach_count must increment by 1 after success_hold_steps "
      "consecutive within-threshold ticks"
    )
    assert cmd.success_achieved.all(), "success_achieved must fire on the success step"

  def test_goal_switches_after_goal_switch_delay_in_success_window(self):
    cmd = _make_reorient_command(num_envs=1)
    _align_object_with_goal(cmd)

    success_hold_steps = cmd.cfg.success_hold_steps
    goal_switch_delay = cmd.cfg.goal_switch_delay

    for _ in range(success_hold_steps):
      cmd._update_command()
    assert cmd.in_success_window.all()
    goal_before_switch = cmd.goal_quat_w.clone()

    extra_ticks_until_switch = goal_switch_delay - 1
    for _ in range(extra_ticks_until_switch):
      cmd._update_command()

    assert cmd.goal_switched.all(), (
      "goal_switched must fire on the step window_timer reaches goal_switch_delay"
    )
    assert not torch.allclose(cmd.goal_quat_w, goal_before_switch, atol=1e-3)
    assert not cmd.in_success_window.any()
    assert cmd.window_timer.item() == 0

  def test_two_phase_state_machine_progression(self):
    cmd = _make_reorient_command(num_envs=2)
    _align_object_with_goal(cmd)

    assert not cmd.in_success_window.any(), "must start in approaching phase"

    for _ in range(cmd.cfg.success_hold_steps):
      cmd._update_command()

    assert cmd.in_success_window.all(), (
      "must transition to success window after holding within threshold"
    )

  def test_goal_reach_count_not_incremented_while_in_success_window(self):
    cmd = _make_reorient_command(num_envs=1)
    _align_object_with_goal(cmd)

    for _ in range(cmd.cfg.success_hold_steps):
      cmd._update_command()
    assert cmd.goal_reach_count.item() == 1
    assert cmd.in_success_window.all()

    extra_ticks = max(1, cmd.cfg.goal_switch_delay // 2)
    for _ in range(extra_ticks):
      cmd._update_command()

    assert cmd.goal_reach_count.item() == 1, (
      "goal_reach_count must not increment while already in success window"
    )


class _FakeEntity:
  def __init__(
    self, body_names, pose_w, root_quat_w=None, site_names=None, site_pose_w=None
  ):
    self._body_names = list(body_names)
    self.data = SimpleNamespace(body_link_pose_w=pose_w)
    if root_quat_w is not None:
      self.data.root_link_quat_w = root_quat_w
      self.data.root_link_pos_w = torch.zeros(root_quat_w.shape[0], 3)
    self._site_names = list(site_names) if site_names is not None else []
    if site_pose_w is not None:
      self.data.site_pose_w = site_pose_w

  def find_bodies(self, pattern, preserve_order: bool = False):
    patterns = pattern if isinstance(pattern, (tuple, list)) else (pattern,)
    regexes = [re.compile(p) for p in patterns]
    ids = [
      i for i, n in enumerate(self._body_names) if any(r.fullmatch(n) for r in regexes)
    ]
    return ids, [self._body_names[i] for i in ids]

  def find_sites(self, pattern, preserve_order: bool = False):
    patterns = pattern if isinstance(pattern, (tuple, list)) else (pattern,)
    regexes = [re.compile(p) for p in patterns]
    ids = [
      i for i, n in enumerate(self._site_names) if any(r.fullmatch(n) for r in regexes)
    ]
    return ids, [self._site_names[i] for i in ids]


def _identity_quat(num_envs: int) -> torch.Tensor:
  q = torch.zeros(num_envs, 4)
  q[:, 0] = 1.0
  return q


def _make_reorient_command(
  num_envs: int,
  success_hold_steps: int = 3,
  goal_switch_delay: int = 4,
) -> InHandReorientCommand:
  body_names = ["robot_palm_link", "finger1"]
  pose_w = torch.zeros(num_envs, len(body_names), 7)
  pose_w[:, :, 3] = 1.0

  site_pose_w = torch.zeros(num_envs, 1, 7)
  site_pose_w[:, :, 3] = 1.0
  robot = _FakeEntity(
    body_names=body_names,
    pose_w=pose_w,
    site_names=["robot_wrist_tag"],
    site_pose_w=site_pose_w,
  )
  obj = _FakeEntity(
    body_names=["cube"],
    pose_w=torch.zeros(num_envs, 1, 7),
    root_quat_w=_identity_quat(num_envs),
  )

  scene = {"robot": robot, "object": obj}
  env = SimpleNamespace(num_envs=num_envs, device="cpu", scene=scene, step_dt=0.02)

  cfg = InHandReorientCommandCfg(
    resampling_time_range=(1.0, 1.0),
    entity_name="object",
    robot_entity_name="robot",
    palm_body_pattern=".*_palm_link",
    success_threshold=0.5,
    success_hold_steps=success_hold_steps,
    goal_switch_delay=goal_switch_delay,
    min_goal_interval=0.0,
  )
  return InHandReorientCommand(cfg, env)


def _align_object_with_goal(cmd: InHandReorientCommand) -> None:
  cmd.object.data.root_link_quat_w = cmd.goal_quat_w.clone()
