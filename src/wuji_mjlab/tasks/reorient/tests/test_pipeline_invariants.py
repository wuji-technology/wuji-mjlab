# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Pipeline invariant tests for the reorient task.

These tests verify critical pipeline invariants via config inspection and
behavior-level assertions (driving real task terms with fake assets) without
spinning up a MuJoCo environment.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch
from wuji_mjlab.tasks.reorient.config.wuji_hand.env_cfgs import (
  wuji_hand_reorient_env_cfg,
)
from wuji_mjlab.tasks.reorient.mdp.commands import (
  InHandReorientCommand,
  InHandReorientCommandCfg,
)
from wuji_mjlab.tasks.reorient.reorient_env_cfg import make_reorient_env_cfg
from wuji_mjlab.tasks.reorient.tests.fakes import make_command_manager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _train_cfg():
  return make_reorient_env_cfg()


def _play_cfg():
  return wuji_hand_reorient_env_cfg(play=True)


# ---------------------------------------------------------------------------
# 1. Disturbance scheduling: interval-based, not every-step
# ---------------------------------------------------------------------------


@pytest.mark.pipeline
class TestDisturbanceScheduling:
  """Disturbance events must use interval-based scheduling so forces are
  applied intermittently, not on every simulation step."""

  def test_object_disturbance_force_uses_interval_mode(self):
    cfg = _train_cfg()
    event = cfg.events["object_disturbance_force"]
    assert event.mode == "interval", (
      f"object_disturbance_force must use 'interval' mode, got '{event.mode}'"
    )

  def test_object_disturbance_force_has_nonzero_interval(self):
    cfg = _train_cfg()
    event = cfg.events["object_disturbance_force"]
    lo, hi = event.interval_range_s
    assert lo > 0 and hi > 0, f"interval_range_s must be positive, got ({lo}, {hi})"
    assert hi >= lo, "interval upper bound must be >= lower bound"

  def test_no_disturbance_events_use_step_mode(self):
    """No event with 'disturbance' in its name should run every step."""
    cfg = _train_cfg()
    for name, event in cfg.events.items():
      if "disturbance" in name and "reset" not in name:
        assert event.mode != "step", (
          f"Event '{name}' should not use 'step' mode for disturbances"
        )


# ---------------------------------------------------------------------------
# 2. Play config overrides: DR events disabled
# ---------------------------------------------------------------------------


@pytest.mark.pipeline
class TestPlayConfigOverrides:
  """Play (evaluation) config must disable domain randomization so that
  policy evaluation uses nominal physics parameters."""

  # DR event keys that must be removed in play mode
  _DR_EVENTS = [
    "object_com",
    "object_friction",
    "robot_friction",
    "robot_geom_size",
    "contact_params",
    "object_size",
    "object_mass",
    "pd_gains",
    "robot_dof_damping",
    "robot_dof_armature",
    "robot_dof_frictionloss",
    "encoder_bias",
    "robot_link_inertia",
    "robot_link_mass",
  ]

  def test_dr_events_removed_in_play_config(self):
    cfg = _play_cfg()
    remaining = [k for k in self._DR_EVENTS if k in cfg.events]
    assert remaining == [], f"Play config still contains DR events: {remaining}"

  def test_disturbance_events_removed_in_play_config(self):
    cfg = _play_cfg()
    remaining = [k for k in cfg.events if "disturbance" in k and "reset" not in k]
    assert remaining == [], (
      f"Play config still contains disturbance events: {remaining}"
    )

  def test_observation_corruption_disabled_in_play_config(self):
    cfg = _play_cfg()
    policy_group = cfg.observations["policy"]
    assert policy_group.enable_corruption is False, (
      "Play config must disable observation corruption for policy group"
    )

  def test_curriculum_cleared_in_play_config(self):
    cfg = _play_cfg()
    assert len(cfg.curriculum) == 0, (
      f"Play config must have empty curriculum, got {list(cfg.curriculum.keys())}"
    )

  def test_train_config_has_dr_events(self):
    """Sanity check: training config should contain the DR events."""
    cfg = _train_cfg()
    present = [k for k in self._DR_EVENTS if k in cfg.events]
    assert len(present) >= 10, (
      f"Training config should have most DR events, only found {len(present)}"
    )


# ---------------------------------------------------------------------------
# 3. Observation encoding: goal orientation uses 6D representation
# ---------------------------------------------------------------------------


@pytest.mark.pipeline
class TestObservationEncoding:
  """Goal orientation error must use 6D rotation encoding, not scalar or
  quaternion, to avoid representation discontinuities."""

  def test_cube_ori_error_uses_6d_function(self):
    cfg = _train_cfg()
    policy_terms = cfg.observations["policy"].terms
    ori_term = policy_terms["cube_ori_error"]
    func_name = ori_term.func.__name__
    assert func_name == "goal_rot_err_6d", (
      f"cube_ori_error must use goal_rot_err_6d, got '{func_name}'"
    )

  def test_critic_ori_error_uses_6d_function(self):
    cfg = _train_cfg()
    critic_terms = cfg.observations["critic"].terms
    ori_term = critic_terms["cube_ori_error"]
    func_name = ori_term.func.__name__
    assert func_name == "goal_rot_err_6d", (
      f"critic cube_ori_error must use goal_rot_err_6d, got '{func_name}'"
    )

  def test_goal_rot_err_6d_returns_6_columns(self):
    """Behavior contract: goal_rot_err_6d returns a (num_envs, 6) tensor."""
    from wuji_mjlab.tasks.reorient.mdp.observations import goal_rot_err_6d

    num_envs = 3
    env = _make_obs_env(num_envs=num_envs)
    result = goal_rot_err_6d(env, command_name="reorient_command")
    assert result.shape == (num_envs, 6), (
      f"goal_rot_err_6d must return shape (N, 6), got {tuple(result.shape)}"
    )


# ---------------------------------------------------------------------------
# 4. Command success counting: goal_reach_count incremented on hold_complete
# ---------------------------------------------------------------------------


@pytest.mark.pipeline
class TestCommandSuccessCounting:
  """goal_reach_count must be incremented exactly when success_achieved fires
  (after success_hold_steps consecutive within-threshold steps).

  These tests drive a real InHandReorientCommand backed by fake entities and
  assert observable state changes, instead of pattern-matching against the
  method source.
  """

  def test_within_threshold_for_hold_steps_increments_goal_reach_count(self):
    """Holding within threshold for success_hold_steps ticks increments
    goal_reach_count by exactly 1 and fires success_achieved on that step."""
    cmd = _make_reorient_command(num_envs=1)
    # Force "within threshold" by aligning object quat with goal quat.
    _align_object_with_goal(cmd)

    success_hold_steps = cmd.cfg.success_hold_steps
    assert cmd.goal_reach_count.item() == 0

    for tick in range(success_hold_steps):
      cmd._update_command()
      if tick < success_hold_steps - 1:
        assert cmd.goal_reach_count.item() == 0, (
          f"goal_reach_count incremented prematurely at tick {tick}"
        )
        assert not cmd._success_achieved.any()

    assert cmd.goal_reach_count.item() == 1, (
      "goal_reach_count must increment by 1 after success_hold_steps "
      "consecutive within-threshold ticks"
    )
    assert cmd._success_achieved.all(), "success_achieved must fire on the success step"

  def test_goal_switches_after_goal_switch_delay_in_success_window(self):
    """After entering the success window, holding for ``goal_switch_delay``
    ticks must trigger ``_goal_switched`` and resample the goal quaternion.

    Once the switch fires, the state machine resets to the approaching
    phase: ``in_success_window`` clears and ``window_timer`` returns to 0.
    """
    cmd = _make_reorient_command(num_envs=1)
    _align_object_with_goal(cmd)

    success_hold_steps = cmd.cfg.success_hold_steps
    goal_switch_delay = cmd.cfg.goal_switch_delay

    for _ in range(success_hold_steps):
      cmd._update_command()
    assert cmd.in_success_window.all()
    goal_before_switch = cmd.goal_quat_w.clone()

    # After ``success_hold_steps`` ticks, window_timer == 1. We need
    # ``goal_switch_delay - 1`` more ticks for window_timer to reach
    # goal_switch_delay and trigger the switch on that same tick.
    extra_ticks_until_switch = goal_switch_delay - 1
    for _ in range(extra_ticks_until_switch):
      cmd._update_command()

    assert cmd._goal_switched.all(), (
      "_goal_switched must fire on the step window_timer reaches goal_switch_delay"
    )
    # New goal sampled (vanishing probability of an exact match).
    assert not torch.allclose(cmd.goal_quat_w, goal_before_switch, atol=1e-3)
    # State machine resets back to approaching.
    assert not cmd.in_success_window.any()
    assert cmd.window_timer.item() == 0

  def test_two_phase_state_machine_progression(self):
    """Both APPROACHING and SUCCESS_WINDOW phases must be reachable: from
    out-of-window the env transitions into in_success_window after
    success_hold_steps consecutive within-threshold ticks."""
    cmd = _make_reorient_command(num_envs=2)
    _align_object_with_goal(cmd)

    assert not cmd.in_success_window.any(), "must start in approaching phase"

    for _ in range(cmd.cfg.success_hold_steps):
      cmd._update_command()

    assert cmd.in_success_window.all(), (
      "must transition to success window after holding within threshold"
    )

  def test_goal_reach_count_not_incremented_while_in_success_window(self):
    """While already in the success window, additional within-threshold
    ticks must not bump goal_reach_count (it only increments on the
    approaching→success transition)."""
    cmd = _make_reorient_command(num_envs=1)
    _align_object_with_goal(cmd)

    for _ in range(cmd.cfg.success_hold_steps):
      cmd._update_command()
    assert cmd.goal_reach_count.item() == 1
    assert cmd.in_success_window.all()

    # Stay in success window for a few ticks (still < goal_switch_delay).
    extra_ticks = max(1, cmd.cfg.goal_switch_delay // 2)
    for _ in range(extra_ticks):
      cmd._update_command()

    assert cmd.goal_reach_count.item() == 1, (
      "goal_reach_count must not increment while already in success window"
    )


# ---------------------------------------------------------------------------
# 5. Curriculum data source: reads from command metrics, not stale buffers
# ---------------------------------------------------------------------------


@pytest.mark.pipeline
class TestEnvCfgTopLevelGroups:
  """make_reorient_env_cfg must expose the expected term groups so the
  thin assembler in reorient_env_cfg.py keeps the public contract intact."""

  def test_keeps_expected_top_level_groups(self):
    cfg = _train_cfg()
    assert set(cfg.observations) == {"policy", "critic"}
    assert "reorient_command" in cfg.commands
    assert "object_disturbance_force" in cfg.events
    assert "orientation_alignment" in cfg.rewards
    assert "success_curriculum" in cfg.curriculum
    assert "joint_pos" in cfg.actions
    assert "time_out" in cfg.terminations


@pytest.mark.pipeline
class TestCurriculumDataSource:
  """Curriculum functions must read live data from command_manager metrics,
  not from stale env-level buffers that could be out of date."""

  def test_success_curriculum_reads_from_command_metrics(self):
    """Behavior contract: reorient_success_curriculum threshold check is
    driven by command.metrics['goal_reach_count'], so feeding a
    large goal_reach_count yields a positive delta."""
    from wuji_mjlab.tasks.reorient.mdp.curriculums import (
      reorient_success_curriculum,
    )

    num_envs = 4
    env = _make_curriculum_env(
      num_envs=num_envs,
      goal_reach_count_metric=torch.full((num_envs,), 5.0),
    )
    env_ids = torch.arange(num_envs)

    out = reorient_success_curriculum(
      env, env_ids, count_threshold=3, delta_per_loop=0.1
    )

    assert out["success_env_frac"] == pytest.approx(1.0), (
      "All envs above threshold should yield success_env_frac == 1.0"
    )
    assert out["delta"] > 0, "When all envs are above threshold, delta must be positive"

  def test_success_curriculum_ignores_stale_env_attribute(self):
    """Behavior contract: the curriculum reads command.metrics, NOT a
    stale env.goal_reach_count buffer. Setting env.goal_reach_count to a
    non-zero value while the command metric stays below threshold must
    produce a negative delta (matching the metric, not the env attr)."""
    from wuji_mjlab.tasks.reorient.mdp.curriculums import (
      reorient_success_curriculum,
    )

    num_envs = 4
    env = _make_curriculum_env(
      num_envs=num_envs,
      goal_reach_count_metric=torch.zeros(num_envs),
    )
    # Deliberately set the (stale) env attribute to a large value. If the
    # function read from env.goal_reach_count, delta would be positive.
    env.goal_reach_count = torch.full((num_envs,), 999.0)
    env_ids = torch.arange(num_envs)

    out = reorient_success_curriculum(
      env, env_ids, count_threshold=3, delta_per_loop=0.1
    )

    assert out["success_env_frac"] == 0.0
    assert out["delta"] < 0, (
      "Delta must be negative (reads from metric, ignores stale env attr)"
    )

  def test_success_curriculum_config_references_command_name(self):
    """The curriculum config must pass the correct command_name parameter."""
    cfg = _train_cfg()
    cur = cfg.curriculum["success_curriculum"]
    assert cur.params["command_name"] == "reorient_command", (
      f"success_curriculum must reference 'reorient_command', "
      f"got '{cur.params['command_name']}'"
    )

  def test_adaptive_episode_curriculum_responds_to_episode_length_buf(self):
    """Behavior contract: feeding longer episodes via env.episode_length_buf
    produces a higher mean_episode_steps in the curriculum output."""
    from wuji_mjlab.tasks.reorient.mdp.curriculums import (
      adaptive_episode_curriculum,
    )

    num_envs = 4
    short_env = _make_curriculum_env(num_envs=num_envs)
    short_env.episode_length_buf = torch.full((num_envs,), 100, dtype=torch.long)

    long_env = _make_curriculum_env(num_envs=num_envs)
    long_env.episode_length_buf = torch.full((num_envs,), 900, dtype=torch.long)

    env_ids = torch.arange(num_envs)
    short_out = adaptive_episode_curriculum(short_env, env_ids, target_steps=800)
    long_out = adaptive_episode_curriculum(long_env, env_ids, target_steps=800)

    assert long_out["mean_episode_steps"] > short_out["mean_episode_steps"]
    assert long_out["reached_target_frac"] == 1.0
    assert short_out["reached_target_frac"] == 0.0


# ---------------------------------------------------------------------------
# Local helpers for the behavior-level assertions above
# ---------------------------------------------------------------------------


class _FakeEntity:
  """Minimal Entity-like stub: ``find_bodies`` + ``data`` namespace."""

  def __init__(self, body_names, pose_w, root_quat_w=None):
    self._body_names = list(body_names)
    self.data = SimpleNamespace(body_link_pose_w=pose_w)
    if root_quat_w is not None:
      self.data.root_link_quat_w = root_quat_w
      self.data.root_link_pos_w = torch.zeros(root_quat_w.shape[0], 3)

  def find_bodies(self, pattern, preserve_order: bool = False):
    patterns = pattern if isinstance(pattern, (tuple, list)) else (pattern,)
    regexes = [re.compile(p) for p in patterns]
    ids = [
      i for i, n in enumerate(self._body_names) if any(r.fullmatch(n) for r in regexes)
    ]
    return ids, [self._body_names[i] for i in ids]


def _identity_quat(num_envs: int) -> torch.Tensor:
  q = torch.zeros(num_envs, 4)
  q[:, 0] = 1.0
  return q


def _make_reorient_command(
  num_envs: int,
  success_hold_steps: int = 3,
  goal_switch_delay: int = 4,
) -> InHandReorientCommand:
  """Build an InHandReorientCommand backed by fake entities (CPU)."""
  body_names = ["robot_palm_link", "finger1"]
  pose_w = torch.zeros(num_envs, len(body_names), 7)
  pose_w[:, :, 3] = 1.0  # identity quat for every body

  robot = _FakeEntity(body_names=body_names, pose_w=pose_w)
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
  """Force the cube quat to match the current goal quat so within_threshold==True."""
  cmd.object.data.root_link_quat_w = cmd.goal_quat_w.clone()


class _FakeScene(dict):
  """Dict-based scene that also carries a ``device`` attribute."""

  def __init__(self, items, device: str = "cpu") -> None:
    super().__init__(items)
    self.device = device


def _make_obs_env(num_envs: int) -> SimpleNamespace:
  """Build an env scaffold sufficient for ``goal_rot_err_6d``."""
  body_names = ["robot_palm_link", "finger1"]
  palm_pose_w = torch.zeros(num_envs, len(body_names), 7)
  palm_pose_w[:, :, 3] = 1.0  # identity quat
  robot = _FakeEntity(body_names=body_names, pose_w=palm_pose_w)

  obj = _FakeEntity(
    body_names=["cube"],
    pose_w=torch.zeros(num_envs, 1, 7),
    root_quat_w=_identity_quat(num_envs),
  )

  command = SimpleNamespace(goal_quat=_identity_quat(num_envs))
  scene = _FakeScene({"robot": robot, "object": obj})
  return SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    scene=scene,
    command_manager=make_command_manager(command),
  )


def _make_curriculum_env(
  num_envs: int = 4,
  goal_reach_count_metric: torch.Tensor | None = None,
) -> SimpleNamespace:
  """Build a SimpleNamespace env supporting both curriculum functions."""
  if goal_reach_count_metric is None:
    goal_reach_count_metric = torch.zeros(num_envs)
  command = SimpleNamespace(metrics={"goal_reach_count": goal_reach_count_metric})
  return SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
    termination_manager=SimpleNamespace(
      terminated=torch.zeros(num_envs, dtype=torch.bool)
    ),
    command_manager=make_command_manager(command),
  )
