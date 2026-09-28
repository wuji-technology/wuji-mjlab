# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Policy-only control and operator-input checks without physical hardware."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from runtime import cli, safety
from runtime import loop as control
from runtime.session import Mode


class Motion:
    def __init__(self, stop_after=99):
        self.reference_qpos = np.full((100, 20), 0.1, dtype=np.float32)
        self.num_frames = len(self.reference_qpos)
        self.steps = 0
        self.done = False
        self.stop_after = stop_after
        self.clock_shifts = []

    def start(self, *, now):
        self.started_at = now

    def step(self, *, now):
        self.steps += 1
        self.done = self.steps >= self.stop_after
        return self.steps

    def shift_clock(self, elapsed):
        self.clock_shifts.append(elapsed)


class Controller:
    def __init__(self):
        self.targets = []

    def get_joint_actual_position(self):
        return np.zeros((5, 4))

    def set_joint_target_position(self, target):
        self.targets.append(np.asarray(target).copy())


class Keys:
    interactive = True

    def __init__(self, values):
        self.values = iter(values)

    def get(self):
        return next(self.values, "r")


@pytest.fixture
def rig(monkeypatch):
    observed_frames = []

    def observe(motion, frame, *args, **kwargs):
        observed_frames.append(frame)
        return np.zeros(155, dtype=np.float32)

    monkeypatch.setattr(control.obs_mod, "build_observation", observe)
    monkeypatch.setattr(control, "pace_to_deadline", lambda deadline: None)
    loaded = SimpleNamespace(
        action_scale=0.5, clip_lo=-0.6, clip_hi=0.6, alpha=0.5,
        infer=Mock(return_value=np.full(20, 0.2, dtype=np.float32)),
    )
    session = SimpleNamespace(
        loaded=loaded, motion=Motion(), objpose=None, viewer=None,
        mode=Mode(obj_from_ref=True), ctrl_dt=0.02,
        hand=SimpleNamespace(
            read_joint_lower_limit=lambda: np.full((5, 4), -1.0),
            read_joint_upper_limit=lambda: np.full((5, 4), 1.0),
        ),
    )
    loop = control.ControlLoop(session)
    monkeypatch.setattr(loop, "tick_idle", lambda ctrl: None)
    return SimpleNamespace(loop=loop, session=session, ctrl=Controller(),
                           frames=observed_frames)


def test_cli_exposes_only_policy_observation_options():
    parser = cli.build_parser()
    group = next(group for group in parser._action_groups if group.title == "mode")
    assert {option for action in group._group_actions for option in action.option_strings} == {
        "--hold-last-frame", "--open-loop", "--obj-from-ref",
    }
    assert {field.name for field in fields(Mode)} == {"open_loop", "obj_from_ref"}
    args = parser.parse_args(["--policy", "policy.onnx", "--motion", "clip.npz"])
    assert Mode(args.open_loop, args.obj_from_ref).needs_camera


def test_viewer_forwards_only_enter_and_reset():
    keys = safety.Keys()
    callback = safety.viewer_key_callback(keys)
    callback(32)
    assert keys.get() is None
    for code, expected in [(257, "\n"), (335, "\n"), (82, "r")]:
        callback(code)
        assert keys.get() == expected


def test_space_cannot_start_control(rig):
    assert rig.loop.run(rig.ctrl, keys=Keys([" ", "r"]))
    assert rig.loop.state == control.FROZEN
    assert not rig.ctrl.targets
    rig.session.loaded.infer.assert_not_called()


def test_two_enters_start_policy_and_space_does_not_change_hold(rig):
    assert rig.loop.run(rig.ctrl, keys=Keys(["\n", " ", "\n", "r"]))
    assert rig.loop.state == control.POLICY
    assert rig.frames == [0, 0, 1]
    assert rig.session.loaded.infer.call_count == 3
    assert len(rig.session.motion.clock_shifts) == 1
    assert len(rig.ctrl.targets) == 3


def test_placement_failure_keeps_hand_frozen(rig, monkeypatch):
    monkeypatch.setattr(rig.loop, "placement", lambda: ("missing pose", "not ready"))
    assert rig.loop.run(rig.ctrl, keys=Keys(["\n", "r"]))
    assert rig.loop.state == control.FROZEN
    assert not rig.ctrl.targets
    rig.session.loaded.infer.assert_not_called()


@pytest.mark.parametrize("state", [control.HOLD, control.POLICY])
@pytest.mark.parametrize("open_loop,obj_from_ref", [(False, False), (False, True),
                                                    (True, False), (True, True)])
def test_every_active_tick_uses_policy(rig, state, open_loop, obj_from_ref):
    rig.loop.open_loop = open_loop
    rig.loop.obj_from_ref = obj_from_ref or open_loop
    rig.loop.objpose = SimpleNamespace(
        snapshot=lambda: {"pen_age_s": 0.0}, has_pose=lambda snap: True,
        local=lambda snap: (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
    )
    rig.loop.enter_state(state, np.zeros(20))
    result = rig.loop.tick(rig.ctrl, now=1.0)
    rig.session.loaded.infer.assert_called_once()
    np.testing.assert_allclose(result.raw, 0.2)
    np.testing.assert_allclose(rig.ctrl.targets[0], 0.15)
    assert rig.session.motion.steps == (0 if state == control.HOLD else 1)


def test_hold_to_policy_preserves_filter_state(rig):
    rig.loop.enter_state(control.HOLD, np.zeros(20))
    rig.loop.tick(rig.ctrl, now=1.0)
    residual = rig.loop.action.prev_residual.copy()
    previous_time = rig.loop.prev_time
    rig.loop.enter_state(control.POLICY, np.zeros(20))
    np.testing.assert_array_equal(rig.loop.action.prev_residual, residual)
    assert rig.loop.prev_time == previous_time


def test_frozen_or_unknown_states_cannot_command(rig):
    with pytest.raises(RuntimeError, match="HOLD or POLICY"):
        rig.loop.tick(rig.ctrl, now=1.0)
    with pytest.raises(ValueError, match="unsupported control state"):
        rig.loop.enter_state("manual", np.zeros(20))
    assert rig.loop.state == control.FROZEN
    assert not rig.ctrl.targets
    rig.session.loaded.infer.assert_not_called()


def test_headless_run_still_uses_policy(rig):
    rig.session.motion.stop_after = 1
    assert rig.loop.run(rig.ctrl) is False
    assert rig.loop.state == control.POLICY
    rig.session.loaded.infer.assert_called_once()
    assert len(rig.ctrl.targets) == 1


@pytest.mark.parametrize("key", ["r", "R"])
def test_reset_returns_without_commanding(rig, key):
    assert rig.loop.run(rig.ctrl, keys=Keys([key]))
    assert not rig.ctrl.targets


def test_invalid_policy_output_never_reaches_controller(rig):
    rig.loop.enter_state(control.POLICY, np.zeros(20))
    rig.session.loaded.infer.return_value = np.full(20, np.nan)
    with pytest.raises(FloatingPointError, match="non-finite"):
        rig.loop.tick(rig.ctrl, now=1.0)
    assert not rig.ctrl.targets
