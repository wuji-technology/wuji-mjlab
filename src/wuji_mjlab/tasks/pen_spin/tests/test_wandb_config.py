# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from rsl_rl.utils import WandbLogWriter
from wuji_mjlab.rl.log_writer import WujiWandbLogWriter
from wuji_mjlab.rl.runner import WujiOnPolicyRunner
from wuji_mjlab.tasks.pen_spin.config.wuji_hand2.rsl_rl.ppo import (
  wuji_hand2_pen_spin_ppo_runner_cfg,
)
from wuji_mjlab.tasks.reorient.config.wuji_hand.rsl_rl.ppo import (
  wuji_hand_reorient_ppo_runner_cfg,
)
from wuji_mjlab.tasks.reorient.config.wuji_hand2.rsl_rl.ppo import (
  wuji_hand2_reorient_ppo_runner_cfg,
)


@pytest.fixture(autouse=True)
def clear_wandb_environment(monkeypatch):
  for name in ("WANDB_PROJECT", "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_USERNAME"):
    monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
  "factory",
  [
    wuji_hand2_pen_spin_ppo_runner_cfg,
    wuji_hand_reorient_ppo_runner_cfg,
    wuji_hand2_reorient_ppo_runner_cfg,
  ],
)
def test_task_config_can_load_without_wandb_credentials(factory):
  assert factory().wandb_project == ""


@pytest.mark.parametrize("logger", ["tensorboard", "wandb"])
def test_only_wandb_requires_credentials_before_runner_initialization(
  monkeypatch, logger
):
  initialize = Mock(return_value=None)
  monkeypatch.setattr(MjlabOnPolicyRunner, "__init__", initialize)
  env = SimpleNamespace(unwrapped=SimpleNamespace())
  cfg = {
    "logger": logger,
    "wandb_project": "stale-project",
    "max_iterations": 100,
    "num_steps_per_env": 40,
  }
  if logger == "wandb":
    with pytest.raises(ValueError, match="WANDB_PROJECT, WANDB_API_KEY"):
      WujiOnPolicyRunner(env, cfg, log_dir="unused")
    initialize.assert_not_called()
  else:
    WujiOnPolicyRunner(env, cfg, log_dir="unused")
    initialize.assert_called_once()
    assert cfg["logger"] == "tensorboard"
    assert env.unwrapped.max_common_steps == 4000


@pytest.mark.parametrize(
  "task",
  [
    "WujiHand_Reorient",
    "WujiHand2_Reorient",
    "WujiHand2_Reorient_50Hz",
    "WujiHand2_PenSpin",
  ],
)
def test_inference_runner_needs_no_wandb_credentials(monkeypatch, task):
  initialize = Mock(return_value=None)
  monkeypatch.setattr(MjlabOnPolicyRunner, "__init__", initialize)
  cfg = asdict(load_rl_cfg(task))
  env = SimpleNamespace(unwrapped=SimpleNamespace())

  load_runner_cls(task)(env, cfg, device="cpu")

  initialize.assert_called_once_with(env, cfg, None, "cpu")
  assert env.unwrapped.max_common_steps == (
    cfg["max_iterations"] * cfg["num_steps_per_env"]
  )


@pytest.mark.parametrize("missing", ["WANDB_PROJECT", "WANDB_API_KEY"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_writer_rejects_missing_credentials_before_wandb_init(
  monkeypatch, missing, value
):
  monkeypatch.setenv("WANDB_PROJECT", "test-project")
  monkeypatch.setenv("WANDB_API_KEY", "test-key")
  if value is None:
    monkeypatch.delenv(missing)
  else:
    monkeypatch.setenv(missing, value)
  initialize = Mock(return_value=None)
  monkeypatch.setattr(WandbLogWriter, "__init__", initialize)
  with pytest.raises(ValueError, match=missing):
    WujiWandbLogWriter("unused", "test-project")
  initialize.assert_not_called()


def test_runner_uses_current_environment_instead_of_saved_project(monkeypatch):
  monkeypatch.setenv("WANDB_PROJECT", "selected-project")
  monkeypatch.setenv("WANDB_API_KEY", "test-key")
  initialize = Mock(return_value=None)
  monkeypatch.setattr(MjlabOnPolicyRunner, "__init__", initialize)
  env = SimpleNamespace(unwrapped=SimpleNamespace())
  cfg = {
    "logger": "wandb",
    "wandb_project": "stale-project",
    "max_iterations": 100,
    "num_steps_per_env": 40,
  }
  WujiOnPolicyRunner(env, cfg, log_dir="unused")
  assert cfg["wandb_project"] == "selected-project"
  assert cfg["logger"]["project_name"] == "selected-project"
  assert "test-key" not in repr(cfg)
  assert env.unwrapped.max_common_steps == 4000
  initialize.assert_called_once()


def test_writer_rejects_saved_project_mismatch(monkeypatch):
  monkeypatch.setenv("WANDB_PROJECT", "selected-project")
  monkeypatch.setenv("WANDB_API_KEY", "test-key")
  initialize = Mock(return_value=None)
  monkeypatch.setattr(WandbLogWriter, "__init__", initialize)
  with pytest.raises(ValueError, match="must match WANDB_PROJECT"):
    WujiWandbLogWriter("unused", "stale-project")
  initialize.assert_not_called()


def test_writer_accepts_explicit_project_and_key(monkeypatch):
  monkeypatch.setenv("WANDB_PROJECT", "selected-project")
  monkeypatch.setenv("WANDB_API_KEY", "test-key")
  initialize = Mock(return_value=None)
  monkeypatch.setattr(WandbLogWriter, "__init__", initialize)
  WujiWandbLogWriter("unused", "selected-project")
  initialize.assert_called_once_with("unused", "selected-project")
