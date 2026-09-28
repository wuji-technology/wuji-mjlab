# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from wuji_mjlab.utils.train_config_utils import load_train_config


def test_load_train_config_flattens_overrides(tmp_path: Path) -> None:
  cfg = tmp_path / "cfg.yaml"
  cfg.write_text(
    "task: WujiHand2_Reorient\n"
    "env:\n"
    "  scene:\n"
    "    num_envs: 4096\n"
    "agent:\n"
    "  max_iterations: 7500\n"
    "  run_name: MyRun\n"
    "args: [--gpu-ids, '0']\n"
  )
  task, overrides, extra_args = load_train_config(cfg)
  assert task == "WujiHand2_Reorient"
  assert "env.scene.num_envs=4096" in overrides
  assert "agent.max_iterations=7500" in overrides
  assert "agent.run_name=MyRun" in overrides
  assert extra_args == ["--gpu-ids", "0"]


def test_load_train_config_rejects_unknown_keys(tmp_path: Path) -> None:
  cfg = tmp_path / "cfg.yaml"
  cfg.write_text("task: WujiHand_Reorient\nunexpected: 1\n")
  with pytest.raises(ValueError, match="Unknown keys"):
    load_train_config(cfg)


@pytest.mark.parametrize(
  "module_name",
  [
    "wuji_mjlab.assets.robots.wuji_hand.wuji_hand_cfg",
    "wuji_mjlab.assets.robots.wuji_hand2.wuji_hand2_cfg",
    "wuji_mjlab.assets.objects.inhand_object.object_cfg",
  ],
)
def test_asset_import_order_registers_all_reorient_tasks(module_name: str) -> None:
  code = (
    "import importlib\n"
    f"importlib.import_module({module_name!r})\n"
    "from mjlab.tasks.registry import list_tasks\n"
    "tasks = set(list_tasks())\n"
    "expected = {'WujiHand_Reorient', 'WujiHand2_Reorient', 'WujiHand2_Reorient_50Hz'}\n"
    "print('registered_tasks:', sorted(tasks))\n"
    "assert expected <= tasks, sorted(expected - tasks)\n"
  )
  result = subprocess.run(
    [sys.executable, "-c", code],
    env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "MUJOCO_GL": "disable"},
    capture_output=True,
    text=True,
    timeout=60,
  )
  assert result.returncode == 0, result.stdout + result.stderr
