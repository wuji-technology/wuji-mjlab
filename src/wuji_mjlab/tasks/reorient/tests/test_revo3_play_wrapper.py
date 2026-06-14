# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import wuji_mjlab.tasks  # noqa: F401
from scripts.play.play_rsl_rl_zero_wrist_pitch import (
  _DEFAULT_REVO3_FIXED_OBJECT_OFFSET_IN_PALM,
  _force_fixed_object_init,
  _resolve_fixed_object_offset,
)
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

_REVO3_FINETUNE_TASK = "Revo3RightHand_Reorient_ReposeReward_FineTune"


def test_revo3_fixed_object_init_uses_calibrated_default_offset():
  env_cfg, _ = prepare_task_cfgs(_REVO3_FINETUNE_TASK, [], play=True)

  fixed_offset = _resolve_fixed_object_offset(
    _REVO3_FINETUNE_TASK,
    env_cfg,
    requested_offset=None,
  )
  _force_fixed_object_init(env_cfg, cube_offset_in_palm=fixed_offset)

  event = env_cfg.events["reset_object_pose"]
  assert event.params["cube_offset_in_palm"] == (
    _DEFAULT_REVO3_FIXED_OBJECT_OFFSET_IN_PALM
  )
  assert event.params["pos_noise"] == 0.0
  assert "pos_noise_curriculum_term" not in event.params


def test_revo3_fixed_object_init_preserves_cli_offset_override():
  custom_offset = (0.023, 0.005, 0.055)
  env_cfg, _ = prepare_task_cfgs(
    _REVO3_FINETUNE_TASK,
    [
      "env.events.reset_object_pose.params.cube_offset_in_palm="
      + repr(custom_offset)
    ],
    play=True,
  )

  fixed_offset = _resolve_fixed_object_offset(
    _REVO3_FINETUNE_TASK,
    env_cfg,
    requested_offset=None,
  )
  _force_fixed_object_init(env_cfg, cube_offset_in_palm=fixed_offset)

  assert fixed_offset is None
  assert env_cfg.events["reset_object_pose"].params["cube_offset_in_palm"] == (
    custom_offset
  )


def test_revo3_fixed_object_init_accepts_explicit_cube_offset_argument():
  custom_offset = (0.024, 0.005, 0.075)
  env_cfg, _ = prepare_task_cfgs(_REVO3_FINETUNE_TASK, [], play=True)

  fixed_offset = _resolve_fixed_object_offset(
    _REVO3_FINETUNE_TASK,
    env_cfg,
    requested_offset=custom_offset,
  )
  _force_fixed_object_init(env_cfg, cube_offset_in_palm=fixed_offset)

  assert env_cfg.events["reset_object_pose"].params["cube_offset_in_palm"] == (
    custom_offset
  )
