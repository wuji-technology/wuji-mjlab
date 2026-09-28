# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import math

import pytest
import torch
import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply
from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
  REORIENT_HAND2_PALM_NORMAL_AXIS,
)
from wuji_mjlab.tasks.reorient.reorient_constants import REORIENT_PALM_NORMAL_AXIS

_PITCHES = (-0.4, -0.2, 0.0, 0.1)
_TASKS = {
  "hand1": ("WujiHand_Reorient", REORIENT_PALM_NORMAL_AXIS),
  "hand2": ("WujiHand2_Reorient_50Hz", REORIENT_HAND2_PALM_NORMAL_AXIS),
}
_TOLERANCE = math.radians(0.5)


@pytest.fixture(scope="module")
def envs():
  built = {}
  try:
    for generation, (task, _) in _TASKS.items():
      cfg = load_env_cfg(task, play=False)
      cfg.scene.num_envs = 2
      cfg.events["reset_robot_pose"].params["pose_range"] = {"pitch": (0.0, 0.0)}
      cfg.events["reset_object_pose"].params["pos_noise"] = 0.0
      built[generation] = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
    yield built
  finally:
    for env in built.values():
      env.close()


def _reset_at_pitch(env: ManagerBasedRlEnv, pitch: float) -> None:
  pose_range = {"pitch": (pitch, pitch)}
  env.cfg.events["reset_robot_pose"].params["pose_range"] = pose_range
  # EventManager deep-copies configured terms at construction; modify its active copy too.
  env.event_manager.get_term_cfg("reset_robot_pose").params["pose_range"] = pose_range
  env.reset()


@pytest.mark.parametrize("generation", ("hand1", "hand2"))
@pytest.mark.parametrize("pitch", _PITCHES)
def test_configured_pitch_tilts_palm_normal_about_world_y(envs, generation, pitch):
  env = envs[generation]
  _reset_at_pitch(env, pitch)
  _, normal_axis = _TASKS[generation]
  robot = env.scene["robot"]
  root_quat = robot.data.root_link_pose_w[:, 3:7]
  local_normal = torch.zeros((env.num_envs, 3), device=env.device)
  local_normal[:, normal_axis] = 1.0
  normals = quat_apply(root_quat, local_normal)
  for normal in normals.tolist():
    measured = math.atan2(normal[0], normal[2])
    assert abs(measured - pitch) <= _TOLERANCE, (
      f"{generation} pitch={pitch:+.4f} rad; measured={measured:+.6f} rad "
      f"({math.degrees(measured):+.3f} deg), normal={normal}"
    )
    assert abs(normal[1]) <= math.sin(_TOLERANCE), (
      f"{generation} pitch={pitch:+.4f} rad; measured={measured:+.6f} rad "
      f"({math.degrees(measured):+.3f} deg), normal={normal}"
    )
