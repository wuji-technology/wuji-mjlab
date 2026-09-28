# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Reward terms shared by the Wuji hand tasks."""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg


def finger_self_collision_penalty(
  env,
  sensor_cfg: SceneEntityCfg = SceneEntityCfg("finger_collision"),
) -> torch.Tensor:
  sensor = env.scene[sensor_cfg.name]
  return torch.sum((sensor.data.found > 0).float(), dim=-1)
