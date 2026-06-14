# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg

from wuji_mjlab.utils.curriculum import get_curriculum_value


class JointPositionOffsetEMAAction(JointPositionAction):
  """Joint position action: default_pos + action * scale, with EMA smoothing and warmup hold."""

  cfg: "JointPositionOffsetEMAActionCfg"

  def __init__(self, cfg: "JointPositionOffsetEMAActionCfg", env):
    super().__init__(cfg, env)

    self._action_scale = cfg.action_scale
    self._ema_alpha = cfg.ema_alpha
    self._warmup_time_s = cfg.warmup_time_s
    self._action_hold_max_prob = cfg.action_hold_max_prob
    self._action_hold_curriculum_term = cfg.action_hold_curriculum_term

    self._default_joint_pos = self._entity.data.default_joint_pos[
      :, self._target_ids
    ].clone()

    soft_limits = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
    self._lower_limits = soft_limits[..., 0]
    self._upper_limits = soft_limits[..., 1]

    self._prev_target = self._default_joint_pos.clone()
    self._prev_effective_action = torch.zeros_like(self._raw_actions)

  def _action_hold_probability(self) -> float:
    if self._action_hold_max_prob <= 0.0:
      return 0.0

    scale = 1.0
    if self._action_hold_curriculum_term:
      scale = get_curriculum_value(self._env, self._action_hold_curriculum_term, 0.0)
    return max(min(float(self._action_hold_max_prob) * float(scale), 1.0), 0.0)

  def process_actions(self, actions: torch.Tensor):
    self._raw_actions[:] = actions
    clamped = torch.clamp(actions, -1.0, 1.0)
    hold_prob = self._action_hold_probability()
    if hold_prob > 0.0:
      hold_mask = (
        torch.rand(actions.shape[0], 1, device=actions.device) < hold_prob
      )
      effective_actions = torch.where(hold_mask, self._prev_effective_action, clamped)
    else:
      effective_actions = clamped
    self._prev_effective_action = effective_actions.clone()

    raw_target = self._default_joint_pos + effective_actions * self._action_scale
    raw_target = torch.clamp(raw_target, self._lower_limits, self._upper_limits)

    smoothed = (
      self._ema_alpha * raw_target + (1.0 - self._ema_alpha) * self._prev_target
    )

    in_warmup = (
      self._env.episode_length_buf * self._env.step_dt < self._warmup_time_s
    ).unsqueeze(-1)
    self._processed_actions = torch.where(in_warmup, self._default_joint_pos, smoothed)
    self._prev_target = self._processed_actions.clone()

  @property
  def processed_action(self) -> torch.Tensor:
    return self._processed_actions

  def reset(self, env_ids: torch.Tensor) -> None:
    super().reset(env_ids)
    self._prev_target[env_ids] = self._default_joint_pos[env_ids]
    self._prev_effective_action[env_ids] = 0.0


@dataclass(kw_only=True)
class JointPositionOffsetEMAActionCfg(JointPositionActionCfg):
  """Configuration for offset + EMA action with warmup."""

  action_scale: float = 0.5
  ema_alpha: float = 0.5
  warmup_time_s: float = 0.4
  action_hold_max_prob: float = 0.0
  action_hold_curriculum_term: str = ""

  def build(self, env) -> JointPositionOffsetEMAAction:
    return JointPositionOffsetEMAAction(self, env)
