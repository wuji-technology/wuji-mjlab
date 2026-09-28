# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

from wuji_mjlab.utils.curriculum import get_curriculum_value


class SingleHandResidualEMAAction(BaseAction):
  """Apply policy residuals to the command-owned joint reference."""

  cfg: SingleHandResidualEMAActionCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    motion = env.command_manager.get_term(cfg.command_name)
    if tuple(self._target_names) != tuple(motion.joint_names):
      raise ValueError("actuated joints must match the command reference order")
    if cfg.offset != 0:
      raise ValueError("offset must be 0 for residual actions.")
    if cfg.clip is not None:
      raise ValueError("BaseAction clip must be None; use residual_clip instead.")
    if not 0 <= cfg.alpha <= 1:
      raise ValueError("alpha must be in [0.0, 1.0].")
    if not 0.0 <= float(cfg.mask_prob) <= 1.0:
      raise ValueError("mask_prob must be in [0.0, 1.0].")
    if cfg.mask_num_dofs < 1:
      raise ValueError("mask_num_dofs must be >= 1.")
    if cfg.mask_duration_max < 1:
      raise ValueError("mask_duration_max must be >= 1.")

    self._targets = torch.zeros_like(self._raw_actions)
    self._targets_valid = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    self._mask_enabled = float(cfg.mask_prob) > 0.0
    self._mask_num_dofs = min(int(cfg.mask_num_dofs), self.action_dim)
    self._mask_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self._mask_remaining = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )
    self._mask_dofs = torch.zeros_like(self._raw_actions, dtype=torch.bool)

  def process_actions(self, actions):
    self._raw_actions[:] = actions
    residual = self._raw_actions * self._scale
    if self.cfg.residual_clip is not None:
      residual = residual.clamp(*self.cfg.residual_clip)
    self._processed_actions[:] = (
      self.cfg.alpha * residual + (1 - self.cfg.alpha) * self._processed_actions
    )
    self._update_action_mask()

    motion = self._env.command_manager.get_term(self.cfg.command_name)
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = motion.joint_pos + self._processed_actions - encoder_bias
    if self._mask_enabled:
      # Masked DoFs stay frozen at the previously executed target.
      frozen = (self._mask_active & self._targets_valid).unsqueeze(-1) & self._mask_dofs
      target = torch.where(frozen, self._targets, target)
    self._targets[:] = target
    self._targets_valid[:] = True

  def _update_action_mask(self):
    if not self._mask_enabled:
      return
    active_at_start = self._mask_active.clone()
    self._mask_remaining[active_at_start] -= 1
    expired = active_at_start & (self._mask_remaining <= 0)
    self._mask_active[expired] = False
    self._mask_dofs[expired] = False
    fire = (~active_at_start) & (
      torch.rand(self.num_envs, device=self.device) < self._mask_prob_eff()
    )
    ids = fire.nonzero(as_tuple=True)[0]
    if len(ids):
      scores = torch.rand(len(ids), self.action_dim, device=self.device)
      chosen = scores.topk(self._mask_num_dofs, dim=1).indices
      self._mask_dofs[ids] = False
      self._mask_dofs[ids.unsqueeze(1), chosen] = True
      self._mask_remaining[ids] = torch.randint(
        int(self.cfg.mask_duration_min),
        self._mask_duration_max_eff() + 1,
        (len(ids),),
        device=self.device,
      )
      self._mask_active[ids] = True

  def _mask_prob_eff(self) -> float:
    """Trigger probability, ramped 0 -> ``mask_prob`` by ``mask_curriculum_term`` when set."""
    term = self.cfg.mask_curriculum_term
    if term is None:
      return float(self.cfg.mask_prob)
    return float(self.cfg.mask_prob) * get_curriculum_value(self._env, term, 0.0)

  def _mask_duration_max_eff(self) -> int:
    """Curriculum-scaled upper bound on the freeze duration."""
    dmin, dmax = int(self.cfg.mask_duration_min), int(self.cfg.mask_duration_max)
    term = self.cfg.mask_curriculum_term
    if term is None:
      return dmax
    scale = get_curriculum_value(self._env, term, 1.0)
    return max(int(round(dmin + (dmax - dmin) * scale)), dmin)

  def apply_actions(self):
    self._entity.set_joint_position_target(self._targets, joint_ids=self._target_ids)

  def reset(self, env_ids=None):
    env_ids = slice(None) if env_ids is None else env_ids
    for value in (
      self._raw_actions,
      self._processed_actions,
      self._targets,
      self._targets_valid,
      self._mask_active,
      self._mask_remaining,
      self._mask_dofs,
    ):
      value[env_ids] = 0


@dataclass(kw_only=True)
class SingleHandResidualEMAActionCfg(BaseActionCfg):
  command_name: str = "motion"
  scale: float | dict[str, float] = 0.5
  residual_clip: tuple[float, float] | None = (-0.6, 0.6)
  alpha: float = 0.5
  mask_prob: float = 0.0
  mask_num_dofs: int = 3
  mask_duration_min: int = 1
  mask_duration_max: int = 10
  # When set, the mask's trigger probability and freeze duration ramp 0 -> full
  # with this curriculum term's scalar.
  mask_curriculum_term: str | None = None

  def build(self, env):
    return SingleHandResidualEMAAction(self, env)
