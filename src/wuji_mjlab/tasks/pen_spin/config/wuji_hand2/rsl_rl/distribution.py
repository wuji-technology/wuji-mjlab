# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Softplus-parameterized Gaussian policy output."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from rsl_rl.modules import HeteroscedasticGaussianDistribution
from torch.distributions import Normal


class SoftplusGaussianDistribution(HeteroscedasticGaussianDistribution):
  """State-dependent std with softplus activation.

  The MLP emits ``[..., 2, output_dim]``; the first slice is the mean and the
  second is mapped to a strictly positive std by ``softplus(scale_raw) +
  min_std``. Unlike a hard clamp the floor is smooth, so gradients keep
  flowing when the raw scale is driven far negative.

  ``std_type`` must be passed by keyword because the base signature places
  ``std_range`` before it; positional passing would silently bind to the wrong
  parameter.
  """

  def __init__(
    self, output_dim: int, init_std: float = 0.5, min_std: float = 0.01
  ) -> None:
    # Reverse-compute the std-head bias so softplus(bias) + min_std == init_std.
    target_softplus = max(init_std - min_std, 1e-6)
    init_bias = math.log(math.exp(target_softplus) - 1.0)
    super().__init__(output_dim, init_bias, std_type="scalar")
    self._min_std = min_std

  def update(self, mlp_output: torch.Tensor) -> None:
    mean, scale_raw = torch.unbind(mlp_output, dim=-2)
    std = F.softplus(scale_raw) + self._min_std
    self._distribution = Normal(mean, std)
