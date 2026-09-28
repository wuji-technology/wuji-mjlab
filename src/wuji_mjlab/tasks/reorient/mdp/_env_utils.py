# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import torch


def resolve_env_ids(env, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  if isinstance(env_ids, slice):
    return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
  return env_ids.to(env.device, dtype=torch.long)
