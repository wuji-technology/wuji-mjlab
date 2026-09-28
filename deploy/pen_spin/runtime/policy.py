# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Load the single fixed-wrist pen-spin policy contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lib.onnx_policy import ONNXPolicy


@dataclass(frozen=True)
class Loaded:
    policy: ONNXPolicy
    ctrl_dt: float
    action_scale: float
    clip_lo: float
    clip_hi: float
    alpha: float

    @property
    def obs_dim(self) -> int:
        return self.policy.obs_dim

    def infer(self, obs: np.ndarray) -> np.ndarray:
        return self.policy.infer(obs)

    def describe(self) -> str:
        return (
            f"fixed-wrist pen-spin obs={self.obs_dim}D  "
            f"{1.0 / self.ctrl_dt:.0f}Hz  scale={self.action_scale} "
            f"clip=[{self.clip_lo},{self.clip_hi}] alpha={self.alpha}"
        )


def load(policy_path: str, *, device: str = "cpu") -> Loaded:  # noqa: ARG001
    policy = ONNXPolicy(policy_path)
    lo, hi = policy.action_residual_clip
    return Loaded(
        policy=policy,
        ctrl_dt=policy.ctrl_dt,
        action_scale=policy.action_scale,
        clip_lo=lo,
        clip_hi=hi,
        alpha=policy.action_ema_alpha,
    )
