# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""rsl_rl backend configuration for the Wuji Hand Reorient task.

Exposes the PPO runner cfg factory consumed by the task registration in
``config/wuji_hand/__init__.py``. RL configs are kept under a backend-specific
subdir so the layout extends naturally if another backend is added.
"""

from .ppo import wuji_hand_reorient_ppo_runner_cfg

__all__ = ["wuji_hand_reorient_ppo_runner_cfg"]
