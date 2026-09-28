# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
def get_wuji_hand2_cfg(hand_side: str = "right"):
  from .wuji_hand2_cfg import get_wuji_hand2_cfg as _get_wuji_hand2_cfg

  return _get_wuji_hand2_cfg(hand_side=hand_side)
