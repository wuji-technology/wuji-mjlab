# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import numpy as np
from wuji_reorient_deploy.drivers.hand2 import _mit_params_value


def test_scalar_kp_kd_returns_a_single_tuple():
  assert _mit_params_value(3.0, 0.2, num_joints=20) == (3.0, 0.2)


def test_array_kp_kd_returns_one_tuple_per_joint_in_order():
  kp = np.array([1.0, 2.0, 3.0])
  kd = np.array([0.1, 0.2, 0.3])
  assert _mit_params_value(kp, kd, num_joints=3) == [(1.0, 0.1), (2.0, 0.2), (3.0, 0.3)]


def test_array_kp_with_scalar_kd_broadcasts_kd_to_every_joint():
  kp = np.array([1.0, 2.0, 3.0])
  assert _mit_params_value(kp, 0.15, num_joints=3) == [
    (1.0, 0.15),
    (2.0, 0.15),
    (3.0, 0.15),
  ]


def test_mismatched_array_length_raises():
  kp = np.array([1.0, 2.0])
  try:
    _mit_params_value(kp, 0.1, num_joints=20)
  except ValueError:
    return
  raise AssertionError("expected a ValueError for a length-mismatched kp array")
