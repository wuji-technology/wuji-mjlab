# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Source pen (+Z shaft) versus legacy tracking pen (+X shaft) body frames.

Same basis change as the new training setup's get_tracking_pen_spec: rotate
source geometry Ry(+90), and right-multiply measured poses by Ry(-90).
This changes orientation coordinates, never the pen center or wrist frame.
"""
import numpy as np

from lib.math_utils import normalize_quat, quat_mul

TRACKING_IN_SOURCE_QUAT = np.array([2**-.5, 0., -2**-.5, 0.])


def source_to_tracking(quat):
    return normalize_quat(quat_mul(np.asarray(quat), TRACKING_IN_SOURCE_QUAT))
