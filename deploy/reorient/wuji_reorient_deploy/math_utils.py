# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Minimal numpy quaternion math."""
from __future__ import annotations

import numpy as np


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Hamilton product a ⊗ b, both (w,x,y,z)."""
  aw, ax, ay, az = a
  bw, bx, by, bz = b
  return np.array([
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ], dtype=np.float64)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
  """Return the conjugate of a quaternion."""
  return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_inv(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
  """Return the numerically guarded inverse of a quaternion."""
  return quat_conjugate(q) / max(float(np.dot(q, q)), eps)


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
  """(w,x,y,z) → 3×3 rotation matrix (pytorch3d convention, row-major)."""
  r, i, j, k = q
  two_s = 2.0 / float(np.dot(q, q))
  return np.array([
    [1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r)],
    [two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r)],
    [two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)],
  ], dtype=np.float64)


def rot6d_from_quat(q: np.ndarray) -> np.ndarray:
  """Last 6 of the flattened row-major rotation matrix (rows 2 & 3)."""
  return matrix_from_quat(q).reshape(9)[3:]


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate vector v by quaternion q (w,x,y,z)."""
  return matrix_from_quat(q) @ np.asarray(v, dtype=np.float64)
