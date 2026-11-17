# SPDX-License-Identifier: Apache-2.0
"""Reflection across Y for the corresponding Wuji Hand 2 palm frames."""

import torch


def points_y(points: torch.Tensor) -> torch.Tensor:
  """Reflect polar vectors, including wrist-local object and link positions."""
  return points * points.new_tensor((1.0, -1.0, 1.0))


def quat_y(quat: torch.Tensor) -> torch.Tensor:
  """Return the proper rotation S R S, with S = diag(1, -1, 1), in wxyz."""
  return quat * quat.new_tensor((1.0, -1.0, 1.0, -1.0))


def rotation_error_y(error: torch.Tensor) -> torch.Tensor:
  """Reflect the last two rows of a flattened rotation matrix."""
  return error * error.new_tensor((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0))
