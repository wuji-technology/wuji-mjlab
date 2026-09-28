# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Shared tensor math helpers."""

from __future__ import annotations

import math

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse


def rot_xy_from_quat(quat: torch.Tensor) -> torch.Tensor:
  rot = matrix_from_quat(quat)
  x_axis = rot[..., :, 0]
  y_axis = rot[..., :, 1]
  return torch.cat((x_axis, y_axis), dim=-1)


def tolerance(
  x: torch.Tensor,
  bounds: tuple[float, float],
  margin: float,
  sigmoid: str = "linear",
) -> torch.Tensor:
  """Plateau-and-decay shaping helper for scalar tensor errors."""
  lower, upper = bounds
  in_bounds = (x >= lower) & (x <= upper)

  if margin <= 0.0:
    return torch.where(in_bounds, torch.ones_like(x), torch.zeros_like(x))

  dist = torch.where(x < lower, lower - x, x - upper)

  if sigmoid == "linear":
    value = 1.0 - dist / margin
  elif sigmoid == "gaussian":
    value = torch.exp(-0.5 * (dist / margin) ** 2)
  else:
    raise ValueError(f"Unsupported sigmoid: {sigmoid}")

  return torch.where(in_bounds, torch.ones_like(x), value.clamp(0.0, 1.0))


def random_quat_uniform(n: int, device: torch.device | str = "cpu") -> torch.Tensor:
  """Sample *n* quaternions uniformly on SO(3) using Shoemake's method.

  Returns:
    Quaternions in (w, x, y, z) convention.
  """
  u1, u2, u3 = torch.rand(3, n, device=device)
  sqrt1 = torch.sqrt(1.0 - u1)
  sqrt_u1 = torch.sqrt(u1)
  a = 2.0 * math.pi * u2
  b = 2.0 * math.pi * u3
  w = sqrt1 * torch.sin(a)
  x = sqrt1 * torch.cos(a)
  y = sqrt_u1 * torch.sin(b)
  z = sqrt_u1 * torch.cos(b)
  return torch.stack([w, x, y, z], dim=-1)


def points_in_object_frame(
  points_w: torch.Tensor,
  object_pos_w: torch.Tensor,
  object_quat_w: torch.Tensor,
) -> torch.Tensor:
  """Transform world-frame points into an object's local frame."""
  batch_size, num_points = points_w.shape[:2]
  rel_w = points_w - object_pos_w.unsqueeze(1)
  quat = object_quat_w.unsqueeze(1).expand(-1, num_points, -1)
  return quat_apply_inverse(
    quat.reshape(batch_size * num_points, 4),
    rel_w.reshape(batch_size * num_points, 3),
  ).reshape(batch_size, num_points, 3)


def point_velocities_in_object_frame(
  points_w: torch.Tensor,
  points_lin_vel_w: torch.Tensor,
  object_pos_w: torch.Tensor,
  object_quat_w: torch.Tensor,
  object_lin_vel_w: torch.Tensor,
  object_ang_vel_w: torch.Tensor,
) -> torch.Tensor:
  """Transform world-frame point velocities into an object's local frame."""
  batch_size, num_points = points_w.shape[:2]
  rel_pos_w = points_w - object_pos_w.unsqueeze(1)
  rel_vel_w = points_lin_vel_w - object_lin_vel_w.unsqueeze(1)
  rel_vel_w = rel_vel_w - torch.cross(
    object_ang_vel_w.unsqueeze(1).expand_as(rel_pos_w),
    rel_pos_w,
    dim=-1,
  )
  quat = object_quat_w.unsqueeze(1).expand(-1, num_points, -1)
  return quat_apply_inverse(
    quat.reshape(batch_size * num_points, 4),
    rel_vel_w.reshape(batch_size * num_points, 3),
  ).reshape(batch_size, num_points, 3)


def contact_mask_from_sensor(
  sensor,
  *,
  found_threshold: float = 0.5,
  force_threshold: float = 1.0e-4,
) -> torch.Tensor | None:
  """Aggregate contact detection methods into a single float mask."""
  data = sensor.data
  contact: torch.Tensor | None = None

  if data.found is not None:
    contact = (data.found > found_threshold).float()

  if data.force is not None:
    force_contact = (torch.linalg.norm(data.force, dim=-1) > force_threshold).float()
    contact = (
      force_contact if contact is None else torch.maximum(contact, force_contact)
    )

  if data.force_history is not None:
    history_contact = (
      (torch.linalg.norm(data.force_history, dim=-1) > force_threshold)
      .any(dim=-1)
      .float()
    )
    contact = (
      history_contact if contact is None else torch.maximum(contact, history_contact)
    )

  return contact
