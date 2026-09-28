# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Physical visibility of observed source patterns, in the pen body frame."""
import numpy as np


def _box_blocks(eye, rays, center, half):
    # Slightly inset solids avoid treating the marker's own surface as a blocker.
    low, high = center - half + 5e-5, center + half - 5e-5
    parallel = np.abs(rays) < 1e-12
    outside = np.any(parallel & ((eye < low) | (eye > high)), axis=1)
    a = np.divide(low-eye, rays, out=np.full_like(rays, -np.inf), where=~parallel)
    b = np.divide(high-eye, rays, out=np.full_like(rays, np.inf), where=~parallel)
    enter = np.maximum(np.minimum(a, b).max(axis=1), 0.0)
    leave = np.minimum(np.maximum(a, b).min(axis=1), 1.0-1e-6)
    return (~outside) & (enter < leave)


def _shaft_blocks(eye, rays, radius, half_length):
    radius, half_length = radius-5e-5, half_length-5e-5
    a = np.sum(rays[:, :2] ** 2, axis=1)
    b = 2 * (rays[:, :2] @ eye[:2])
    c = eye[:2] @ eye[:2] - radius**2
    disc = b*b - 4*a*c
    parallel = a < 1e-20
    root = np.sqrt(np.maximum(disc, 0))
    lo = np.divide(-b-root, 2*a, out=np.full_like(a, -np.inf), where=~parallel)
    hi = np.divide(-b+root, 2*a, out=np.full_like(a, np.inf), where=~parallel)
    radial_valid = np.where(parallel, c < 0, disc > 0)
    zparallel = np.abs(rays[:, 2]) < 1e-12
    zlo = np.divide(-half_length-eye[2], rays[:, 2],
                    out=np.full_like(a, -np.inf), where=~zparallel)
    zhi = np.divide(half_length-eye[2], rays[:, 2],
                    out=np.full_like(a, np.inf), where=~zparallel)
    zvalid = (~zparallel) | (abs(eye[2]) < half_length)
    enter = np.maximum(np.maximum(lo, np.minimum(zlo, zhi)), 0.0)
    leave = np.minimum(np.minimum(hi, np.maximum(zlo, zhi)), 1.0-1e-6)
    return radial_valid & zvalid & (enter < leave)


def visibility_failure(layout, ids, rotation, translation):
    """Empty string iff every observed pattern is in front, facing and unblocked.

    This checks the two heads and shaft, not external hands/table geometry.
    Both planar pose hypotheses can pass; visibility alone cannot resolve that.
    """
    if not (np.isfinite(rotation).all() and np.isfinite(translation).all()):
        return "nonfinite pose"
    corners = np.stack([layout.corners[tid] for tid in ids])
    camera_points = corners @ rotation.T + translation.reshape(3)
    if np.any(camera_points[..., 2] <= 1e-6):
        return "behind camera"
    eye = -rotation.T @ translation.reshape(3)
    centers = corners.mean(axis=1)
    facing = np.sum(layout.outward_normals_for(ids) * (eye-centers), axis=1)
    if np.any(facing <= 1e-6):
        return "back-facing marker"
    # Sample the centre and four interior points, avoiding raster boundary noise.
    samples = np.concatenate([centers[:, None], centers[:, None]+.6*(corners-centers[:, None])], axis=1)
    rays = samples.reshape(-1, 3)-eye
    for name, center in layout.box_centers.items():
        if _box_blocks(eye, rays, center, layout.box_half_sizes[name]).any():
            return "marker occluded by head"
    if _shaft_blocks(eye, rays, layout.shaft_radius, layout.shaft_half_length).any():
        return "marker occluded by shaft"
    return ""
