# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Reference ghost overlay helpers for the optional MuJoCo viewer.

The hand ghost is attached as a second copy of the shared robot entity and the
pen pose is drawn from the same local-frame data used by the policy.

"""

from __future__ import annotations

import mujoco
import numpy as np


def find_object_visual_geom_ids(main_model: mujoco.MjModel, body_id: int) -> np.ndarray:
    """Return MAIN-model geom ids on body_id that are visible meshes."""
    ids: list[int] = []
    for g in range(main_model.ngeom):
        if int(main_model.geom_bodyid[g]) == body_id:
            ids.append(g)
    ids_arr = np.asarray(ids, dtype=np.int64)
    if ids_arr.size == 0:
        return ids_arr
    mesh_kind = int(mujoco.mjtGeom.mjGEOM_MESH)
    is_mesh = main_model.geom_type[ids_arr] == mesh_kind
    is_visible = main_model.geom_rgba[ids_arr, 3] > 0.0
    visible_meshes = ids_arr[is_mesh & is_visible]
    if visible_meshes.size > 0:
        return visible_meshes
    visible_any = ids_arr[is_visible]
    return visible_any if visible_any.size > 0 else ids_arr


def draw_main_model_geoms(
    scn: mujoco.MjvScene,
    main_model: mujoco.MjModel,
    geom_ids: np.ndarray,
    body_pos: np.ndarray,
    body_rot: np.ndarray,
    alpha: float,
) -> None:
    """Inject decoration copies of selected main-model geoms at a target pose.

    body_pos / body_rot define the world pose of the (notional) body the geoms
    belong to. Each geom's local pos/quat (from main_model) is composed with
    that pose. Mesh geoms get the `2 * dataid` quirk (mjvScene stores 2 entries
    per mesh — render mesh + convex hull).
    """
    body_pos = np.asarray(body_pos, dtype=np.float64)
    body_rot = np.asarray(body_rot, dtype=np.float64)
    for geom_id in geom_ids:
        if scn.ngeom >= scn.maxgeom:
            return
        geom_id = int(geom_id)
        geom_type = int(main_model.geom_type[geom_id])
        local_pos = main_model.geom_pos[geom_id].astype(np.float64)
        local_rot_flat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(
            local_rot_flat, main_model.geom_quat[geom_id].astype(np.float64)
        )
        local_rot = local_rot_flat.reshape(3, 3)
        world_pos = body_pos + body_rot @ local_pos
        world_rot = body_rot @ local_rot
        rgba = main_model.geom_rgba[geom_id].copy().astype(np.float32)
        rgba[3] = alpha

        geom = scn.geoms[scn.ngeom]
        scn.ngeom += 1
        mujoco.mjv_initGeom(
            geom,
            type=geom_type,
            size=main_model.geom_size[geom_id].astype(np.float64),
            pos=world_pos,
            mat=world_rot.reshape(-1),
            rgba=rgba,
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            geom.dataid = 2 * int(main_model.geom_dataid[geom_id])
            geom.matid = int(main_model.geom_matid[geom_id])
            geom.texcoord = 1


def add_axes(
    scn: mujoco.MjvScene,
    origin: np.ndarray,
    rotation: np.ndarray,
    scale: float,
    alpha: float = 0.95,
    width: float = 0.0035,
) -> None:
    """Append RGB axis arrows at (origin, rotation) to scn user_scn-style scene."""
    colors = (
        np.array([0.95, 0.10, 0.10, alpha], dtype=np.float32),  # X red
        np.array([0.10, 0.95, 0.10, alpha], dtype=np.float32),  # Y green
        np.array([0.10, 0.10, 0.95, alpha], dtype=np.float32),  # Z blue
    )
    for axis_idx in range(3):
        if scn.ngeom >= scn.maxgeom:
            return
        end = origin + rotation[:, axis_idx] * scale
        geom = scn.geoms[scn.ngeom]
        scn.ngeom += 1
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        mujoco.mjv_initGeom(
            geom,
            type=int(mujoco.mjtGeom.mjGEOM_ARROW),
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.zeros(9),
            rgba=colors[axis_idx],
        )
        mujoco.mjv_connector(
            geom,
            type=int(mujoco.mjtGeom.mjGEOM_ARROW),
            width=width,
            from_=origin.astype(np.float64),
            to=end.astype(np.float64),
        )
