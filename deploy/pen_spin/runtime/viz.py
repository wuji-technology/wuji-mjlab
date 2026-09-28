# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Mirror joint feedback, camera observations and reference poses in MuJoCo.

Wrist-local poses are lifted using the palm pose queried from the viewer model."""

from __future__ import annotations

import threading

import numpy as np
from lib.ghost_overlay import (
    add_axes,
    draw_main_model_geoms,
    find_object_visual_geom_ids,
)
from lib.math_utils import matrix_from_quat
from lib.motion_clip import RIGHT_FINGER_JOINT_NAMES
from lib.tag_frame import lift_wrist_local_to_scene

GHOST_ALPHA = 0.22
AXES_SCALE = 0.08


class SceneHandles:
    """qpos addresses and body/geom ids the per-tick update writes into.

    Resolved once. The ghost joint names are derived from the canonical list
    rather than re-spelled, because a locally spelled f-string name is exactly
    how the overlay silently died once before: no static check sees it, and the
    only symptom is one WARNING line and a window that renders nothing.
    """

    def __init__(self, model):
        import mujoco

        self.finger_qposadr = np.array(
            [model.joint(n).qposadr[0] for n in RIGHT_FINGER_JOINT_NAMES])

        ghost_names = [f"ghost_{n}" for n in RIGHT_FINGER_JOINT_NAMES]
        try:
            self.ghost_qposadr = np.array(
                [model.joint(n).qposadr[0] for n in ghost_names])
        except KeyError:
            self.ghost_qposadr = None

        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
        self.obj_qposadr = int(model.jnt_qposadr[jid]) if jid >= 0 else None
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.obj_body_id = bid if bid >= 0 else None
        self.obj_geom_ids = (find_object_visual_geom_ids(model, bid)
                             if bid >= 0 else None)

    @property
    def has_object(self) -> bool:
        return self.obj_geom_ids is not None and self.obj_geom_ids.size > 0


class Viewer:
    """Per-tick viewer update. ``update()`` is called from the control loop."""

    def __init__(self, model, data, viewer, handles, *, palm_pos, palm_quat):
        self.model = model
        self.data = data
        self.viewer = viewer
        self.h = handles
        self.palm_pos = palm_pos
        self.palm_quat = palm_quat
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        return self.viewer.is_running()

    def lift(self, obj_local) -> tuple[np.ndarray, np.ndarray]:
        """7-D wrist-local pose -> scene world, via the RENDERED palm."""
        p = np.asarray(obj_local, dtype=np.float64)
        return lift_wrist_local_to_scene(p[:3], p[3:7], self.palm_pos, self.palm_quat)

    def update(self, res, motion) -> None:
        """Push one tick into the scene and redraw the overlay.

        The two object poses are deliberately different sources:

          * **solid** object body <- ``res.obj_local``, the live camera pose the
            policy just observed. Falls back to the reference when the object is
            not being sensed (--obj-from-ref), where they legitimately coincide.
          * **ghost** object <- the reference clip at this frame.

        Their separation IS the measurement. Deriving both from the clip makes
        them agree by construction, and the window then confirms nothing.
        """
        import mujoco

        if not self.is_running():
            return
        ref_local = np.asarray(motion.object_pose_local[res.frame], dtype=np.float64)
        ref_pos_w, ref_quat_w = self.lift(ref_local)
        live_local = ref_local if res.obj_local is None else np.asarray(
            res.obj_local, dtype=np.float64)
        live_pos_w, live_quat_w = self.lift(live_local)

        with self._lock:
            self.data.qpos[self.h.finger_qposadr] = res.actual
            if self.h.ghost_qposadr is not None:
                self.data.qpos[self.h.ghost_qposadr] = motion.reference_qpos[res.frame]
            if self.h.obj_qposadr is not None:
                q0 = self.h.obj_qposadr
                self.data.qpos[q0:q0 + 3] = live_pos_w
                self.data.qpos[q0 + 3:q0 + 7] = live_quat_w
            mujoco.mj_forward(self.model, self.data)

        scn = self.viewer.user_scn
        scn.ngeom = 0  # reset, else decor accumulates every tick
        if self.h.has_object:
            ref_rot = matrix_from_quat(ref_quat_w)
            # Translucent copy at the REFERENCE pose.
            draw_main_model_geoms(scn, self.model, self.h.obj_geom_ids,
                                  ref_pos_w, ref_rot, alpha=GHOST_ALPHA)
            # Axes: solid at where the object actually is, translucent at the ref.
            real_rot = matrix_from_quat(self.data.xquat[self.h.obj_body_id])
            real_pos = self.data.xpos[self.h.obj_body_id].astype(np.float64)
            add_axes(scn, real_pos, real_rot, scale=AXES_SCALE, alpha=0.95, width=0.0040)
            add_axes(scn, ref_pos_w, ref_rot, scale=AXES_SCALE, alpha=0.35, width=0.0028)
        self.viewer.sync()

    def close(self) -> None:
        self.viewer.close()


def open_viewer(scn, *, key_callback=None):
    """Open a window on an ALREADY-BUILT scene.

    ``scn`` is a runtime.scene.Scene. It is passed in rather than built here
    because the policy path needs the same compiled model (home keyframe, palm
    pose, jmode override) whether or not a window is wanted — building a second
    one here would let the rendered scene and the observed scene drift apart,
    which is precisely the drift this window exists to detect.

    ``key_callback`` receives GLFW keycodes so the operator can drive the run
    without clicking back to the terminal (see runtime.safety.Keys).
    """
    import mujoco.viewer

    kw = {} if key_callback is None else {"key_callback": key_callback}
    viewer = mujoco.viewer.launch_passive(scn.model, scn.data, **kw)
    return Viewer(scn.model, scn.data, viewer, SceneHandles(scn.model),
                  palm_pos=scn.palm_pos, palm_quat=scn.palm_quat)
