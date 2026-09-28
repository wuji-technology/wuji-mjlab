# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""The MuJoCo scene — built on EVERY run, not only with --sim.

Two values the policy path needs come from the compiled model, which is why this
cannot live inside the optional viewer:

  * ``home_qpos`` — the physical ease-to-start seed. The pen-spin observation
    uses the training task's explicit all-zero default joint position instead.
  * ``palm_pos`` / ``palm_quat`` — the rendered palm pose the viewer lifts the
    object with, and the thing ``--jmode`` actually changes.

``--jmode`` selects the calibrated wrist pose for the inclined or flat-hand
mount. Both the real and the ghost palm are patched so
rendering, the palm cache, and what the policy assumes cannot disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lib.motion_clip import RIGHT_FINGER_JOINT_NAMES

# Physical wrist pose for the flat-hand scene mount, metres and wxyz.
# Scene placement is independent of the camera's tag-to-wrist calibration.
NEG90_MOUNT_POS = (-0.0847999, 0.00300004, -0.025949842)
NEG90_MOUNT_QUAT = (0.5, 0.5, 0.5, 0.5)
PALM_BODY = "right_palm_link"
GHOST_PALM_BODY = "ghost_right_palm_link"


@dataclass
class Scene:
    model: object
    data: object
    home_qpos: np.ndarray      # (20,) float32 — the joint_pos_rel offset
    finger_qposadr: np.ndarray
    palm_pos: np.ndarray
    palm_quat: np.ndarray
    jmode: str

    def describe(self) -> str:
        return (f"jmode={self.jmode} palm_pos={np.round(self.palm_pos, 6).tolist()} "
                f"palm_quat={np.round(self.palm_quat, 6).tolist()}")


def apply_jmode(model, jmode: str) -> list[str]:
    """Patch the palm mount angle. Returns the bodies actually overridden."""
    import mujoco

    if jmode == "default":
        return []           # scene_builder supplies the inclined mount pose
    if jmode != "neg90":
        raise ValueError(f"unknown jmode {jmode!r} (want 'neg90' or 'default')")
    done = []
    for name in (PALM_BODY, GHOST_PALM_BODY):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            # The ghost palm only exists in a ghost build; a missing REAL palm is
            # a broken scene, not an optional feature.
            if name == PALM_BODY:
                raise RuntimeError(f"scene has no '{PALM_BODY}' body to override")
            continue
        model.body_pos[bid] = NEG90_MOUNT_POS
        model.body_quat[bid] = NEG90_MOUNT_QUAT
        done.append(name)
    return done


def build(*, jmode: str = "neg90", ghost: bool = False) -> Scene:
    """Compose the scene, apply the mount override, and read what the run needs.

    ``ghost`` attaches the translucent overlay copy — only useful with a viewer,
    but harmless otherwise. The mount override must happen BEFORE mj_forward,
    since the palm pose is what it changes.
    """
    import mujoco
    from lib import scene_builder

    model = scene_builder.build(ghost=ghost)
    apply_jmode(model, jmode)
    data = mujoco.MjData(model)
    # MjData starts at qpos0 — all zeros for the fingers — NOT at the home
    # keyframe. Reading default_finger_qpos without this reset silently yields
    # zeros, which is 0.8 rad off on the very first joint. The keyframe is the
    # one scene_builder._rebuild_keyframes() reassembles after the attaches.
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    adr = np.array([model.joint(n).qposadr[0] for n in RIGHT_FINGER_JOINT_NAMES])
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PALM_BODY)
    if bid < 0:
        raise RuntimeError(f"scene has no '{PALM_BODY}' body")
    return Scene(
        model=model,
        data=data,
        home_qpos=data.qpos[adr].astype(np.float32).copy(),
        finger_qposadr=adr,
        palm_pos=data.xpos[bid].astype(np.float64).copy(),
        palm_quat=data.xquat[bid].astype(np.float64).copy(),
        jmode=jmode,
    )
