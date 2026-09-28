# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Compose a deployment viewer from the shared robot and pen MJCF assets."""

from __future__ import annotations

import warnings
from pathlib import Path

import mujoco

from lib.hand2_model import load_hand_spec

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parents[2]
ASSETS = ROOT / "src" / "wuji_mjlab" / "assets"

# Robot and object XMLs are shared training assets. Do not copy task-specific
# deploy variants here: one public XML per physical asset is the release rule.
ROBOT_MJCF = ASSETS / "robots" / "wuji_hand2" / "mjcf" / "right_mjlab.xml"
SCENE_XML = _HERE.parent / "scenes" / "deploy.xml"
PEN_MJCF = ASSETS / "objects" / "aruco_pen250" / "xmls" / "pen_tracking.xml"

#: Calibrated tag100 -> physical wrist pose, metres and wxyz quaternion.
PALM_MOUNT_POS = (-0.08441049950924943, 0.00300004, -0.0309017758746512)
PALM_MOUNT_QUAT = (
    0.5416685725611728, 0.45452740016366616,
    0.45452740016366616, 0.5416685725611728,
)

ROBOT_MOUNT_BODY = "tag100_body"

def _attach(
    world: mujoco.MjSpec,
    child_path: Path,
    body_name: str | None,   # None -> worldbody
    *,
    prefix: str = "",
    pos=(0.0, 0.0, 0.0),
    quat=(1.0, 0.0, 0.0, 0.0),
) -> None:
    """Attach an entity while preserving its asset-relative mesh paths."""
    child = (load_hand_spec(child_path) if child_path == ROBOT_MJCF
             else mujoco.MjSpec.from_file(str(child_path)))
    parent = world.worldbody if body_name is None else world.body(body_name)
    if parent is None:
        raise KeyError(
            f"scene has no body {body_name!r} to attach {child_path.name} under; "
            f"the scene XML must declare it (see lib/scene_builder module docstring)"
        )
    frame = parent.add_frame()
    frame.pos = list(pos)
    frame.quat = list(quat)
    # The entities' <option> blocks conflict on attach; the scene is never
    # stepped, so silence that warning and keep real warnings visible.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Attach conflict.*")
        world.attach(child, prefix=prefix, frame=frame)


#: Translucent blue for the ghost overlay hand.
GHOST_RGBA = (0.2, 0.6, 1.0, 0.3)


def build(*, ghost: bool = False) -> mujoco.MjModel:
    """Compose and compile the deployment viewer scene.

    Args:
        ghost: Add a translucent, non-colliding reference hand.

    Returns:
        The compiled model for pose visualization."""
    world = mujoco.MjSpec.from_file(str(SCENE_XML))
    # MjSpec.attach mutates keyframes, so preserve them before adding joints.
    authored_keys = [(k.name, list(k.qpos), list(k.ctrl)) for k in world.keys]
    # Authored keyframes require the hand joints before the object freejoint.
    _attach(world, ROBOT_MJCF, ROBOT_MOUNT_BODY,
            pos=PALM_MOUNT_POS, quat=PALM_MOUNT_QUAT)
    if ghost:
        _attach(world, ROBOT_MJCF, ROBOT_MOUNT_BODY, prefix="ghost_",
                pos=PALM_MOUNT_POS, quat=PALM_MOUNT_QUAT)
        _make_ghost(world)
    _attach(world, PEN_MJCF, None)

    _rebuild_keyframes(world, authored_keys, ghost=ghost)
    _add_palm_frame_viz(world)
    return world.compile()


#: Finger DoF of one hand. The scene keyframe is authored for ONE hand plus the
#: object; the ghost copy adds this many more qpos slots in between.
ROBOT_NQ = 20


def _rebuild_keyframes(world: mujoco.MjSpec, authored, *, ghost: bool) -> None:
    """Restore keyframes after attachment changes the joint layout."""
    for key, (_name, qpos, ctrl) in zip(world.keys, authored, strict=True):
        hand, obj = qpos[:ROBOT_NQ], qpos[ROBOT_NQ:]
        key.qpos = hand + (hand if ghost else []) + obj
        key.ctrl = ctrl[:ROBOT_NQ]


def _make_ghost(world: mujoco.MjSpec) -> None:
    """Remove actuation and collisions from the reference hand overlay."""
    # Ghost actuators would change the control-array layout.
    for act in list(world.actuators):
        if act.name.startswith("ghost_"):
            world.delete(act)
    for body in world.bodies:
        if not body.name.startswith("ghost_"):
            continue
        for geom in list(body.geoms):
            # Collision geoms carry contype/conaffinity; the overlay has none.
            # (Element removal is spec.delete(el) — MjsGeom has no .delete().)
            if geom.contype or geom.conaffinity:
                world.delete(geom)
                continue
            geom.rgba = list(GHOST_RGBA)
            geom.density = 0.0


def _add_palm_frame_viz(world: mujoco.MjSpec) -> None:
    """Add a wrist-frame marker beneath the attached palm."""
    palm = world.body("right_palm_link")
    if palm is None:                       # entity not attached (shouldn't happen)
        return
    frame_body = palm.add_body(name="palm_frame_body", pos=[0.0, 0.0, 0.05])
    frame_body.add_site(name="palm_frame", pos=[0.0, 0.0, 0.0], group=0,
                        type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005, 0, 0],
                        rgba=[1, 1, 0, 1])
    # Assign quaternions directly: geom.alt.euler does not update these spec geoms.
    R = 0.7071067811865476
    axes = (("x", [0.02, 0, 0], [R, 0, R, 0], [1, 0, 0, 0.9]),
            ("y", [0, 0.02, 0], [R, R, 0, 0], [0, 1, 0, 0.9]),
            ("z", [0, 0, 0.02], [1, 0, 0, 0], [0, 0, 1, 0.9]))
    for axis, pos, quat, rgba in axes:
        frame_body.add_geom(
            name=f"palm_frame_{axis}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=pos, quat=quat, size=[0.003, 0.02, 0], rgba=rgba, group=0,
            contype=0, conaffinity=0, mass=0.0,
        )
