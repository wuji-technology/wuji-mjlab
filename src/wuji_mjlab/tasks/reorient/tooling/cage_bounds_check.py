# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Interactive Wuji Hand 2 cage-boundary check at the reorient home pose."""

from __future__ import annotations

import re
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np
import torch

from wuji_mjlab.tasks.reorient.config.wuji_hand2.env_cfgs import (
  wuji_hand2_reorient_env_cfg,
)
from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
  REORIENT_HAND2_ROOT_POS,
  REORIENT_HAND2_ROOT_ROT,
)
from wuji_mjlab.tasks.reorient.mdp.cage import (
  _CAGE_UP_MARGIN,
  _compute_cage_bounds_in_palm,
)
from wuji_mjlab.tasks.reorient.tooling.hand2_check_scene import (
  add_viewer_environment,
  attach_robot,
  hold_home_pose,
)

_EXPECTED_RAW_LO = np.array((-0.0426, -0.0083, -0.1295))
_EXPECTED_RAW_HI = np.array((0.0352, 0.0685, 0.0))
_EXPECTED_LO = np.array((-0.0526, -0.0183, -0.1395))
_EXPECTED_HI = np.array((0.0452, 0.0985, 0.0100))
_EXPECTED_CUBE_IN_PALM = np.array((0.00419996, 0.061349842, -0.0795001))
_REFERENCE_ATOL = 1.5e-4


@dataclass(frozen=True)
class CageBoundsCheckConfig:
  """Interactive cage-boundary options."""

  box_opacity: float = 0.12


def _body_ids_matching(
  model: mujoco.MjModel, patterns: str | tuple[str, ...]
) -> list[int]:
  if isinstance(patterns, str):
    patterns = (patterns,)
  regexes = tuple(re.compile(pattern) for pattern in patterns)
  return [
    body_id
    for body_id in range(model.nbody)
    if (name := model.body(body_id).name)
    and any(regex.fullmatch(name) for regex in regexes)
  ]


def _home_cage_bounds(
  robot: mujoco.MjSpec,
  *,
  body_patterns: str | tuple[str, ...],
  margin: float,
  up_axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  model = robot.compile()
  data = mujoco.MjData(model)
  hold_home_pose(model, data, ("",))
  mujoco.mj_forward(model, data)

  body_ids = _body_ids_matching(model, body_patterns)
  if len(body_ids) != 21:
    raise ValueError(f"expected 21 cage bodies, got {len(body_ids)}")
  palm_id = model.body("right_palm_link").id
  palm_pos = data.xpos[palm_id]
  palm_rot = data.xmat[palm_id].reshape(3, 3)
  hand_in_palm = np.stack(
    [palm_rot.T @ (data.xpos[body_id] - palm_pos) for body_id in body_ids]
  )
  raw_lo = hand_in_palm.min(axis=0)
  raw_hi = hand_in_palm.max(axis=0)
  lo, hi = _compute_cage_bounds_in_palm(
    torch.from_numpy(hand_in_palm).unsqueeze(0),
    margin=margin,
    up_margin=_CAGE_UP_MARGIN,
    up_axis=up_axis,
  )
  return raw_lo, raw_hi, lo[0].numpy(), hi[0].numpy()


def _add_cage_visuals(
  robot: mujoco.MjSpec,
  *,
  raw_lo: np.ndarray,
  raw_hi: np.ndarray,
  lo: np.ndarray,
  hi: np.ndarray,
  margin: float,
  up_axis: int,
  opacity: float,
) -> None:
  if not 0.0 < opacity < 1.0:
    raise ValueError(f"box_opacity must be in (0, 1), got {opacity}")
  palm = robot.body("right_palm_link")

  ordinary_lo = raw_lo - margin
  ordinary_hi = raw_hi + margin
  palm.add_geom(
    name="cage_margin_1cm",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=0.5 * (ordinary_lo + ordinary_hi),
    size=0.5 * (ordinary_hi - ordinary_lo),
    rgba=(0.1, 0.45, 1.0, opacity),
    contype=0,
    conaffinity=0,
    group=1,
    density=0,
  )

  extra_lo = ordinary_lo.copy()
  extra_lo[up_axis] = ordinary_hi[up_axis]
  extra_hi = ordinary_hi.copy()
  extra_hi[up_axis] = hi[up_axis]
  palm.add_geom(
    name="cage_extra_up_margin_2cm",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=0.5 * (extra_lo + extra_hi),
    size=0.5 * (extra_hi - extra_lo),
    rgba=(0.1, 1.0, 0.25, min(1.0, 2.5 * opacity)),
    contype=0,
    conaffinity=0,
    group=1,
    density=0,
  )

  face_size = 0.5 * (hi - lo)
  face_size[up_axis] = 0.0006
  face_pos = 0.5 * (lo + hi)
  face_pos[up_axis] = hi[up_axis]
  palm.add_geom(
    name="cage_up_margin_3cm_face",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=face_pos,
    size=face_size,
    rgba=(0.05, 1.0, 0.15, 0.75),
    contype=0,
    conaffinity=0,
    group=1,
    density=0,
  )


def build_cage_bounds_check_model(
  box_opacity: float = 0.12,
) -> tuple[mujoco.MjModel, dict[str, np.ndarray | float | int]]:
  """Compile the production cage bounds as palm-frame visual geoms."""
  env_cfg = wuji_hand2_reorient_env_cfg(num_envs=1)
  cage_cfg = env_cfg.rewards["cage_escape"]
  margin = float(cage_cfg.params["margin"])
  up_axis = int(cage_cfg.params["up_axis"])
  body_patterns = cage_cfg.params["robot_cfg"].body_names
  if margin != 0.01 or _CAGE_UP_MARGIN != 0.03 or up_axis != 1:
    raise ValueError(
      f"unexpected cage parameters: margin={margin}, "
      f"up_margin={_CAGE_UP_MARGIN}, up_axis={up_axis}"
    )

  robot = env_cfg.scene.entities["robot"].spec_fn()
  raw_lo, raw_hi, lo, hi = _home_cage_bounds(
    robot,
    body_patterns=body_patterns,
    margin=margin,
    up_axis=up_axis,
  )
  np.testing.assert_allclose(raw_lo, _EXPECTED_RAW_LO, atol=_REFERENCE_ATOL)
  np.testing.assert_allclose(raw_hi, _EXPECTED_RAW_HI, atol=_REFERENCE_ATOL)
  np.testing.assert_allclose(lo, _EXPECTED_LO, atol=_REFERENCE_ATOL)
  np.testing.assert_allclose(hi, _EXPECTED_HI, atol=_REFERENCE_ATOL)
  _add_cage_visuals(
    robot,
    raw_lo=raw_lo,
    raw_hi=raw_hi,
    lo=lo,
    hi=hi,
    margin=margin,
    up_axis=up_axis,
    opacity=box_opacity,
  )

  scene = mujoco.MjSpec()
  scene.option.timestep = 0.01
  attach_robot(
    scene,
    robot,
    prefix="robot/",
    root_pos=REORIENT_HAND2_ROOT_POS,
    root_rot=REORIENT_HAND2_ROOT_ROT,
    x_shift=0.0,
  )
  add_viewer_environment(scene)
  model = scene.compile()

  data = mujoco.MjData(model)
  hold_home_pose(model, data, ("robot/",))
  mujoco.mj_forward(model, data)
  palm_id = model.body("robot/right_palm_link").id
  cube_id = model.body("robot_cube_reference").id
  palm_pos = data.xpos[palm_id]
  palm_rot = data.xmat[palm_id].reshape(3, 3)
  cube_in_palm = palm_rot.T @ (data.xpos[cube_id] - palm_pos)
  np.testing.assert_allclose(cube_in_palm, _EXPECTED_CUBE_IN_PALM, atol=_REFERENCE_ATOL)

  values: dict[str, np.ndarray | float | int] = {
    "raw_lo": raw_lo,
    "raw_hi": raw_hi,
    "lo": lo,
    "hi": hi,
    "cube_in_palm": cube_in_palm,
    "margin": margin,
    "up_margin": _CAGE_UP_MARGIN,
    "up_axis": up_axis,
  }
  return model, values


def launch_cage_bounds_check(config: CageBoundsCheckConfig) -> None:
  """Launch the native interactive cage-boundary viewer."""
  model, values = build_cage_bounds_check_model(box_opacity=config.box_opacity)
  data = mujoco.MjData(model)
  hold_home_pose(model, data, ("robot/",))
  mujoco.mj_forward(model, data)

  print("Compiled cage-boundary check: pose=home, body_origins=21")
  print(
    "raw_lo=" + np.array2string(values["raw_lo"], precision=6) + " "
    "raw_hi=" + np.array2string(values["raw_hi"], precision=6)
  )
  print(
    "cage_lo=" + np.array2string(values["lo"], precision=6) + " "
    "cage_hi=" + np.array2string(values["hi"], precision=6)
  )
  print(
    "cube_in_palm="
    + np.array2string(values["cube_in_palm"], precision=6)
    + " up_axis=Y(1)"
  )
  print("Legend: BLUE = ordinary 1 cm AABB; GREEN = extra Y-hi slab and 3 cm face.")
  print("The cage bounds body origins, not the rendered mesh surface.")
  mujoco.viewer.launch(model, data)
