# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Common Wuji Hand robot config helpers.

The Wuji Hand 1 MJCF and meshes are read from the ``wuji-description``
submodule next to this package (``hand/body-with-soft``). ``_get_spec`` applies
the task's edits to it at load time and leaves the files untouched.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco

if TYPE_CHECKING:
  from mjlab.entity import EntityCfg

_ASSET_DIR = Path(__file__).resolve().parent
WUJI_HAND_DESCRIPTION_DIR = (
  _ASSET_DIR.parent / "wuji-description" / "hand" / "body-with-soft"
)
_UPSTREAM_MODEL = "wujihand-right"
_TASK_MODEL = "wujihand-right-mjlab"
_COLLISION_GEOMS = (
  "right_palm_collision",
  "right_finger1_link2_col",
  "right_finger1_link2_softbody_col",
  "right_finger1_link3_col",
  "right_finger1_link4_col",
  *(f"right_finger{f}_link{link}_col" for f in range(2, 6) for link in (2, 3, 4)),
)
# Fingertip sites on each distal link, in its body frame.
_TIP_SITES = {
  "right_finger1_link4": (0.00026399, -0.00020119, 0.032808),
  "right_finger2_link4": (-0.000749345, 0.0, 0.024),
  "right_finger3_link4": (-0.00075, 0.0, 0.024),
  "right_finger4_link4": (-0.000811164, 0.0, 0.0239995),
  "right_finger5_link4": (-0.000828016, 0.0, 0.0259994),
}
_PALM_EXCLUDES = (
  "right_finger1_link1",
  "right_finger1_link2",
  "right_finger2_link2",
  "right_finger3_link2",
  "right_finger4_link2",
  "right_finger5_link2",
)


def update_assets(
  assets: dict,
  path: Path,
  meshdir: str | None = None,
  glob: str = "*",
  recursive: bool = False,
) -> None:
  """Read files from a directory and stuff their bytes into ``assets``."""
  for f in Path(path).glob(glob):
    if f.is_file():
      asset_key = f"{meshdir}/{f.name}" if meshdir else f.name
      assets[asset_key] = f.read_bytes()
    elif f.is_dir() and recursive:
      update_assets(assets, f, meshdir, glob, recursive)


def wuji_hand_xml_path(hand_side: str = "right") -> Path:
  """Return the Wuji Hand 1 MJCF path."""
  if hand_side != "right":
    raise ValueError(f"Unsupported hand_side '{hand_side}'. Only 'right' is supported.")
  return WUJI_HAND_DESCRIPTION_DIR / "mjcf" / f"{hand_side}_simplified.xml"


def _get_assets(xml_path: Path, spec: mujoco.MjSpec) -> dict[str, bytes]:
  mesh_dir = xml_path.parent / spec.meshdir
  return {
    f"{spec.meshdir}/{mesh.file}" if spec.meshdir else mesh.file: (
      mesh_dir / mesh.file
    ).read_bytes()
    for mesh in spec.meshes
  }


def _apply_task_edits(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Apply the task's edits: options, contact defaults, names, sites, excludes."""
  if spec.modelname != _UPSTREAM_MODEL or spec.sites or spec.excludes:
    raise ValueError(
      f"Wuji Hand 1 MJCF {spec.modelname!r} is not the upstream hand the task "
      "edits expect (model name, no sites, no excludes)"
    )
  spec.modelname = _TASK_MODEL

  builtin = mujoco.MjSpec()
  spec.option.jacobian = builtin.option.jacobian
  spec.option.tolerance = builtin.option.tolerance
  spec.option.timestep = 0.01
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
  spec.option.iterations = 5
  spec.option.ls_iterations = 8
  spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP

  # Contact defaults for every geom the upstream MJCF leaves at MuJoCo's.
  default = builtin.default.geom
  for geom in spec.geoms:
    if (
      list(geom.friction) != list(default.friction)
      or list(geom.solref) != list(default.solref)
      or list(geom.solimp) != list(default.solimp)
      or geom.priority != default.priority
    ):
      raise ValueError(f"Wuji Hand 1 geom on {geom.parent.name!r} sets its own contact")
    if geom.group == default.group:
      geom.group = 3
    geom.friction = (0.7, 0.005, 0.0001)
    geom.solref = (0.02, 1.5)
    geom.priority = 1

  for geom in spec.geoms:
    if not (geom.contype or geom.conaffinity):
      continue
    body = geom.parent.name
    if body == "right_palm_link":
      geom.name = "right_palm_collision"
    elif geom.meshname.endswith("_softbody_simplified"):
      geom.name = f"{body}_softbody_col"
    else:
      geom.name = f"{body}_col"
  named = sorted(geom.name for geom in spec.geoms if geom.name)
  if named != sorted(_COLLISION_GEOMS):
    raise ValueError(
      f"Wuji Hand 1 collision geoms {named} differ from {sorted(_COLLISION_GEOMS)}"
    )

  for body, pos in _TIP_SITES.items():
    finger = body.removesuffix("_link4")
    spec.body(body).add_site(name=f"{finger}_tip", pos=pos, group=4)
  for link in _PALM_EXCLUDES:
    spec.add_exclude(bodyname1="right_palm_link", bodyname2=link)
  return spec


def _get_spec(xml_path: Path) -> mujoco.MjSpec:
  if not xml_path.is_file():
    raise FileNotFoundError(
      f"Wuji Hand 1 MJCF not found: {xml_path}. Fetch the wuji-description "
      "submodule with `git submodule update --init`."
    )
  spec = mujoco.MjSpec.from_file(str(xml_path))
  spec.assets = _get_assets(xml_path, spec)
  return _apply_task_edits(spec)


def get_wuji_hand_cfg(hand_side: str = "right") -> EntityCfg:
  """Build the Wuji Hand EntityCfg for the canonical mesh-palm + softbody-thumb XML."""
  from mjlab.actuator import XmlActuatorCfg
  from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

  xml_path = wuji_hand_xml_path(hand_side)
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.5),
      rot=(1.0, 0.0, 0.0, 0.0),
      joint_pos={
        ".*_finger1_joint1": 0.34,
        ".*_finger1_joint2": -0.08,
        ".*_finger1_joint3": 0.97,
        ".*_finger1_joint4": 0.78,
        ".*_finger2_joint1": 0.93,
        ".*_finger2_joint2": 0.17,
        ".*_finger2_joint3": 0.96,
        ".*_finger2_joint4": 0.80,
        ".*_finger3_joint1": 1.02,
        ".*_finger3_joint2": -0.02,
        ".*_finger3_joint3": 0.83,
        ".*_finger3_joint4": 0.76,
        ".*_finger4_joint1": 0.98,
        ".*_finger4_joint2": -0.16,
        ".*_finger4_joint3": 0.71,
        ".*_finger4_joint4": 0.88,
        ".*_finger5_joint1": 0.83,
        ".*_finger5_joint2": -0.21,
        ".*_finger5_joint3": 0.76,
        ".*_finger5_joint4": 1.23,
      },
      joint_vel={".*": 0.0},
    ),
    spec_fn=partial(_get_spec, xml_path),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        XmlActuatorCfg(
          target_names_expr=(".*_finger.*_joint.*",),
        ),
      ),
      soft_joint_pos_limit_factor=0.9,
    ),
  )
