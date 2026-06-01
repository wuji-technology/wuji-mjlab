# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Common Wuji Hand robot config helpers."""

from functools import partial
from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

_ASSET_DIR = Path(__file__).resolve().parent


def update_assets(
  assets: dict,
  path: Path,
  meshdir: str | None = None,
  glob: str = "*",
  recursive: bool = False,
) -> None:
  """Read files from a directory and stuff their bytes into ``assets``.

  Inlined here to keep wuji-side asset loading stable across mjlab versions.
  """
  for f in Path(path).glob(glob):
    if f.is_file():
      asset_key = f"{meshdir}/{f.name}" if meshdir else f.name
      assets[asset_key] = f.read_bytes()
    elif f.is_dir() and recursive:
      update_assets(assets, f, meshdir, glob, recursive)


def _resolve_wuji_hand_xml(hand_side: str) -> Path:
  if hand_side not in ("right", "left"):
    raise ValueError(
      f"Unsupported hand_side '{hand_side}'. Expected 'right' or 'left'."
    )
  return _ASSET_DIR / "mjcf" / f"{hand_side}_mjlab.xml"


def _get_assets(xml_path: Path, meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  mesh_root = (xml_path.parent / meshdir).resolve()
  update_assets(assets, mesh_root, meshdir, glob="*.STL")
  update_assets(assets, mesh_root, meshdir, glob="*.stl")
  return assets


def _get_spec(xml_path: Path) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(xml_path))
  spec.assets = _get_assets(xml_path, spec.meshdir)
  return spec


WUJI_HAND_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlActuatorCfg(
      target_names_expr=(".*_finger.*_joint.*",),
    ),
  ),
  soft_joint_pos_limit_factor=0.9,
)


WUJI_HAND_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
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
)


def get_wuji_hand_cfg(hand_side: str = "right") -> EntityCfg:
  """Build the Wuji Hand EntityCfg for the canonical mesh-palm + softbody-thumb XML."""
  xml_path = _resolve_wuji_hand_xml(hand_side)
  return EntityCfg(
    init_state=WUJI_HAND_HOME_KEYFRAME,
    spec_fn=partial(_get_spec, xml_path),
    articulation=WUJI_HAND_ARTICULATION,
  )
