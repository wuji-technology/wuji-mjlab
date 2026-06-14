# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Common Revo3 right-hand robot config helpers."""

from functools import partial
from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

_ASSET_DIR = Path(__file__).resolve().parent


def _resolve_revo3_hand_xml(hand_side: str) -> Path:
  if hand_side != "right":
    raise ValueError(f"Unsupported Revo3 hand_side '{hand_side}'. Expected 'right'.")
  return _ASSET_DIR / "mjcf" / "revo3_right_mjlab.xml"


def _get_assets(xml_path: Path) -> dict[str, bytes]:
  spec = mujoco.MjSpec.from_file(str(xml_path))
  assets: dict[str, bytes] = {}
  meshdir = Path(spec.meshdir) if spec.meshdir else Path()
  for mesh in spec.meshes:
    mesh_file = getattr(mesh, "file", "")
    if not mesh_file:
      continue
    asset_key = str(meshdir / mesh_file) if spec.meshdir else mesh_file
    mesh_path = (xml_path.parent / asset_key).resolve()
    assets[asset_key] = mesh_path.read_bytes()
  return assets


def _get_spec(xml_path: Path) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(xml_path))
  spec.assets = _get_assets(xml_path)
  return spec


REVO3_HAND_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlActuatorCfg(
      target_names_expr=("right_.*_joint",),
    ),
  ),
  soft_joint_pos_limit_factor=0.9,
)


REVO3_RIGHT_HAND_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.5),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    "right_thumb_CMP_joint": 0.8,
    "right_thumb_CMR_joint": 0.4,
    "right_thumb_MCP_joint": 0.285,
    "right_thumb_PIP_joint": 0.582,
    "right_thumb_DIP_joint": 0.582,
    "right_index_MPR_joint": 0.163,
    "right_index_MCP_joint": 0.441,
    "right_index_PIP_joint": 0.822,
    "right_index_DIP_joint": 0.494,
    "right_middle_MPR_joint": -0.0832,
    "right_middle_MCP_joint": 0.448,
    "right_middle_PIP_joint": 0.883,
    "right_middle_DIP_joint": 0.305,
    "right_ring_MPR_joint": -0.2618,
    "right_ring_MCP_joint": 0.594,
    "right_ring_PIP_joint": 1.13,
    "right_ring_DIP_joint": 0.18,
    "right_little_MPR_joint": -0.228,
    "right_little_MCP_joint": 1.2,
    "right_little_PIP_joint": 0.794,
    "right_little_DIP_joint": 0.407,
  },
  joint_vel={".*": 0.0},
)


REVO3_RIGHT_JOINT_NAMES: tuple[str, ...] = (
  "right_thumb_CMP_joint",
  "right_thumb_CMR_joint",
  "right_thumb_MCP_joint",
  "right_thumb_PIP_joint",
  "right_thumb_DIP_joint",
  "right_index_MPR_joint",
  "right_index_MCP_joint",
  "right_index_PIP_joint",
  "right_index_DIP_joint",
  "right_middle_MPR_joint",
  "right_middle_MCP_joint",
  "right_middle_PIP_joint",
  "right_middle_DIP_joint",
  "right_ring_MPR_joint",
  "right_ring_MCP_joint",
  "right_ring_PIP_joint",
  "right_ring_DIP_joint",
  "right_little_MPR_joint",
  "right_little_MCP_joint",
  "right_little_PIP_joint",
  "right_little_DIP_joint",
)


def get_revo3_hand_cfg(hand_side: str = "right") -> EntityCfg:
  """Build the Revo3 right-hand EntityCfg."""
  xml_path = _resolve_revo3_hand_xml(hand_side)
  return EntityCfg(
    init_state=REVO3_RIGHT_HAND_HOME_KEYFRAME,
    spec_fn=partial(_get_spec, xml_path),
    articulation=REVO3_HAND_ARTICULATION,
  )
