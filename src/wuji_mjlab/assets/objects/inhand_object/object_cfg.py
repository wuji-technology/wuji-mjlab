# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Common in-hand object config helpers."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco

if TYPE_CHECKING:
  from mjlab.entity import EntityCfg

_ASSET_DIR = Path(__file__).resolve().parent

INHAND_OBJECT_XML: Path = _ASSET_DIR / "xmls" / "cube.xml"
assert INHAND_OBJECT_XML.exists(), f"Missing in-hand object XML: {INHAND_OBJECT_XML}"

# Baseline cube — reflects values inside cube.xml. If cube.xml changes, update here.
BASELINE_EDGE_M = 0.054
BASELINE_MASS_KG = 0.120
CUBE_DENSITY = BASELINE_MASS_KG / (BASELINE_EDGE_M**3)  # ≈ 762.08 kg/m^3


def _get_spec(edge_m: float | None = None) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(INHAND_OBJECT_XML))
  if edge_m is None:
    return spec

  half = edge_m / 2.0
  mass = CUBE_DENSITY * (edge_m**3)

  cube_geom = next((g for g in spec.geoms if g.name == "cube"), None)
  if cube_geom is None:
    raise ValueError(f"{INHAND_OBJECT_XML} must contain a geom named 'cube'")
  cube_geom.size = [half, half, half]
  cube_geom.mass = mass

  # Keep the visual ArUco box in lockstep with the collision box, otherwise a
  # non-baseline cube renders (and its goal ghost, derived from this geom) at
  # the wrong 54 mm size.
  cube_visual = next((g for g in spec.geoms if g.name == "cube_visual"), None)
  if cube_visual is None:
    raise ValueError(f"{INHAND_OBJECT_XML} must contain a geom named 'cube_visual'")
  cube_visual.size = [half, half, half]

  return spec


def get_inhand_object_cfg(edge_m: float | None = None) -> EntityCfg:
  from mjlab.entity import EntityCfg

  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(-0.1404, 0.0042, 0.52),
      rot=(0.3894, -0.3796, -0.5980, -0.5888),
      lin_vel=(0.0, 0.0, 0.0),
      ang_vel=(0.0, 0.0, 0.0),
    ),
    spec_fn=partial(_get_spec, edge_m),
  )
