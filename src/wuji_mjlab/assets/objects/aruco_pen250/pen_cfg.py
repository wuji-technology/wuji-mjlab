# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""250 mm pen in the +X tracking frame."""

import math
from functools import partial
from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg

PEN_XML = Path(__file__).resolve().parent / "xmls" / "pen_tracking.xml"


def get_tracking_pen_spec(*, mass: float = 0.030) -> mujoco.MjSpec:
  """Load the tracking-frame pen and scale its inertia to ``mass``."""
  if not math.isfinite(mass) or mass <= 0:
    raise ValueError(f"Invalid pen mass: {mass}")

  spec = mujoco.MjSpec.from_file(str(PEN_XML))
  source = spec.compile().body("cube")
  body = spec.body("cube")
  body.explicitinertial = 1
  body.mass = mass
  body.ipos = source.ipos.copy()
  body.iquat = source.iquat.copy()
  body.inertia = source.inertia.copy() * (mass / float(source.mass[0]))
  return spec


def get_pen_cfg(mass: float = 0.030) -> EntityCfg:
  return EntityCfg(spec_fn=partial(get_tracking_pen_spec, mass=mass))
