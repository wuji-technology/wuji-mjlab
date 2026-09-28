# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from wuji_mjlab.assets.objects.inhand_object.object_cfg import INHAND_OBJECT_XML

_EXPECTED = {
  "fileright": "RIGHT_purple.png",
  "fileleft": "LEFT_blue.png",
  "fileup": "TOP_red.png",
  "filedown": "BOTTOM_cyan.png",
  "filefront": "FRONT_green.png",
  "fileback": "BACK_black.png",
}


def _root():
  return ET.parse(INHAND_OBJECT_XML).getroot()


def _aruco_texture():
  for tex in _root().iter("texture"):
    if tex.get("name") == "arucocube":
      return tex
  raise AssertionError("no <texture name='arucocube'> in cube.xml")


def test_arucocube_is_a_cube_texture():
  assert _aruco_texture().get("type") == "cube"


def test_arucocube_face_mapping_matches_physical_cube():
  tex = _aruco_texture()
  for attr, fname in _EXPECTED.items():
    val = tex.get(attr, "")
    assert val.endswith(fname), f"{attr} must map to {fname}, got {val!r}"


def test_arucocube_files_exist():
  tex_dir = Path(INHAND_OBJECT_XML).parent.parent / "textures"
  for fname in _EXPECTED.values():
    assert (tex_dir / fname).is_file(), f"missing texture {fname}"


def test_object_cube_visual_is_arucocube_box():
  # A type="cube" texture only maps correctly on a box; on the dex_cube mesh (2D UVs) it scrambles.
  visual = [g for g in _root().iter("geom") if g.get("group") == "2"]
  assert visual, "no group-2 visual geom found in cube.xml"
  assert len(visual) == 1, "expected exactly one group-2 visual geom"
  g = visual[0]
  assert g.get("type") == "box", f"visual geom must be a box, got {g.get('type')!r}"
  assert g.get("material") == "arucocube"
  assert g.get("mesh") is None, "visual box must not reference the dex_cube mesh"
