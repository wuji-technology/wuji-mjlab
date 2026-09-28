# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Measure ArUco corners from the delivered 1x2 MJCF's rendered faces.

Uses the actual textures and MuJoCo cube-map texture mapping, including the
flipped top head. All 18 source patterns define a custom codebook. Run with
MUJOCO_GL=egl on a headless machine. Units in the output are metres.

    python tools/gen_pen_tags_1x2.py --delivery <source delivery dir> \
        --output config/pen_tags.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np
from lib.source_patterns import DICTIONARY_NAME, extract


def generate(delivery: Path, shaft_length_mm: float | None = None) -> dict:
  delivery = delivery.resolve()
  patterns, dictionary, code_distance, texture_hashes = extract(delivery)
  root = ET.parse(delivery / "pen.xml").getroot()
  root.find("compiler").set("texturedir", str(delivery / "textures"))
  body = root.find("worldbody/body[@name='pen']")
  shaft = body.find("geom[@name='shaft']")
  ends = np.fromstring(shaft.get("fromto"), sep=" ").reshape(2, 3)
  half_shaft = (ends[1, 2] - ends[0, 2]) / 2
  if shaft_length_mm is not None:
    if not np.isfinite(shaft_length_mm) or shaft_length_mm <= 0:
      raise ValueError("shaft length must be finite and positive")
    half_shaft = shaft_length_mm / 2000
    shaft.set("fromto", f"0 0 {-half_shaft} 0 0 {half_shaft}")
  boxes = {}
  for name, sign in (("A_bottom", -1), ("A_top", 1)):
    geom = body.find(f"geom[@name='{name}']")
    size = np.fromstring(geom.get("size"), sep=" ")
    center = np.fromstring(geom.get("pos"), sep=" ")
    if shaft_length_mm is not None:
      center[2] = sign * (half_shaft + size[2])
      geom.set("pos", " ".join(map(str, center)))
    boxes[name] = (center, size)

  resolution = 1600
  root.find("visual/global").set("offwidth", str(resolution))
  root.find("visual/global").set("offheight", str(resolution))
  faces = json.loads((delivery / "face_mapping.json").read_text())
  views = []
  distance, fovy = 0.07, 40.0
  for index, face in enumerate(faces):
    box = "A_" + face["head"]
    center, size = boxes[box]
    normal = np.array(face["normal"], dtype=float)
    up = np.array([0.0, 0.0, 1.0]) if normal[2] == 0 else np.array([0.0, 1.0, 0.0])
    right = np.cross(up, normal)
    surface = center + normal * size
    position = surface + normal * distance
    ET.SubElement(
      root.find("worldbody"),
      "camera",
      {
        "name": f"measure_{index}",
        "pos": " ".join(map(str, position)),
        "xyaxes": " ".join(map(str, np.r_[right, up])),
        "fovy": str(fovy),
      },
    )
    views.append((box, face, surface, right, up, normal))

  model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  params = cv2.aruco.DetectorParameters()
  params.detectInvertedMarker = True
  params.errorCorrectionRate = 0.0
  params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
  detector = cv2.aruco.ArucoDetector(dictionary, params)
  metres_per_pixel = 2 * distance * np.tan(np.radians(fovy / 2)) / resolution
  tags = {}
  with mujoco.Renderer(model, resolution, resolution) as renderer:
    for index, (box, face, surface, right, up, normal) in enumerate(views):
      renderer.update_scene(data, camera=f"measure_{index}")
      gray = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2GRAY)
      corners, ids, _ = detector.detectMarkers(gray)
      if ids is None:
        raise ValueError(f"no tags detected on {face['texture']}")
      expected_face = {p["id"] for p in patterns if p["texture"] == face["texture"]}
      if set(ids.flatten()) != expected_face:
        raise ValueError(f"source patterns missing or misplaced on {face['texture']}")
      for quad, tid in zip(corners, ids.flatten(), strict=True):
        key = str(int(tid))
        if key in tags:
          raise ValueError(f"duplicate marker ID {tid}")
        xy = (quad[0] - (resolution - 1) / 2) * metres_per_pixel
        points = surface + xy[:, 0, None] * right - xy[:, 1, None] * up
        inward = np.cross(points[1] - points[0], points[2] - points[1])
        if np.dot(inward, normal) >= 0:
          raise ValueError(f"tag {tid}: incorrect corner winding")
        side = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).mean()
        axis = int(np.flatnonzero(normal)[0])
        tags[key] = {
          "pattern_name": patterns[int(tid)]["name"],
          "box": box,
          "face_normal": ("+" if normal[axis] > 0 else "-") + "XYZ"[axis],
          "texture": face["texture"],
          "center_m": points.mean(axis=0).tolist(),
          "size_m": float(side),
          "corners_m": points.tolist(),
        }
  expected = {str(p["id"]) for p in patterns}
  if set(tags) != expected:
    raise ValueError(
      f"rendered IDs {sorted(tags)} differ from delivery IDs {sorted(expected)}"
    )
  return {
    "description": "1x2 delivery: actual MJCF face renders, OpenCV canonical corner order; origin at shaft midpoint, +Z toward A_top. Units: metres.",
    "source_delivery": delivery.name,
    "dictionary": DICTIONARY_NAME,
    "shaft_length_override_mm": shaft_length_mm,
    "source_xml_sha256": hashlib.sha256(
      (delivery / "pen.xml").read_bytes()
    ).hexdigest(),
    "source_texture_sha256": texture_hashes,
    "custom_dictionary": {
      "marker_size": 4,
      "patterns": patterns,
      "minimum_rotated_hamming_distance": code_distance,
    },
    "inverted_markers": True,
    "error_correction_rate": 0.0,
    "pen_total_length_m": float(
      boxes["A_top"][0][2]
      + boxes["A_top"][1][2]
      - boxes["A_bottom"][0][2]
      + boxes["A_bottom"][1][2]
    ),
    "shaft_half_length_m": float(half_shaft),
    "shaft_radius_m": float(shaft.get("size")),
    "box_centers_m": {k: c.tolist() for k, (c, _) in boxes.items()},
    "box_half_sizes_m": {k: s.tolist() for k, (_, s) in boxes.items()},
    "tag_size_m": float(np.median([t["size_m"] for t in tags.values()])),
    "corner_measurement_pixel_m": float(metres_per_pixel),
    "tags": dict(sorted(tags.items(), key=lambda item: int(item[0]))),
  }


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--delivery", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--shaft-length-mm",
    type=float,
    help="exposed shaft length; total length is this + 72 mm",
  )
  args = parser.parse_args()
  layout = generate(args.delivery, args.shaft_length_mm)
  args.output.write_text(json.dumps(layout, indent=2) + "\n")
  print(
    f"Wrote {len(layout['tags'])} tags, total length "
    f"{layout['pen_total_length_m'] * 1000:.1f} mm to {args.output}"
  )
