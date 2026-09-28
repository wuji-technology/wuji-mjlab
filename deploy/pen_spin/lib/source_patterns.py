# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Build a pen-specific codebook from the delivered source artwork only.

IDs are local indices in face_mapping.json order, then row/column order within
each face. They do not refer to any predefined ArUco dictionary. Canonical
orientation is the patch orientation in its delivered texture.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

DICTIONARY_NAME = "SOURCE_1X2_18"


def make_dictionary(bits):
    raw = np.asarray(bits)
    if raw.ndim != 3 or raw.shape[1:] != (4, 4) or not np.all((raw == 0) | (raw == 1)):
        raise ValueError("source codebook must contain binary 4x4 matrices")
    matrices = raw.astype(np.uint8)
    distances = [np.count_nonzero(a != np.rot90(b, rotation))
                 for i, a in enumerate(matrices) for j, b in enumerate(matrices)
                 for rotation in range(4) if i != j or rotation != 0]
    if not distances or min(distances) == 0:
        raise ValueError("source patterns have duplicate codes or ambiguous rotations")
    byte_list = np.concatenate([cv2.aruco.Dictionary_getByteListFromBits(b) for b in matrices])
    # Exact matching: do not silently repair or substitute source patterns.
    return cv2.aruco.Dictionary(byte_list, 4, 0), int(min(distances))


def extract(delivery: Path):
    faces = json.loads((delivery / "face_mapping.json").read_text())
    manifest = json.loads((delivery / "tag_checks.json").read_text())
    if not isinstance(manifest, dict) or manifest.get("pattern_count") != 18:
        raise ValueError("expected the 18-pattern source delivery manifest")
    patterns, hashes = [], {}
    tile_pixels = 432  # 18 mm at 24 px/mm; restore cube texture aspect ratio first
    for face in faces:
        path = delivery / "textures" / face["texture"]
        hashes[face["texture"]] = hashlib.sha256(path.read_bytes()).hexdigest()
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"unreadable texture {path}")
        width, height = face["physical_width_mm"] * 24, face["physical_height_mm"] * 24
        if width % tile_pixels or height % tile_pixels:
            raise ValueError(f"unexpected face dimensions: {face}")
        gray = cv2.resize(gray, (width, height))
        for row in range(height // tile_pixels):
            for col in range(width // tile_pixels):
                tile = gray[row*tile_pixels:(row+1)*tile_pixels,
                            col*tile_pixels:(col+1)*tile_pixels]
                contours, _ = cv2.findContours((tile < 128).astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    raise ValueError(f"no source border in {path}, slot {row}/{col}")
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                if min(w, h) < .8*tile_pixels or abs(w-h) > .03*tile_pixels:
                    raise ValueError(f"invalid source border in {path}, slot {row}/{col}")
                # Artwork: black outer ring, white ring, 4x4 black/white data.
                # The physical black data becomes logical 1 for inverted detection.
                bits = []
                for r in range(4):
                    line = []
                    for c in range(4):
                        cy, cx = int(y+(r+2.5)*h/8), int(x+(c+2.5)*w/8)
                        samples = tile[cy-3:cy+4, cx-3:cx+4] < 128
                        if .1 < samples.mean() < .9:
                            raise ValueError(f"ambiguous source cell in {path}: {r}/{c}")
                        line.append(int(samples.mean() >= .5))
                    bits.append(line)
                patterns.append({"id": len(patterns), "name": f"{path.stem}_r{row}_c{col}",
                                 "texture": path.name, "row": row, "column": col,
                                 "bits": bits})
    if len(patterns) != manifest["pattern_count"] or len(manifest["patterns"]) != len(patterns):
        raise ValueError("extracted pattern count differs from source manifest")
    dictionary, distance = make_dictionary([p["bits"] for p in patterns])
    return patterns, dictionary, distance, hashes
