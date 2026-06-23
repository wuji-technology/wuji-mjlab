# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Load the existing deploy cube ArUco layout without importing MVS camera code."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEPLOY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUBE_CONFIG_FILE = DEPLOY_ROOT / "reorient" / "config" / "cube_tags.json"


@dataclass(frozen=True)
class CubeLayout:
    """OpenCV board plus the metadata needed for fallback PnP matching."""

    config_path: Path
    cube_size: float
    tag_size: float
    tag_center_offset: float
    board: Any
    obj_points: list[np.ndarray]
    ids: np.ndarray
    tag_to_face: dict[int, str]

    @property
    def marker_ids(self) -> list[int]:
        return [int(x) for x in self.ids.flatten()]

    def match_image_points(
        self,
        corners: list[np.ndarray],
        ids: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, list[int]]:
        """Match detected 2D corners to this board's 3D object points."""
        if ids is None or len(ids) == 0:
            return None, None, []

        id_to_obj = {
            int(marker_id): self.obj_points[idx]
            for idx, marker_id in enumerate(self.ids.flatten())
        }
        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []
        used_ids: list[int] = []
        for idx, marker_id in enumerate(ids.flatten()):
            marker_id = int(marker_id)
            if marker_id not in id_to_obj:
                continue
            object_points.extend(id_to_obj[marker_id])
            image_points.extend(corners[idx].reshape(4, 2))
            used_ids.append(marker_id)

        if not object_points:
            return None, None, []
        return (
            np.asarray(object_points, dtype=np.float32),
            np.asarray(image_points, dtype=np.float32),
            used_ids,
        )


def resolve_cube_config_path(arg: str | Path | None) -> Path:
    """Resolve a --cube argument using the same rules as deploy/reorient."""
    if arg is None or str(arg) in {"", "default"}:
        return DEFAULT_CUBE_CONFIG_FILE

    path = Path(arg)
    if path.exists():
        return path.resolve()

    candidate = DEFAULT_CUBE_CONFIG_FILE.parent / f"cube_tags{arg}.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"--cube={arg!r}: not a path nor a known cube_tags suffix. Tried {candidate}."
    )


def load_cube_layout(
    arg: str | Path | None = None,
    *,
    dictionary: Any | None = None,
) -> CubeLayout:
    config_path = resolve_cube_config_path(arg)
    with config_path.open("r") as f:
        cfg = json.load(f)

    dictionary = dictionary or cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return build_cube_layout(config_path, cfg, dictionary=dictionary)


def build_cube_layout(
    config_path: str | Path,
    cfg: dict[str, Any],
    *,
    dictionary: Any,
) -> CubeLayout:
    cube_size = float(cfg["cube_size"])
    tag_size = float(cfg["tag_size"])
    tag_center_offset = float(cfg["tag_center_offset"])
    face_rotations = {
        "TOP": 0,
        "BOTTOM": 0,
        "FRONT": 0,
        "BACK": 0,
        "LEFT": 0,
        "RIGHT": 0,
    }
    face_rotations.update({str(k): int(v) for k, v in cfg.get("face_rotations", {}).items()})

    faces_config = cfg.get("faces_config") or _default_faces_config()
    tag_map = {
        str(face): {int(marker_id): str(pos) for marker_id, pos in markers.items()}
        for face, markers in faces_config.items()
    }
    tag_to_face = {
        int(marker_id): face
        for face, markers in tag_map.items()
        for marker_id in markers.keys()
    }

    faces = _load_face_axes(cfg, cube_size)
    board_corners: list[np.ndarray] = []
    board_ids: list[list[int]] = []
    for face_name, (center, u_axis, v_axis) in faces.items():
        face_tags = _build_face_tags(
            center,
            u_axis,
            v_axis,
            tag_size=tag_size,
            tag_center_offset=tag_center_offset,
            rotation=face_rotations.get(face_name, 0),
        )
        for marker_id, pos in tag_map[face_name].items():
            board_corners.append(face_tags[pos])
            board_ids.append([int(marker_id)])

    sorted_idx = np.argsort([marker_id[0] for marker_id in board_ids])
    board_corners = [board_corners[idx].astype(np.float32) for idx in sorted_idx]
    board_ids_arr = np.asarray([board_ids[idx] for idx in sorted_idx], dtype=np.int32)

    if hasattr(cv2.aruco, "Board_create"):
        board = cv2.aruco.Board_create(board_corners, dictionary, board_ids_arr)
    else:
        board = cv2.aruco.Board(board_corners, dictionary, board_ids_arr)

    return CubeLayout(
        config_path=Path(config_path),
        cube_size=cube_size,
        tag_size=tag_size,
        tag_center_offset=tag_center_offset,
        board=board,
        obj_points=board_corners,
        ids=board_ids_arr,
        tag_to_face=tag_to_face,
    )


def _load_face_axes(
    cfg: dict[str, Any],
    cube_size: float,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    half = cube_size * 0.5
    face_axes = cfg.get("face_axes")
    if face_axes:
        return {
            str(name): (
                np.asarray(axes["center"], dtype=np.float64) * half,
                np.asarray(axes["u"], dtype=np.float64),
                np.asarray(axes["v"], dtype=np.float64),
            )
            for name, axes in face_axes.items()
        }

    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    return {
        "TOP": (half * z_axis, x_axis, y_axis),
        "BOTTOM": (-half * z_axis, x_axis, -y_axis),
        "FRONT": (-half * y_axis, x_axis, z_axis),
        "BACK": (half * y_axis, -x_axis, z_axis),
        "LEFT": (-half * x_axis, -y_axis, z_axis),
        "RIGHT": (half * x_axis, y_axis, z_axis),
    }


def _build_face_tags(
    face_center: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    *,
    tag_size: float,
    tag_center_offset: float,
    rotation: int,
) -> dict[str, np.ndarray]:
    half_tag = tag_size * 0.5

    tags: dict[str, np.ndarray] = {}
    centers = {
        "T": face_center + tag_center_offset * v_axis,
        "B": face_center - tag_center_offset * v_axis,
        "L": face_center - tag_center_offset * u_axis,
        "R": face_center + tag_center_offset * u_axis,
    }
    for pos, center in centers.items():
        corners = np.array(
            [
                center - half_tag * u_axis + half_tag * v_axis,
                center + half_tag * u_axis + half_tag * v_axis,
                center + half_tag * u_axis - half_tag * v_axis,
                center - half_tag * u_axis - half_tag * v_axis,
            ],
            dtype=np.float32,
        )
        tags[pos] = _rotate_corners(corners, rotation)
    return tags


def _rotate_corners(corners: np.ndarray, rotation: int) -> np.ndarray:
    turns = (int(rotation) // 90) % 4
    if turns == 0:
        return corners
    if turns == 1:
        return np.array([corners[3], corners[0], corners[1], corners[2]], dtype=np.float32)
    if turns == 2:
        return np.array([corners[2], corners[3], corners[0], corners[1]], dtype=np.float32)
    return np.array([corners[1], corners[2], corners[3], corners[0]], dtype=np.float32)


def _default_faces_config() -> dict[str, dict[int, str]]:
    return {
        "TOP": {0: "L", 1: "B", 2: "T", 3: "R"},
        "BOTTOM": {8: "R", 9: "T", 10: "B", 11: "L"},
        "FRONT": {16: "R", 17: "T", 18: "B", 19: "L"},
        "BACK": {20: "B", 21: "R", 22: "L", 23: "T"},
        "LEFT": {4: "R", 5: "T", 6: "B", 7: "L"},
        "RIGHT": {12: "B", 13: "R", 14: "L", 15: "T"},
    }
