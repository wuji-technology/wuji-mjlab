# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import warnings

import cv2
import numpy as np
import pytest
from deploy.vision.aruco.camera_adapter import OrbbecCameraAdapter
from deploy.vision.aruco.cube_layout import load_cube_layout
from deploy.vision.aruco.pose_estimator import ArucoPoseEstimator
from deploy.vision.aruco.wuji_cube_tracker import WujiCubeTracker, WujiTrackerConfig


def test_cube_layout_loads_existing_default_config() -> None:
    layout = load_cube_layout("default")

    assert layout.config_path.name == "cube_tags.json"
    assert layout.cube_size == pytest.approx(0.054)
    assert layout.tag_size == pytest.approx(0.013)
    assert len(layout.marker_ids) == 24
    assert len(layout.obj_points) == 24
    assert all(points.shape == (4, 3) for points in layout.obj_points)
    assert layout.board is not None


def test_synthetic_marker_pose_has_positive_z() -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 7, 160)
    image = np.full((480, 640), 255, dtype=np.uint8)
    image[160:320, 240:400] = marker
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    camera_matrix = np.array(
        [
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    estimator = ArucoPoseEstimator(
        marker_length_m=0.04,
        algorithm="opencv",
        ema_alpha_t=1.0,
        ema_alpha_r=1.0,
    )

    detections = estimator.detect(image_bgr, camera_matrix, np.zeros(5))

    assert detections.ids is not None
    assert 7 in detections.ids.flatten()
    assert len(detections.marker_poses) == 1
    assert detections.marker_poses[0].marker_id == 7
    assert detections.marker_poses[0].tvec_m[2] > 0.0


def test_default_intrinsics_warn_once_and_use_zero_distortion() -> None:
    adapter = OrbbecCameraAdapter(
        backend="v4l2",
        width=640,
        height=480,
        open_device=False,
    )

    with pytest.warns(RuntimeWarning, match="No camera calibration found"):
        info = adapter.get_camera_info()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        info_again = adapter.get_camera_info()

    assert not caught
    assert info.calibrated is False
    assert info.source == "fallback"
    assert info.width == 640
    assert info.height == 480
    assert info.camera_matrix[0, 0] == pytest.approx(640.0)
    assert info.camera_matrix[1, 1] == pytest.approx(640.0)
    assert np.allclose(info.dist_coeffs, np.zeros(5))
    assert np.allclose(info_again.camera_matrix, info.camera_matrix)


def test_wuji_algorithm_requires_cube_layout() -> None:
    with pytest.raises(ValueError, match="requires --cube not none"):
        ArucoPoseEstimator(algorithm="wuji")


def test_wuji_tracker_ippe_iterative_outputs_positive_z() -> None:
    layout = load_cube_layout("default")
    camera_matrix = _camera_matrix()
    corners, ids = _project_markers(layout, [0, 1], camera_matrix, tvec=np.array([0.01, -0.02, 0.45]))
    tracker = WujiCubeTracker(layout, WujiTrackerConfig(reproj_threshold=6.0))

    result = tracker.detect_cube_pose(corners, ids, camera_matrix, np.zeros(5))

    assert result is not None
    assert result.tvec_m[2] > 0.0
    assert result.dominant_face == "TOP"
    assert result.reproj_error_px < 1.0
    assert result.n_tags == 2


def test_wuji_tracker_reprojection_gate_rejects_bad_pose() -> None:
    layout = load_cube_layout("default")
    camera_matrix = _camera_matrix()
    corners, ids = _project_markers(layout, [0, 1], camera_matrix, tvec=np.array([0.01, -0.02, 0.45]))
    corners[0] = corners[0].copy()
    corners[0][0, 0] += np.array([25.0, 20.0], dtype=np.float32)
    tracker = WujiCubeTracker(layout, WujiTrackerConfig(reproj_threshold=0.01))

    result = tracker.detect_cube_pose(corners, ids, camera_matrix, np.zeros(5))

    assert result is None
    assert tracker.reproj_error_px > 0.01


def test_wuji_tracker_lost_frame_reacquire_resets_position_filter() -> None:
    layout = load_cube_layout("default")
    camera_matrix = _camera_matrix()
    tracker = WujiCubeTracker(layout, WujiTrackerConfig(position_alpha=0.1, reproj_threshold=6.0))

    corners, ids = _project_markers(layout, [0, 1], camera_matrix, tvec=np.array([0.0, 0.0, 0.45]))
    first = tracker.detect_cube_pose(corners, ids, camera_matrix, np.zeros(5))
    assert first is not None

    tracker.detect_cube_pose([], None, camera_matrix, np.zeros(5))
    corners, ids = _project_markers(layout, [0, 1], camera_matrix, tvec=np.array([0.0, 0.0, 0.80]))
    second = tracker.detect_cube_pose(corners, ids, camera_matrix, np.zeros(5))

    assert second is not None
    assert second.tvec_m[2] > 0.70


def test_wuji_tracker_dominant_face_hysteresis_keeps_previous_face_on_tie() -> None:
    layout = load_cube_layout("default")
    camera_matrix = _camera_matrix()
    tracker = WujiCubeTracker(layout, WujiTrackerConfig(reproj_threshold=6.0))

    corners, ids = _project_markers(layout, [0], camera_matrix, tvec=np.array([0.0, 0.0, 0.45]))
    first = tracker.detect_cube_pose(corners, ids, camera_matrix, np.zeros(5))
    assert first is not None
    assert first.dominant_face == "TOP"

    corners, ids = _project_markers(layout, [20, 0], camera_matrix, tvec=np.array([0.0, 0.0, 0.45]))
    second = tracker.detect_cube_pose(corners, ids, camera_matrix, np.zeros(5))

    assert second is not None
    assert second.dominant_face == "TOP"


def _camera_matrix() -> np.ndarray:
    return np.array(
        [
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _project_markers(
    layout,
    marker_ids: list[int],
    camera_matrix: np.ndarray,
    *,
    tvec: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    id_to_obj = {
        int(marker_id): layout.obj_points[idx]
        for idx, marker_id in enumerate(layout.ids.flatten())
    }
    rvec = np.array([0.2, -0.1, 0.05], dtype=np.float64)
    corners = []
    ids = []
    for marker_id in marker_ids:
        image_points, _ = cv2.projectPoints(
            id_to_obj[int(marker_id)],
            rvec,
            tvec.reshape(3, 1),
            camera_matrix,
            np.zeros(5),
        )
        corners.append(image_points.reshape(1, 4, 2).astype(np.float32))
        ids.append([int(marker_id)])
    return corners, np.asarray(ids, dtype=np.int32)
