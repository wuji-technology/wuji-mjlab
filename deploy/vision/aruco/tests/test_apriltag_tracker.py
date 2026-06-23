# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import numpy as np
import pytest
from deploy.vision.aruco.apriltag_tracker import (
    DEFAULT_APRILTAG_FAMILY,
    DEFAULT_APRILTAG_SIZE_M,
    AprilTagTracker,
)


def test_apriltag_detector_uses_wuji_parameters() -> None:
    tracker = AprilTagTracker(detector_cls=FakeDetector, enable_clahe=False)

    assert tracker.detector.kwargs == {
        "families": "tag36h11",
        "nthreads": 4,
        "quad_decimate": 1.0,
        "quad_sigma": 0.0,
        "decode_sharpening": 0.25,
    }
    assert tracker.family == DEFAULT_APRILTAG_FAMILY
    assert tracker.tag_size_m == pytest.approx(DEFAULT_APRILTAG_SIZE_M)


def test_apriltag_pose_builds_camera_transform() -> None:
    tracker = AprilTagTracker(detector_cls=FakeDetector, enable_clahe=False, tag_size_m=0.048)
    tracker.detector.results = [
        FakeAprilTagResult(
            tag_id=0,
            pose_R=np.eye(3),
            pose_t=np.array([[0.1], [0.2], [0.3]]),
        )
    ]
    camera_matrix = np.array(
        [
            [800.0, 0.0, 320.0],
            [0.0, 810.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    poses = tracker.detect(np.full((20, 20, 3), 255, dtype=np.uint8), camera_matrix)

    assert len(poses) == 1
    pose = poses[0]
    assert pose.tag_id == 0
    assert np.allclose(pose.R, np.eye(3))
    assert np.allclose(pose.tvec_m, [0.1, 0.2, 0.3])
    assert np.allclose(pose.quat_xyzw, [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(pose.T_camera_tag[:3, 3], [0.1, 0.2, 0.3])
    assert np.allclose(pose.T_camera_tag[:3, :3], np.eye(3))

    call = tracker.detector.detect_calls[0]
    assert call["estimate_tag_pose"] is True
    assert call["camera_params"] == (800.0, 810.0, 320.0, 240.0)
    assert call["tag_size"] == pytest.approx(0.048)


def test_apriltag_target_id_filter() -> None:
    tracker = AprilTagTracker(detector_cls=FakeDetector, enable_clahe=False, target_id=0)
    tracker.detector.results = [
        FakeAprilTagResult(tag_id=0, pose_R=np.eye(3), pose_t=np.array([[0.0], [0.0], [0.5]])),
        FakeAprilTagResult(tag_id=1, pose_R=np.eye(3), pose_t=np.array([[0.0], [0.0], [0.7]])),
    ]

    poses = tracker.detect(np.full((20, 20), 255, dtype=np.uint8), _camera_matrix())

    assert [pose.tag_id for pose in poses] == [0]


class FakeDetector:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.results = []
        self.detect_calls = []

    def detect(self, _gray, **kwargs):
        self.detect_calls.append(kwargs)
        return self.results


class FakeAprilTagResult:
    def __init__(self, *, tag_id: int, pose_R: np.ndarray, pose_t: np.ndarray) -> None:
        self.tag_id = tag_id
        self.pose_R = pose_R
        self.pose_t = pose_t
        self.corners = np.array(
            [
                [10.0, 10.0],
                [18.0, 10.0],
                [18.0, 18.0],
                [10.0, 18.0],
            ],
            dtype=np.float32,
        )
        self.center = np.array([14.0, 14.0], dtype=np.float32)


def _camera_matrix() -> np.ndarray:
    return np.array(
        [
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
