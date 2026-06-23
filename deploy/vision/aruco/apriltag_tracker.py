# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji-compatible AprilTag pose tracking for Orbbec visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_APRILTAG_FAMILY = "tag36h11"
DEFAULT_APRILTAG_SIZE_M = 0.048


@dataclass(frozen=True)
class AprilTagPose:
    tag_id: int
    corners: np.ndarray
    center: np.ndarray
    R: np.ndarray
    tvec_m: np.ndarray
    quat_xyzw: np.ndarray
    T_camera_tag: np.ndarray


class AprilTagTracker:
    """AprilTag detector with the same parameters as Wuji's observer."""

    def __init__(
        self,
        *,
        family: str = DEFAULT_APRILTAG_FAMILY,
        tag_size_m: float = DEFAULT_APRILTAG_SIZE_M,
        target_id: int | None = None,
        detector_cls: Any | None = None,
        enable_clahe: bool = True,
        clahe_clip: float = 2.0,
        clahe_tile: tuple[int, int] = (8, 8),
    ) -> None:
        self.family = family
        self.tag_size_m = float(tag_size_m)
        self.target_id = target_id
        self.detector_kwargs = {
            "families": family,
            "nthreads": 4,
            "quad_decimate": 1.0,
            "quad_sigma": 0.0,
            "decode_sharpening": 0.25,
        }
        if detector_cls is None:
            try:
                from pupil_apriltags import Detector as detector_cls
            except ImportError as exc:
                raise RuntimeError(
                    "AprilTag visualization requires pupil-apriltags. "
                    "Install/use the deploy environment or run without --enable-apriltag."
                ) from exc
        self.detector = detector_cls(**self.detector_kwargs)
        self._clahe = (
            cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=clahe_tile)
            if enable_clahe
            else None
        )

    def detect(
        self,
        image_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray | None = None,
    ) -> list[AprilTagPose]:
        """Detect tags and estimate pose in the camera coordinate frame."""
        del dist_coeffs  # pupil_apriltags follows Wuji and uses pinhole camera_params only.
        gray = self.preprocess_image(image_bgr)
        camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        results = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(
                float(camera_matrix[0, 0]),
                float(camera_matrix[1, 1]),
                float(camera_matrix[0, 2]),
                float(camera_matrix[1, 2]),
            ),
            tag_size=self.tag_size_m,
        )
        poses = [_make_pose(result) for result in results]
        if self.target_id is not None:
            poses = [pose for pose in poses if pose.tag_id == self.target_id]
        return poses

    def preprocess_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """Use the same min-channel grayscale style as Wuji's vision loop."""
        if image_bgr.ndim == 2:
            gray = image_bgr
        elif image_bgr.ndim == 3 and image_bgr.shape[2] >= 3:
            gray = np.minimum(
                np.minimum(image_bgr[:, :, 0], image_bgr[:, :, 1]),
                image_bgr[:, :, 2],
            )
        else:
            raise ValueError(f"unsupported image shape for AprilTag detection: {image_bgr.shape}")

        gray = np.ascontiguousarray(gray.astype(np.uint8, copy=False))
        if self._clahe is not None:
            return self._clahe.apply(gray)
        return gray


def _make_pose(result: Any) -> AprilTagPose:
    R = np.asarray(result.pose_R, dtype=np.float64).reshape(3, 3)
    tvec = np.asarray(result.pose_t, dtype=np.float64).reshape(3)
    return AprilTagPose(
        tag_id=int(result.tag_id),
        corners=np.asarray(result.corners, dtype=np.float32).reshape(4, 2),
        center=np.asarray(result.center, dtype=np.float32).reshape(2),
        R=R,
        tvec_m=tvec,
        quat_xyzw=Rotation.from_matrix(R).as_quat(),
        T_camera_tag=_make_transform(R, tvec),
    )


def _make_transform(R: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform
