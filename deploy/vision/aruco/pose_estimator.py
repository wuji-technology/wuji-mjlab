# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""ArUco detection, single-marker pose, and cube-board pose estimation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Hashable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from deploy.vision.aruco.cube_layout import CubeLayout
from deploy.vision.aruco.wuji_cube_tracker import (
    WujiCubeTracker,
    load_wuji_tracker_config,
)


@dataclass(frozen=True)
class MarkerPose:
    marker_id: int
    rvec: np.ndarray
    tvec_m: np.ndarray
    R: np.ndarray
    quat_xyzw: np.ndarray
    T_camera_marker: np.ndarray


@dataclass(frozen=True)
class BoardPose:
    ids_used: list[int]
    rvec: np.ndarray
    tvec_m: np.ndarray
    R: np.ndarray
    quat_xyzw: np.ndarray
    T_camera_cube: np.ndarray
    algorithm: str = "opencv"
    dominant_face: str | None = None
    reproj_error_px: float | None = None
    n_tags: int = 0


@dataclass(frozen=True)
class ArucoDetections:
    corners: list[np.ndarray]
    ids: np.ndarray | None
    rejected: list[np.ndarray]
    marker_poses: list[MarkerPose]
    board_pose: BoardPose | None


class PoseEMAFilter:
    """Translation EMA plus sign-continuous quaternion EMA for orientation."""

    def __init__(self, alpha_t: float = 0.6, alpha_r: float = 0.6) -> None:
        if not 0.0 < alpha_t <= 1.0:
            raise ValueError("alpha_t must be in (0, 1]")
        if not 0.0 < alpha_r <= 1.0:
            raise ValueError("alpha_r must be in (0, 1]")
        self.alpha_t = float(alpha_t)
        self.alpha_r = float(alpha_r)
        self._state: dict[Hashable, tuple[np.ndarray, np.ndarray]] = {}

    def update(self, key: Hashable, R: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        quat = Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()

        prev = self._state.get(key)
        if prev is None:
            self._state[key] = (tvec.copy(), quat.copy())
            return np.asarray(R, dtype=np.float64), tvec

        prev_t, prev_q = prev
        if float(np.dot(quat, prev_q)) < 0.0:
            quat = -quat
        filt_t = self.alpha_t * tvec + (1.0 - self.alpha_t) * prev_t
        filt_q = self.alpha_r * quat + (1.0 - self.alpha_r) * prev_q
        filt_q /= np.linalg.norm(filt_q)
        self._state[key] = (filt_t.copy(), filt_q.copy())
        return Rotation.from_quat(filt_q).as_matrix(), filt_t

    def reset(self, key: Hashable | None = None) -> None:
        if key is None:
            self._state.clear()
        else:
            self._state.pop(key, None)


class ArucoPoseEstimator:
    """Detect ArUco markers and estimate poses in the camera coordinate frame."""

    def __init__(
        self,
        *,
        marker_length_m: float = 0.013,
        cube_layout: CubeLayout | None = None,
        dictionary_id: int = cv2.aruco.DICT_4X4_50,
        algorithm: str = "wuji",
        ema_alpha_t: float = 0.6,
        ema_alpha_r: float = 0.6,
        observer_config: str | Path | None = None,
        process_noise: float | None = None,
        measurement_noise: float | None = None,
        position_alpha: float | None = None,
        reproj_threshold: float | None = None,
    ) -> None:
        if algorithm not in {"wuji", "opencv"}:
            raise ValueError(f"unsupported ArUco pose algorithm: {algorithm!r}")
        if algorithm == "wuji" and cube_layout is None:
            raise ValueError("Wuji cube algorithm requires --cube not none")

        self.algorithm = algorithm
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.marker_length_m = float(cube_layout.tag_size if cube_layout else marker_length_m)
        self.cube_layout = cube_layout
        self.pose_filter = PoseEMAFilter(alpha_t=ema_alpha_t, alpha_r=ema_alpha_r)
        self.wuji_tracker = None
        if algorithm == "wuji":
            self.wuji_tracker = WujiCubeTracker(
                cube_layout,
                load_wuji_tracker_config(
                    observer_config,
                    process_noise=process_noise,
                    measurement_noise=measurement_noise,
                    position_alpha=position_alpha,
                    reproj_threshold=reproj_threshold,
                ),
            )

        if hasattr(cv2.aruco, "DetectorParameters_create"):
            self.detector_params = cv2.aruco.DetectorParameters_create()
        else:
            self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = (
            cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )

    def detect(
        self,
        image_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> ArucoDetections:
        gray = self.wuji_tracker.preprocess_image(image_bgr) if self.wuji_tracker else _as_gray(image_bgr)
        camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)

        if self.detector is None:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray,
                self.dictionary,
                parameters=self.detector_params,
            )
        else:
            corners, ids, rejected = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            if self.wuji_tracker is not None:
                self.wuji_tracker.detect_cube_pose([], None, camera_matrix, dist_coeffs)
            return ArucoDetections(
                corners=[],
                ids=None,
                rejected=list(rejected),
                marker_poses=[],
                board_pose=None,
            )

        if self.wuji_tracker is not None:
            corners, ids = self.wuji_tracker.filter_detections(list(corners), np.asarray(ids, dtype=np.int32))
            if ids is None or len(ids) == 0:
                self.wuji_tracker.detect_cube_pose([], None, camera_matrix, dist_coeffs)
                return ArucoDetections(
                    corners=[],
                    ids=None,
                    rejected=list(rejected),
                    marker_poses=[],
                    board_pose=None,
                )

        marker_poses = self._estimate_marker_poses(corners, ids, camera_matrix, dist_coeffs)
        board_pose = self._estimate_board_pose(corners, ids, camera_matrix, dist_coeffs)
        return ArucoDetections(
            corners=list(corners),
            ids=np.asarray(ids, dtype=np.int32),
            rejected=list(rejected),
            marker_poses=marker_poses,
            board_pose=board_pose,
        )

    def reset_filters(self) -> None:
        self.pose_filter.reset()
        if self.wuji_tracker is not None:
            self.wuji_tracker.reset()

    def _estimate_marker_poses(
        self,
        corners: list[np.ndarray],
        ids: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> list[MarkerPose]:
        rvecs, tvecs, _obj_points = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            self.marker_length_m,
            camera_matrix,
            dist_coeffs,
        )
        poses: list[MarkerPose] = []
        for idx, marker_id in enumerate(ids.flatten()):
            rvec = np.asarray(rvecs[idx], dtype=np.float64).reshape(3)
            tvec = np.asarray(tvecs[idx], dtype=np.float64).reshape(3)
            R, _ = cv2.Rodrigues(rvec)
            R, tvec = self.pose_filter.update(("marker", int(marker_id)), R, tvec)
            rvec, _ = cv2.Rodrigues(R)
            poses.append(_make_marker_pose(int(marker_id), rvec.reshape(3), tvec, R))
        return poses

    def _estimate_board_pose(
        self,
        corners: list[np.ndarray],
        ids: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> BoardPose | None:
        if self.cube_layout is None:
            return None

        ids_used = [
            int(marker_id)
            for marker_id in ids.flatten()
            if int(marker_id) in self.cube_layout.tag_to_face
        ]
        if not ids_used:
            return None

        if self.wuji_tracker is not None:
            return self._estimate_wuji_board_pose(corners, ids, camera_matrix, dist_coeffs)

        board_result = self._estimate_pose_board(corners, ids, camera_matrix, dist_coeffs)
        if board_result is None:
            board_result = self._solve_pnp_board(corners, ids, camera_matrix, dist_coeffs)
        if board_result is None:
            return None

        rvec, tvec = board_result
        R, _ = cv2.Rodrigues(rvec)
        R, tvec = self.pose_filter.update("cube_board", R, tvec)
        rvec, _ = cv2.Rodrigues(R)
        return _make_board_pose(
            ids_used,
            rvec.reshape(3),
            tvec,
            R,
            algorithm="opencv",
            n_tags=len(ids_used),
        )

    def _estimate_wuji_board_pose(
        self,
        corners: list[np.ndarray],
        ids: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> BoardPose | None:
        result = self.wuji_tracker.detect_cube_pose(corners, ids, camera_matrix, dist_coeffs)
        if result is None:
            return None
        return _make_board_pose(
            result.ids_used,
            result.rvec,
            result.tvec_m,
            result.R,
            algorithm="wuji",
            dominant_face=result.dominant_face,
            reproj_error_px=result.reproj_error_px,
            n_tags=result.n_tags,
        )

    def _estimate_pose_board(
        self,
        corners: list[np.ndarray],
        ids: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not hasattr(cv2.aruco, "estimatePoseBoard"):
            return None
        try:
            retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
                corners,
                ids,
                self.cube_layout.board,
                camera_matrix,
                dist_coeffs,
                None,
                None,
            )
        except cv2.error:
            return None
        if int(retval) <= 0:
            return None
        return np.asarray(rvec, dtype=np.float64).reshape(3), np.asarray(tvec, dtype=np.float64).reshape(3)

    def _solve_pnp_board(
        self,
        corners: list[np.ndarray],
        ids: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        obj_pts, img_pts, _used_ids = self.cube_layout.match_image_points(corners, ids)
        if obj_pts is None or img_pts is None or len(obj_pts) < 4:
            return None
        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None
        return np.asarray(rvec, dtype=np.float64).reshape(3), np.asarray(tvec, dtype=np.float64).reshape(3)


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"unsupported image shape for ArUco detection: {image.shape}")


def _make_marker_pose(marker_id: int, rvec: np.ndarray, tvec: np.ndarray, R: np.ndarray) -> MarkerPose:
    return MarkerPose(
        marker_id=int(marker_id),
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3),
        tvec_m=np.asarray(tvec, dtype=np.float64).reshape(3),
        R=np.asarray(R, dtype=np.float64).reshape(3, 3),
        quat_xyzw=Rotation.from_matrix(R).as_quat(),
        T_camera_marker=_make_transform(R, tvec),
    )


def _make_board_pose(
    ids_used: list[int],
    rvec: np.ndarray,
    tvec: np.ndarray,
    R: np.ndarray,
    *,
    algorithm: str,
    dominant_face: str | None = None,
    reproj_error_px: float | None = None,
    n_tags: int = 0,
) -> BoardPose:
    unique_ids = sorted(set(int(marker_id) for marker_id in ids_used))
    return BoardPose(
        ids_used=unique_ids,
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3),
        tvec_m=np.asarray(tvec, dtype=np.float64).reshape(3),
        R=np.asarray(R, dtype=np.float64).reshape(3, 3),
        quat_xyzw=Rotation.from_matrix(R).as_quat(),
        T_camera_cube=_make_transform(R, tvec),
        algorithm=algorithm,
        dominant_face=dominant_face,
        reproj_error_px=reproj_error_px,
        n_tags=int(n_tags or len(unique_ids)),
    )


def _make_transform(R: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform
