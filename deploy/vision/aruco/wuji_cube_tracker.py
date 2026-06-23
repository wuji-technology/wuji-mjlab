# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji cube ArUco tracking core, decoupled from the original camera observer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from deploy.vision.aruco.cube_layout import CubeLayout

DEFAULT_OBSERVER_CONFIG_FILE = (
    Path(__file__).resolve().parents[2] / "reorient" / "config" / "observer.yaml"
)
CORNER_FILTER_ALPHA = 1.0


@dataclass(frozen=True)
class WujiTrackerConfig:
    process_noise: float = 0.5
    measurement_noise: float = 0.1
    position_alpha: float = 0.8
    reproj_threshold: float = 6.0
    enable_clahe: bool = True
    clahe_clip: float = 2.0
    clahe_tile: tuple[int, int] = (8, 8)
    corner_filter_alpha: float = CORNER_FILTER_ALPHA


@dataclass(frozen=True)
class WujiCubePoseResult:
    R: np.ndarray
    tvec_m: np.ndarray
    rvec: np.ndarray
    ids_used: list[int]
    n_tags: int
    dominant_face: str | None
    reproj_error_px: float


def load_wuji_tracker_config(
    config_file: str | Path | None = DEFAULT_OBSERVER_CONFIG_FILE,
    *,
    process_noise: float | None = None,
    measurement_noise: float | None = None,
    position_alpha: float | None = None,
    reproj_threshold: float | None = None,
) -> WujiTrackerConfig:
    """Load observer.yaml-compatible defaults and apply CLI overrides."""
    if config_file is None:
        config_file = DEFAULT_OBSERVER_CONFIG_FILE
    values: dict[str, Any] = {
        "rotation_filter": {"process_noise": 0.5, "measurement_noise": 0.1},
        "position_filter": {"alpha": 0.8},
        "pnp": {"reproj_threshold": 6.0},
        "preprocess": {"enable_clahe": True, "clahe_clip": 2.0, "clahe_tile": [8, 8]},
    }
    if config_file is not None and Path(config_file).exists():
        with Path(config_file).open("r") as f:
            loaded = yaml.safe_load(f) or {}
        for section, defaults in values.items():
            if isinstance(loaded.get(section), dict):
                defaults.update(loaded[section])

    if process_noise is not None:
        values["rotation_filter"]["process_noise"] = process_noise
    if measurement_noise is not None:
        values["rotation_filter"]["measurement_noise"] = measurement_noise
    if position_alpha is not None:
        values["position_filter"]["alpha"] = position_alpha
    if reproj_threshold is not None:
        values["pnp"]["reproj_threshold"] = reproj_threshold

    tile = values["preprocess"]["clahe_tile"]
    return WujiTrackerConfig(
        process_noise=float(values["rotation_filter"]["process_noise"]),
        measurement_noise=float(values["rotation_filter"]["measurement_noise"]),
        position_alpha=float(values["position_filter"]["alpha"]),
        reproj_threshold=float(values["pnp"]["reproj_threshold"]),
        enable_clahe=bool(values["preprocess"]["enable_clahe"]),
        clahe_clip=float(values["preprocess"]["clahe_clip"]),
        clahe_tile=(int(tile[0]), int(tile[1])),
    )


class SO3KalmanFilter:
    """SO(3) rotation Kalman filter in tangent space, copied from Wuji observer."""

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1) -> None:
        self.state = np.zeros(3)
        self.covariance = np.eye(3) * 0.1
        self.Q = np.eye(3) * process_noise
        self.R_noise = np.eye(3) * measurement_noise
        self.is_initialized = False
        self.reference_rot = np.eye(3)
        self.filtered_rot = np.eye(3)

    def update(self, rotation_matrix: np.ndarray) -> np.ndarray:
        if not self.is_initialized:
            self.reference_rot = rotation_matrix.copy()
            self.filtered_rot = rotation_matrix.copy()
            self.is_initialized = True
            return rotation_matrix

        R_relative = rotation_matrix @ self.reference_rot.T
        z_local = cv2.Rodrigues(R_relative)[0].flatten()

        self.covariance = self.covariance + self.Q
        S = self.covariance + self.R_noise
        K_gain = self.covariance @ np.linalg.inv(S)
        self.state = self.state + K_gain @ (z_local - self.state)
        self.covariance = (np.eye(3) - K_gain) @ self.covariance

        R_filtered_local, _ = cv2.Rodrigues(self.state.reshape(3, 1))
        R_filtered_global = R_filtered_local @ self.reference_rot
        if np.linalg.norm(self.state) > 1.5:
            self.reference_rot = R_filtered_global.copy()
            self.state = np.zeros(3)

        self.filtered_rot = R_filtered_global
        return self.filtered_rot

    def reset(self) -> None:
        self.state = np.zeros(3)
        self.covariance = np.eye(3) * 0.1
        self.is_initialized = False
        self.reference_rot = np.eye(3)
        self.filtered_rot = np.eye(3)


class VectorLowPassFilter:
    """Simple low-pass filter for position."""

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = float(alpha)
        self.filtered_val: np.ndarray | None = None

    def update(self, val: np.ndarray) -> np.ndarray:
        val = np.asarray(val, dtype=np.float64).reshape(3)
        if self.filtered_val is None:
            self.filtered_val = val.copy()
            return self.filtered_val
        self.filtered_val = self.alpha * val + (1.0 - self.alpha) * self.filtered_val
        return self.filtered_val

    def reset(self) -> None:
        self.filtered_val = None


class CornerEMAFilter:
    """Per-marker-ID corner EMA filter from Wuji observer."""

    def __init__(self, alpha: float = CORNER_FILTER_ALPHA, max_age: int = 5) -> None:
        self.alpha = float(alpha)
        self.max_age = int(max_age)
        self._state: dict[int, np.ndarray] = {}
        self._age: dict[int, int] = {}

    def update(
        self,
        corners: list[np.ndarray],
        ids: np.ndarray | None,
    ) -> tuple[list[np.ndarray], np.ndarray | None]:
        if ids is None or len(ids) == 0:
            self._age_missing()
            return corners, ids

        seen: set[int] = set()
        filtered: list[np.ndarray] = []
        for idx, marker_id in enumerate(ids.flatten()):
            marker_id = int(marker_id)
            seen.add(marker_id)
            pts = corners[idx].reshape(4, 2).astype(np.float32)
            if marker_id in self._state:
                pts = self.alpha * pts + (1.0 - self.alpha) * self._state[marker_id]
            self._state[marker_id] = pts.copy()
            self._age[marker_id] = 0
            filtered.append(pts.reshape(1, 4, 2).astype(np.float32))

        for marker_id in list(self._age):
            if marker_id not in seen:
                self._age[marker_id] += 1
                if self._age[marker_id] > self.max_age:
                    del self._age[marker_id]
                    del self._state[marker_id]
        return filtered, ids

    def reset(self) -> None:
        self._state.clear()
        self._age.clear()

    def _age_missing(self) -> None:
        for marker_id in list(self._age):
            self._age[marker_id] += 1
            if self._age[marker_id] > self.max_age:
                del self._age[marker_id]
                del self._state[marker_id]


class WujiCubeTracker:
    """Wuji dominant-face IPPE+ITERATIVE cube pose tracker."""

    def __init__(
        self,
        cube_layout: CubeLayout,
        config: WujiTrackerConfig | None = None,
    ) -> None:
        self.cube_layout = cube_layout
        self.config = config or WujiTrackerConfig()
        self.filter_R = SO3KalmanFilter(
            process_noise=self.config.process_noise,
            measurement_noise=self.config.measurement_noise,
        )
        self.filter_t = VectorLowPassFilter(alpha=self.config.position_alpha)
        self.corner_filter = CornerEMAFilter(alpha=self.config.corner_filter_alpha)
        self.filt_R = np.eye(3)
        self.filt_t = np.zeros(3)
        self._ippe_locked_idx = 0
        self._lost_frames = 0
        self._prev_dominant_face: str | None = None
        self._dominant_face: str | None = None
        self._active_faces: set[str] = set()
        self._reproj_err = 0.0
        self._clahe = (
            cv2.createCLAHE(
                clipLimit=self.config.clahe_clip,
                tileGridSize=self.config.clahe_tile,
            )
            if self.config.enable_clahe
            else None
        )

    @property
    def dominant_face(self) -> str | None:
        return self._dominant_face

    @property
    def reproj_error_px(self) -> float:
        return self._reproj_err

    def preprocess_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """Wuji preprocessing: min-channel grayscale plus optional CLAHE."""
        if image_bgr.ndim == 2:
            gray = image_bgr
        elif image_bgr.ndim == 3 and image_bgr.shape[2] >= 3:
            gray = np.minimum(
                np.minimum(image_bgr[:, :, 0], image_bgr[:, :, 1]),
                image_bgr[:, :, 2],
            )
        else:
            raise ValueError(f"unsupported image shape for Wuji preprocessing: {image_bgr.shape}")

        if self._clahe is not None:
            return self._clahe.apply(gray)
        return gray

    def filter_detections(
        self,
        corners: list[np.ndarray],
        ids: np.ndarray | None,
    ) -> tuple[list[np.ndarray], np.ndarray | None]:
        if ids is None or len(ids) == 0:
            return self.corner_filter.update([], None)

        valid_indices = [
            idx
            for idx, marker_id in enumerate(ids.flatten())
            if int(marker_id) in self.cube_layout.tag_to_face
        ]
        if not valid_indices:
            return self.corner_filter.update([], None)

        filtered_corners = [corners[idx] for idx in valid_indices]
        filtered_ids = ids[valid_indices]
        return self.corner_filter.update(filtered_corners, filtered_ids)

    def detect_cube_pose(
        self,
        corners: list[np.ndarray],
        ids: np.ndarray | None,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> WujiCubePoseResult | None:
        """Detect cube pose via Wuji IPPE + ITERATIVE dominant-face strategy."""
        camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)

        if ids is None or len(ids) == 0:
            self._dominant_face = None
            self._lost_frames += 1
            return None

        face_counts = self._count_faces(ids)
        if not face_counts:
            self._dominant_face = None
            self._lost_frames += 1
            return None

        best_face = max(face_counts, key=face_counts.get)
        if (
            self._prev_dominant_face is not None
            and self._prev_dominant_face in face_counts
            and face_counts.get(self._prev_dominant_face, 0) >= face_counts[best_face]
        ):
            best_face = self._prev_dominant_face
        self._dominant_face = best_face
        self._prev_dominant_face = best_face
        self._active_faces = {best_face}

        valid_indices = [
            idx
            for idx, marker_id in enumerate(ids.flatten())
            if self.cube_layout.tag_to_face.get(int(marker_id)) == best_face
        ]
        if valid_indices:
            corners = [corners[idx] for idx in valid_indices]
            ids = ids[valid_indices]

        obj_pts, img_pts, ids_used = self.cube_layout.match_image_points(corners, ids)
        if obj_pts is None or img_pts is None or len(obj_pts) < 4:
            self._lost_frames += 1
            return None

        ippe_result = cv2.solvePnPGeneric(
            obj_pts,
            img_pts,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
        n_sol, rvecs_ippe, tvecs_ippe, reproj_errors = ippe_result[:4]
        if int(n_sol) == 0:
            self._lost_frames += 1
            return None

        best_idx = self._choose_ippe_solution(rvecs_ippe, reproj_errors)
        self._ippe_locked_idx = best_idx
        pick_rvec = np.asarray(rvecs_ippe[best_idx], dtype=np.float64).copy()
        pick_tvec = np.asarray(tvecs_ippe[best_idx], dtype=np.float64).copy()

        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            camera_matrix,
            dist_coeffs,
            rvec=pick_rvec,
            tvec=pick_tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            self._lost_frames += 1
            return None

        reproj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
        reproj_err = float(np.mean(np.linalg.norm(img_pts.reshape(-1, 2) - reproj_pts.reshape(-1, 2), axis=1)))
        self._reproj_err = reproj_err
        if reproj_err > self.config.reproj_threshold:
            self._lost_frames += 1
            return None

        if self._lost_frames > 0:
            self.corner_filter.reset()
            self.filter_R.reset()
            self.filter_t.reset()

        self._lost_frames = 0
        R, _ = cv2.Rodrigues(rvec)
        self.filt_R = self.filter_R.update(R)
        self.filt_t = self.filter_t.update(np.asarray(tvec, dtype=np.float64).reshape(3))
        filt_rvec, _ = cv2.Rodrigues(self.filt_R)

        return WujiCubePoseResult(
            R=self.filt_R.copy(),
            tvec_m=self.filt_t.copy(),
            rvec=filt_rvec.reshape(3),
            ids_used=ids_used,
            n_tags=len(ids_used),
            dominant_face=self._dominant_face,
            reproj_error_px=reproj_err,
        )

    def reset(self) -> None:
        self.corner_filter.reset()
        self.filter_R.reset()
        self.filter_t.reset()
        self.filt_R = np.eye(3)
        self.filt_t = np.zeros(3)
        self._ippe_locked_idx = 0
        self._lost_frames = 0
        self._prev_dominant_face = None
        self._dominant_face = None
        self._active_faces = set()
        self._reproj_err = 0.0

    def _count_faces(self, ids: np.ndarray) -> dict[str, int]:
        face_counts: dict[str, int] = {}
        for marker_id in ids.flatten():
            face = self.cube_layout.tag_to_face.get(int(marker_id))
            if face is not None:
                face_counts[face] = face_counts.get(face, 0) + 1
        return face_counts

    def _choose_ippe_solution(
        self,
        rvecs_ippe: list[np.ndarray] | tuple[np.ndarray, ...],
        reproj_errors: np.ndarray,
    ) -> int:
        n_sol = len(rvecs_ippe)
        if n_sol == 1:
            return 0
        if not self.filter_R.is_initialized or self._lost_frames > 0:
            return 0

        R_prev = self.filt_R
        dists = []
        for rvec in rvecs_ippe:
            R_i, _ = cv2.Rodrigues(rvec)
            diff = cv2.Rodrigues(R_prev.T @ R_i)[0]
            dists.append(float(np.linalg.norm(diff)))

        locked = min(self._ippe_locked_idx, n_sol - 1)
        other = 1 - locked if n_sol == 2 else int(np.argmin(reproj_errors))
        re_locked = float(np.asarray(reproj_errors[locked]).item())
        re_other = float(np.asarray(reproj_errors[other]).item())
        if (re_other < re_locked * 0.8) and (dists[other] < dists[locked] * 0.33):
            return other
        return locked
