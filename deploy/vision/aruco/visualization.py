# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""OpenCV visualization helpers for ArUco pose debugging."""

from __future__ import annotations

import cv2
import numpy as np

from deploy.vision.aruco.apriltag_tracker import AprilTagPose
from deploy.vision.aruco.pose_estimator import ArucoDetections


def draw_detections(
    image_bgr: np.ndarray,
    detections: ArucoDetections,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    marker_axis_length_m: float,
    board_axis_length_m: float | None = None,
    apriltag_poses: list[AprilTagPose] | None = None,
    apriltag_axis_length_m: float = 0.024,
    fps: float | None = None,
    calibrated: bool = False,
) -> np.ndarray:
    """Draw marker outlines, IDs, axes, board axis, and FPS overlay."""
    out = image_bgr.copy()
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)

    if detections.ids is not None and len(detections.corners) > 0:
        cv2.aruco.drawDetectedMarkers(out, detections.corners, detections.ids)
        for marker_pose in detections.marker_poses:
            cv2.drawFrameAxes(
                out,
                camera_matrix,
                dist_coeffs,
                marker_pose.rvec.reshape(3, 1),
                marker_pose.tvec_m.reshape(3, 1),
                marker_axis_length_m,
                2,
            )

    apriltag_poses = apriltag_poses or []
    for tag_pose in apriltag_poses:
        _draw_apriltag(
            out,
            tag_pose,
            camera_matrix,
            dist_coeffs,
            axis_length_m=apriltag_axis_length_m,
        )

    if detections.board_pose is not None:
        length = board_axis_length_m or marker_axis_length_m * 2.0
        cv2.drawFrameAxes(
            out,
            camera_matrix,
            dist_coeffs,
            detections.board_pose.rvec.reshape(3, 1),
            detections.board_pose.tvec_m.reshape(3, 1),
            length,
            3,
        )

    _draw_status(out, detections, apriltag_count=len(apriltag_poses), fps=fps, calibrated=calibrated)
    return out


def _draw_apriltag(
    image_bgr: np.ndarray,
    pose: AprilTagPose,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    axis_length_m: float,
) -> None:
    corners = pose.corners.astype(int)
    center = tuple(pose.center.astype(int))
    cv2.polylines(image_bgr, [corners], True, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(image_bgr, center, 4, (255, 0, 255), -1, cv2.LINE_AA)
    label_pos = tuple(corners[0])
    cv2.putText(
        image_bgr,
        f"tag {pose.tag_id}",
        label_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image_bgr,
        f"tag {pose.tag_id}",
        label_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
    rvec, _ = cv2.Rodrigues(pose.R)
    cv2.drawFrameAxes(
        image_bgr,
        camera_matrix,
        dist_coeffs,
        rvec.reshape(3, 1),
        pose.tvec_m.reshape(3, 1),
        axis_length_m,
        2,
    )


def _draw_status(
    image_bgr: np.ndarray,
    detections: ArucoDetections,
    *,
    apriltag_count: int,
    fps: float | None,
    calibrated: bool,
) -> None:
    y = 28
    marker_count = 0 if detections.ids is None else len(detections.ids)
    lines = [
        f"markers: {marker_count}",
        f"apriltags: {apriltag_count}",
        f"calib: {'camera_info' if calibrated else 'fallback'}",
    ]
    if fps is not None:
        lines.insert(0, f"FPS: {fps:.1f}")
    if detections.board_pose is None:
        lines.append("board: no")
    else:
        pose = detections.board_pose
        lines.append(f"algorithm: {pose.algorithm}")
        lines.append(f"board: yes tags={pose.n_tags}")
        if pose.dominant_face is not None:
            lines.append(f"face: {pose.dominant_face}")
        if pose.reproj_error_px is not None:
            lines.append(f"reproj: {pose.reproj_error_px:.2f}px")

    for line in lines:
        cv2.putText(
            image_bgr,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_bgr,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28
