# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Run Orbbec ArUco 6D pose detection from V4L2 or ROS2 image topics."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import cv2

from deploy.vision.aruco.apriltag_tracker import (
    DEFAULT_APRILTAG_FAMILY,
    DEFAULT_APRILTAG_SIZE_M,
    AprilTagPose,
    AprilTagTracker,
)
from deploy.vision.aruco.camera_adapter import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_V4L2_DEVICE,
    DEFAULT_WIDTH,
    OrbbecCameraAdapter,
)
from deploy.vision.aruco.cube_layout import load_cube_layout
from deploy.vision.aruco.pose_estimator import ArucoDetections, ArucoPoseEstimator
from deploy.vision.aruco.visualization import draw_detections
from deploy.vision.aruco.wuji_cube_tracker import DEFAULT_OBSERVER_CONFIG_FILE


def main() -> int:
    args = _parse_args()

    cube_layout = None
    if args.cube.lower() != "none":
        cube_layout = load_cube_layout(args.cube)
        print(
            f"Loaded cube layout: {cube_layout.config_path} "
            f"({len(cube_layout.marker_ids)} tags, tag_size={cube_layout.tag_size:.4f}m)"
        )

    marker_length = args.marker_size if args.marker_size is not None else 0.013
    estimator = ArucoPoseEstimator(
        marker_length_m=marker_length,
        cube_layout=cube_layout,
        algorithm=args.algorithm,
        ema_alpha_t=args.ema_alpha,
        ema_alpha_r=args.ema_alpha,
        observer_config=args.observer_config,
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
        position_alpha=args.position_alpha,
        reproj_threshold=args.reproj_threshold,
    )
    apriltag_tracker = (
        AprilTagTracker(
            family=args.apriltag_family,
            tag_size_m=args.apriltag_size,
            target_id=args.apriltag_id,
        )
        if args.enable_apriltag
        else None
    )

    camera = OrbbecCameraAdapter(
        backend=args.backend,
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        rgb_topic=args.rgb_topic,
        depth_topic=args.depth_topic,
        camera_info_topic=args.camera_info_topic,
        calibration_file=args.calibration_file,
    )

    print(
        f"Orbbec ArUco demo running: backend={camera.backend}, "
        f"device={camera.device}, algorithm={args.algorithm}, "
        f"apriltag={args.enable_apriltag}, preview={not args.no_preview}"
    )
    print("Press q or ESC in the preview window to quit.")

    frame_idx = 0
    last_print = 0.0
    last_frame_time = time.perf_counter()
    fps_ema: float | None = None

    try:
        while True:
            frame = camera.wait_for_rgb_frame(timeout_s=2.0) if args.once else camera.get_rgb_frame()
            if frame is None:
                if args.once:
                    raise RuntimeError("timed out waiting for an RGB frame")
                time.sleep(0.01)
                continue

            frame_idx += 1
            now_perf = time.perf_counter()
            dt = max(now_perf - last_frame_time, 1.0e-6)
            last_frame_time = now_perf
            fps = 1.0 / dt
            fps_ema = fps if fps_ema is None else 0.2 * fps + 0.8 * fps_ema

            camera_info = camera.get_camera_info()
            camera_matrix, dist_coeffs = camera_info.as_intrinsics()
            detections = estimator.detect(frame, camera_matrix, dist_coeffs)
            apriltag_poses = (
                apriltag_tracker.detect(frame, camera_matrix, dist_coeffs)
                if apriltag_tracker is not None
                else []
            )

            now = time.time()
            if args.once or now - last_print >= args.print_period:
                print(_detections_to_json(frame_idx, camera_info, detections, apriltag_poses))
                last_print = now

            if not args.no_preview:
                display = draw_detections(
                    frame,
                    detections,
                    camera_matrix,
                    dist_coeffs,
                    marker_axis_length_m=estimator.marker_length_m * 0.75,
                    board_axis_length_m=(cube_layout.cube_size * 0.5 if cube_layout else None),
                    apriltag_poses=apriltag_poses,
                    apriltag_axis_length_m=args.apriltag_size * 0.5,
                    fps=fps_ema,
                    calibrated=camera_info.calibrated,
                )
                cv2.imshow("aruco_debug", display)
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q")}:
                    break

            if args.once:
                break
    finally:
        camera.close()
        if not args.no_preview:
            cv2.destroyAllWindows()
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orbbec ArUco 6D pose demo")
    parser.add_argument("--backend", choices=["auto", "v4l2", "ros2"], default="auto")
    parser.add_argument("--device", default=DEFAULT_V4L2_DEVICE, help="V4L2 device path or index")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--rgb-topic", default=None)
    parser.add_argument("--depth-topic", default=None)
    parser.add_argument("--camera-info-topic", default=None)
    parser.add_argument("--calibration-file", default=None)
    parser.add_argument(
        "--cube",
        default="default",
        help="Cube tags config: default, none, a path, or suffix such as 36 / 40_5.",
    )
    parser.add_argument("--algorithm", choices=["wuji", "opencv"], default="wuji")
    parser.add_argument("--observer-config", default=str(DEFAULT_OBSERVER_CONFIG_FILE))
    parser.add_argument("--process-noise", type=float, default=None)
    parser.add_argument("--measurement-noise", type=float, default=None)
    parser.add_argument("--position-alpha", type=float, default=None)
    parser.add_argument("--reproj-threshold", type=float, default=None)
    parser.add_argument("--enable-apriltag", action="store_true", help="Enable Wuji-compatible AprilTag detection")
    parser.add_argument("--apriltag-family", default=DEFAULT_APRILTAG_FAMILY)
    parser.add_argument("--apriltag-size", type=float, default=DEFAULT_APRILTAG_SIZE_M)
    parser.add_argument("--apriltag-id", type=int, default=None, help="Only show this AprilTag ID; default shows all")
    parser.add_argument("--marker-size", type=float, default=None, help="Single-marker size in meters when --cube none")
    parser.add_argument("--ema-alpha", type=float, default=0.6, help="EMA alpha for translation and rotation smoothing")
    parser.add_argument("--print-period", type=float, default=0.5, help="Seconds between JSON status lines")
    parser.add_argument("--no-preview", action="store_true", help="Disable cv2.imshow preview")
    parser.add_argument("--once", action="store_true", help="Process one RGB frame and exit")
    return parser.parse_args()


def _detections_to_json(
    frame_idx: int,
    camera_info: Any,
    detections: ArucoDetections,
    apriltag_poses: list[AprilTagPose],
) -> str:
    payload: dict[str, Any] = {
        "timestamp": time.time(),
        "frame": frame_idx,
        "camera_frame": camera_info.frame_id,
        "camera_info_source": camera_info.source,
        "calibrated": bool(camera_info.calibrated),
        "markers": [
            {
                "marker_id": pose.marker_id,
                "translation_m": _xyz_dict(pose.tvec_m),
                "rotation_matrix": pose.R.tolist(),
                "quaternion_xyzw": pose.quat_xyzw.tolist(),
                "T_camera_marker": pose.T_camera_marker.tolist(),
            }
            for pose in detections.marker_poses
        ],
        "apriltags": [
            {
                "tag_id": pose.tag_id,
                "translation_m": _xyz_dict(pose.tvec_m),
                "rotation_matrix": pose.R.tolist(),
                "quaternion_xyzw": pose.quat_xyzw.tolist(),
                "T_camera_tag": pose.T_camera_tag.tolist(),
            }
            for pose in apriltag_poses
        ],
    }
    if detections.board_pose is not None:
        pose = detections.board_pose
        payload["board"] = {
            "algorithm": pose.algorithm,
            "ids_used": pose.ids_used,
            "dominant_face": pose.dominant_face,
            "reproj_error_px": pose.reproj_error_px,
            "n_tags": pose.n_tags,
            "translation_m": _xyz_dict(pose.tvec_m),
            "rotation_matrix": pose.R.tolist(),
            "quaternion_xyzw": pose.quat_xyzw.tolist(),
            "T_camera_cube": pose.T_camera_cube.tolist(),
        }
    else:
        payload["board"] = None
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _xyz_dict(vec: Any) -> dict[str, float]:
    return {
        "x": float(vec[0]),
        "y": float(vec[1]),
        "z": float(vec[2]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
