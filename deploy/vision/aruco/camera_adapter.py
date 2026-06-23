# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Camera adapter layer for Orbbec RGB-D cameras.

The default path is intentionally lightweight: Orbbec Gemini devices expose
their color stream as UVC, so OpenCV/V4L2 can provide RGB frames without
pulling ROS2 or Orbbec SDK bindings into the deploy environment. ROS2 support
is lazy-loaded so this module remains importable on machines without ROS2.
"""

from __future__ import annotations

import glob
import os
import subprocess
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_V4L2_DEVICE = "/dev/video2"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30


@dataclass(frozen=True)
class CameraInfoData:
    """Minimal camera info needed by OpenCV pose estimation."""

    width: int
    height: int
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    frame_id: str = "camera_color_optical_frame"
    calibrated: bool = False
    source: str = "fallback"

    def as_intrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        return self.camera_matrix.copy(), self.dist_coeffs.copy()


def make_default_camera_info(
    width: int,
    height: int,
    *,
    frame_id: str = "camera_color_optical_frame",
) -> CameraInfoData:
    """Create a conservative pinhole model when no calibration is available."""
    focal = float(max(width, height))
    camera_matrix = np.array(
        [
            [focal, 0.0, (float(width) - 1.0) * 0.5],
            [0.0, focal, (float(height) - 1.0) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return CameraInfoData(
        width=int(width),
        height=int(height),
        camera_matrix=camera_matrix,
        dist_coeffs=np.zeros(5, dtype=np.float64),
        frame_id=frame_id,
        calibrated=False,
        source="fallback",
    )


class OrbbecCameraAdapter:
    """Unified RGB/depth/camera-info access for Orbbec demos.

    `get_rgb_frame()` returns an OpenCV BGR image. The name follows the camera
    stream role, while the memory layout follows OpenCV conventions.
    """

    def __init__(
        self,
        *,
        backend: str = "auto",
        device: str | int | None = DEFAULT_V4L2_DEVICE,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
        rgb_topic: str | None = None,
        depth_topic: str | None = None,
        camera_info_topic: str | None = None,
        calibration_file: str | os.PathLike[str] | None = None,
        open_device: bool = True,
    ) -> None:
        self.requested_backend = backend
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.camera_info_topic = camera_info_topic
        self.calibration_file = Path(calibration_file) if calibration_file else None

        self.backend = self._resolve_backend(backend)
        self._cap: cv2.VideoCapture | None = None
        self._ros_node: Any | None = None
        self._ros_thread: threading.Thread | None = None
        self._ros_bridge: Any | None = None
        self._rclpy: Any | None = None
        self._ros_started_here = False
        self._latest_rgb: np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None
        self._latest_camera_info: CameraInfoData | None = None
        self._lock = threading.Lock()
        self._warned_default_calibration = False
        self._warned_no_depth = False

        self._file_camera_info = self._load_calibration_file(self.calibration_file)
        if open_device:
            self.open()

    def _resolve_backend(self, backend: str) -> str:
        normalized = backend.lower()
        if normalized not in {"auto", "v4l2", "ros2"}:
            raise ValueError(f"unsupported camera backend: {backend!r}")
        if normalized == "auto":
            if self.rgb_topic or self.camera_info_topic:
                return "ros2"
            return "v4l2"
        return normalized

    def open(self) -> None:
        if self.backend == "v4l2":
            self._open_v4l2()
        elif self.backend == "ros2":
            self._open_ros2()

    def _open_v4l2(self) -> None:
        if self._cap is not None:
            return
        selected = self._select_v4l2_device(self.device)
        errors: list[str] = []
        for device in _candidate_v4l2_devices(selected):
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if not cap.isOpened():
                errors.append(f"{device!r}: open failed")
                cap.release()
                continue

            preferred_fourcc = _preferred_fourcc(device)
            if preferred_fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*preferred_fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ok, frame = cap.read()
            if not ok or frame is None:
                errors.append(f"{device!r}: read failed")
                cap.release()
                continue
            if frame.ndim != 3 or frame.shape[2] < 3:
                errors.append(f"{device!r}: not a color frame, shape={frame.shape}")
                cap.release()
                continue

            if device != selected:
                warnings.warn(
                    f"Requested V4L2 device {selected!r} was not usable as RGB; using {device!r}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.height, self.width = frame.shape[:2]
            self.device = device
            self._cap = cap
            return

        raise RuntimeError("failed to open a V4L2 RGB camera device; tried " + "; ".join(errors))

    def _open_ros2(self) -> None:
        if self._ros_node is not None:
            return
        try:
            import rclpy
            from cv_bridge import CvBridge
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 backend requested, but rclpy/sensor_msgs/cv_bridge are not importable. "
                "Start this demo inside a ROS2 Python environment or use --backend v4l2."
            ) from exc

        if not rclpy.ok():
            rclpy.init(args=None)
            self._ros_started_here = True
        self._rclpy = rclpy
        self._ros_bridge = CvBridge()
        self._ros_node = rclpy.create_node("orbbec_aruco_camera_adapter")

        rgb_topic = self.rgb_topic or "/camera/color/image_raw"
        self._ros_node.create_subscription(Image, rgb_topic, self._on_ros_rgb, 10)
        if self.depth_topic:
            self._ros_node.create_subscription(Image, self.depth_topic, self._on_ros_depth, 10)
        if self.camera_info_topic:
            self._ros_node.create_subscription(CameraInfo, self.camera_info_topic, self._on_ros_camera_info, 10)

        self._ros_thread = threading.Thread(target=rclpy.spin, args=(self._ros_node,), daemon=True)
        self._ros_thread.start()

    def _on_ros_rgb(self, msg: Any) -> None:
        frame = self._ros_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._lock:
            self._latest_rgb = frame
            self.width = int(msg.width)
            self.height = int(msg.height)

    def _on_ros_depth(self, msg: Any) -> None:
        depth = self._ros_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        with self._lock:
            self._latest_depth = depth

    def _on_ros_camera_info(self, msg: Any) -> None:
        camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        dist_coeffs = np.asarray(msg.d, dtype=np.float64)
        if dist_coeffs.size == 0:
            dist_coeffs = np.zeros(5, dtype=np.float64)
        frame_id = getattr(getattr(msg, "header", None), "frame_id", "") or "camera_color_optical_frame"
        info = CameraInfoData(
            width=int(msg.width),
            height=int(msg.height),
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            frame_id=frame_id,
            calibrated=bool(camera_matrix[0, 0] and camera_matrix[1, 1]),
            source="ros2",
        )
        with self._lock:
            self._latest_camera_info = info

    def get_rgb_frame(self) -> np.ndarray | None:
        if self.backend == "v4l2":
            if self._cap is None:
                self._open_v4l2()
            ok, frame = self._cap.read()
            if not ok or frame is None:
                return None
            self.height, self.width = frame.shape[:2]
            return frame

        with self._lock:
            if self._latest_rgb is None:
                return None
            return self._latest_rgb.copy()

    def get_depth_frame(self) -> np.ndarray | None:
        if self.backend == "v4l2":
            if not self._warned_no_depth:
                warnings.warn(
                    "V4L2 backend exposes the Orbbec color stream only; depth frame is unavailable.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_no_depth = True
            return None

        with self._lock:
            if self._latest_depth is None:
                return None
            return self._latest_depth.copy()

    def get_camera_info(self) -> CameraInfoData:
        if self.backend == "ros2":
            with self._lock:
                if self._latest_camera_info is not None:
                    return self._latest_camera_info

        if self._file_camera_info is not None:
            return self._resize_camera_info(self._file_camera_info, self.width, self.height)

        if not self._warned_default_calibration:
            warnings.warn(
                "No camera calibration found; using default pinhole intrinsics. "
                "Run camera calibration for metric-quality pose estimates.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_default_calibration = True
        return make_default_camera_info(self.width, self.height)

    def get_intrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        return self.get_camera_info().as_intrinsics()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._ros_node is not None and self._rclpy is not None:
            self._ros_node.destroy_node()
            if self._ros_started_here:
                self._rclpy.shutdown()
            if self._ros_thread is not None:
                self._ros_thread.join(timeout=1.0)
            self._ros_node = None
            self._ros_thread = None

    def __enter__(self) -> "OrbbecCameraAdapter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def wait_for_rgb_frame(self, timeout_s: float = 5.0) -> np.ndarray | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frame = self.get_rgb_frame()
            if frame is not None:
                return frame
            time.sleep(0.01)
        return None

    @staticmethod
    def _select_v4l2_device(device: str | int | None) -> str | int:
        if device is not None:
            return int(device) if isinstance(device, str) and device.isdigit() else device

        discovered = _discover_orbbec_v4l2_devices()
        if discovered:
            return discovered[0]
        if os.path.exists(DEFAULT_V4L2_DEVICE):
            return DEFAULT_V4L2_DEVICE
        video_nodes = sorted(glob.glob("/dev/video*"))
        if video_nodes:
            return video_nodes[0]
        raise RuntimeError("no V4L2 camera devices found")

    @staticmethod
    def _load_calibration_file(path: Path | None) -> CameraInfoData | None:
        if path is None:
            return None
        if not path.exists():
            raise FileNotFoundError(f"camera calibration file not found: {path}")
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("pyyaml is required to load camera calibration files") from exc

        with path.open("r") as f:
            cfg = yaml.safe_load(f) or {}

        if "camera_matrix" in cfg:
            data = cfg["camera_matrix"].get("data", cfg["camera_matrix"])
            camera_matrix = np.asarray(data, dtype=np.float64).reshape(3, 3)
            dist_data = cfg.get("distortion_coefficients", {}).get("data", [])
            dist_coeffs = np.asarray(dist_data or [0, 0, 0, 0, 0], dtype=np.float64)
            return CameraInfoData(
                width=int(cfg.get("image_width", 0) or camera_matrix[0, 2] * 2 + 1),
                height=int(cfg.get("image_height", 0) or camera_matrix[1, 2] * 2 + 1),
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                calibrated=True,
                source=str(path),
            )

        if "intrinsics" in cfg:
            intr = cfg["intrinsics"]
            roi = cfg.get("roi", {})
            offset_x = float(roi.get("offset_x", 0.0))
            offset_y = float(roi.get("offset_y", 0.0))
            camera_matrix = np.array(
                [
                    [float(intr["fx"]), 0.0, float(intr["cx"]) - offset_x],
                    [0.0, float(intr["fy"]), float(intr["cy"]) - offset_y],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            dist = cfg.get("distortion", {})
            dist_coeffs = np.array(
                [
                    float(dist.get("k1", 0.0)),
                    float(dist.get("k2", 0.0)),
                    float(dist.get("p1", 0.0)),
                    float(dist.get("p2", 0.0)),
                    float(dist.get("k3", 0.0)),
                ],
                dtype=np.float64,
            )
            width = int(roi.get("width", cfg.get("image_width", 0)) or 0)
            height = int(roi.get("height", cfg.get("image_height", 0)) or 0)
            return CameraInfoData(
                width=width,
                height=height,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                calibrated=True,
                source=str(path),
            )

        raise ValueError(f"unsupported camera calibration file schema: {path}")

    @staticmethod
    def _resize_camera_info(info: CameraInfoData, width: int, height: int) -> CameraInfoData:
        if info.width <= 0 or info.height <= 0 or (info.width == width and info.height == height):
            return info
        sx = float(width) / float(info.width)
        sy = float(height) / float(info.height)
        camera_matrix = info.camera_matrix.copy()
        camera_matrix[0, 0] *= sx
        camera_matrix[0, 2] *= sx
        camera_matrix[1, 1] *= sy
        camera_matrix[1, 2] *= sy
        return CameraInfoData(
            width=width,
            height=height,
            camera_matrix=camera_matrix,
            dist_coeffs=info.dist_coeffs.copy(),
            frame_id=info.frame_id,
            calibrated=info.calibrated,
            source=info.source,
        )


def _discover_orbbec_v4l2_devices() -> list[str]:
    """Return likely Orbbec color video nodes, best effort."""
    try:
        proc = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []

    devices: list[str] = []
    current_is_orbbec = False
    for raw_line in proc.stdout.splitlines():
        line = raw_line.rstrip()
        if not line:
            current_is_orbbec = False
            continue
        if not line.startswith("\t"):
            current_is_orbbec = "orbbec" in line.lower()
            continue
        node = line.strip()
        if current_is_orbbec and node.startswith("/dev/video"):
            devices.append(node)
    return devices


def _candidate_v4l2_devices(preferred: str | int) -> list[str | int]:
    candidates: list[str | int] = []
    seen: set[str] = set()

    def add(device: str | int) -> None:
        key = _v4l2_device_key(device)
        if key not in seen:
            candidates.append(device)
            seen.add(key)

    add(preferred)

    orbbec_devices = sorted(
        _discover_orbbec_v4l2_devices(),
        key=lambda device: (_v4l2_device_score(device), str(device)),
    )
    for device in orbbec_devices:
        add(device)

    other_devices = sorted(
        glob.glob("/dev/video*"),
        key=lambda device: (_v4l2_device_score(device), str(device)),
    )
    for device in other_devices:
        add(device)

    return candidates


def _v4l2_device_key(device: str | int) -> str:
    return f"/dev/video{device}" if isinstance(device, int) else str(device)


def _preferred_fourcc(device: str | int) -> str | None:
    formats = _v4l2_formats(device)
    for fourcc in ("MJPG", "YUYV", "BGR3", "RGB3"):
        if fourcc in formats:
            return fourcc
    return None


def _v4l2_device_score(device: str | int) -> int:
    formats = _v4l2_formats(device)
    if "MJPG" in formats:
        return 0
    if "YUYV" in formats:
        return 1
    if formats.intersection({"BGR3", "RGB3", "BA24", "RGBP"}):
        return 2
    if formats.intersection({"Z16", "Z16 ", "GREY", "BA81", "NV12", "YV12"}):
        return 100
    if not formats:
        return 50
    return 20


def _v4l2_formats(device: str | int) -> set[str]:
    if isinstance(device, int):
        device = f"/dev/video{device}"
    try:
        proc = subprocess.run(
            ["v4l2-ctl", "-d", str(device), "--list-formats-ext"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return set()
    if proc.returncode != 0:
        return set()

    formats: set[str] = set()
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith("[") or "'" not in line:
            continue
        parts = line.split("'")
        if len(parts) >= 2:
            formats.add(parts[1].strip())
    return formats
