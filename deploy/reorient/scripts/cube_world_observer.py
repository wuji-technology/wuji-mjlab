#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Detects the cube pose relative to the world frame defined by the wrist AprilTag, and publishes it over ZMQ for run_policy.py and render_viewer.py (message schema: wuji_reorient_deploy/zmq_bridge.py)."""
from collections import deque
import json
import os
import sys
import queue
import threading
import time
from ctypes import *
from pathlib import Path

import cv2
import numpy as np
import yaml
import zmq
from scipy.linalg import inv
from scipy.spatial.transform import Rotation
from wuji_reorient_deploy.camera_config import (
    get_camera_matrix,
    get_dist_coeffs,
    load_camera_config,
    setup_camera_capture,
    setup_camera_roi,
)
from wuji_reorient_deploy.config_loader import cube_port
from wuji_reorient_deploy.cube_geom import (
    DEFAULT_CUBE_CONFIG_FILE as CUBE_CONFIG_FILE,
)
from wuji_reorient_deploy.cube_geom import (
    resolve_cube_config_path,
)
from wuji_reorient_deploy.mvs_sdk import ensure_mvs_importable

ROOT_DIR = str(Path(__file__).resolve().parents[1])

# The Hikvision MVS SDK is a vendor install (not on PyPI, not vendored here),
# so MvImport has to come off a system path; see wuji_reorient_deploy/mvs_sdk.py.
ensure_mvs_importable()

from MvImport.MvCameraControl_class import *

try:
    from pupil_apriltags import Detector as AprilTagDetector
except ImportError:
    print("ERROR: pupil_apriltags not installed. Run: pip install pupil-apriltags")
    sys.exit(1)

_cam_cfg = load_camera_config()
K = get_camera_matrix(_cam_cfg)
DIST_COEFFS = get_dist_coeffs(_cam_cfg)

OBSERVER_CONFIG_FILE = os.path.join(ROOT_DIR, "config", "observer.yaml")

WORLD_TAG_ID = 0
# pupil_apriltags expects tag size, not the full printed tile; any
# size error scales every position reconstructed in the wrist frame.
WORLD_TAG_SIZE = 0.0504  # AprilTag tag size (white/black border edge), 50.4 mm
WORLD_SAMPLE_FRAMES = 100

# AprilTag detector (X-right, Y-down, Z-into-tag) -> MuJoCo wrist tag:
# 180° rotation about X, preserving X and flipping Y and Z.
WORLD_FRAME_CORRECTION = np.array([
    [ 1.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0],
    [ 0.0,  0.0, -1.0],
])


def parse_axis_remap(remap_str):
    """Parse axis remapping string to rotation matrix.

    Args:
        remap_str: String like "+x +z -y" specifying how AprilTag axes map to new world axes
                   Format: "new_X new_Y new_Z" where each is ±x, ±y, or ±z
    """
    axis_map = {
        '+x': np.array([1, 0, 0]),
        '-x': np.array([-1, 0, 0]),
        '+y': np.array([0, 1, 0]),
        '-y': np.array([0, -1, 0]),
        '+z': np.array([0, 0, 1]),
        '-z': np.array([0, 0, -1]),
    }

    parts = remap_str.lower().split()
    if len(parts) != 3:
        raise ValueError(f"Axis remap must have 3 parts, got: {remap_str}")

    # Rows, not columns: row i is the AprilTag axis that becomes new axis i.
    R = np.zeros((3, 3))
    for i, part in enumerate(parts):
        if part not in axis_map:
            raise ValueError(f"Invalid axis: {part}. Use +x,-x,+y,-y,+z,-z")
        R[i, :] = axis_map[part]

    det = np.linalg.det(R)
    if not np.isclose(abs(det), 1.0):
        raise ValueError(f"Invalid axis remap: axes not orthogonal (det={det:.3f})")
    if det < 0:
        raise ValueError(f"Invalid axis remap: forms left-handed system (det={det:.3f}). "
                        "Hint: flip one axis sign to make it right-handed.")

    return R


# Rotation from the ArUco board's axes to the MuJoCo cube-mesh axes; the format
# matches WORLD_FRAME_CORRECTION (matrix or axis-remap string).
CUBE_FRAME_CORRECTION = None

FACE_COLORS = {
    'TOP':    ('Cyan',   (255, 255, 0)),    # BGR
    'BOTTOM': ('Blue',   (255, 0, 0)),
    'FRONT':  ('Red',    (0, 0, 255)),
    'BACK':   ('White',  (255, 255, 255)),
    'LEFT':   ('Green',  (0, 255, 0)),
    'RIGHT':  ('Yellow', (0, 255, 255)),
}


def load_observer_config():
    """Load observer configuration from YAML file."""
    defaults = {
        'rotation_filter': {
            'process_noise': 0.1,
            'measurement_noise': 0.3,
        },
        'position_filter': {
            'alpha': 0.6,
        },
        'pnp': {
            'reproj_threshold': 6.0,
        },
        'preprocess': {
            'enable_clahe': True,
            'clahe_clip': 2.0,
            'clahe_tile': [8, 8],
        },
    }

    if os.path.exists(OBSERVER_CONFIG_FILE):
        try:
            with open(OBSERVER_CONFIG_FILE, 'r') as f:
                cfg = yaml.safe_load(f)
            for key in defaults:
                if key in cfg:
                    defaults[key].update(cfg[key])
            print(f"Loaded observer config from {OBSERVER_CONFIG_FILE}")
        except Exception as e:
            print(f"Warning: Failed to load observer config: {e}, using defaults")

    return defaults


class SO3KalmanFilter:
    """SO(3) rotation Kalman filter in tangent space."""

    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        self.state = np.zeros(3)
        self.covariance = np.eye(3) * 0.1
        self.Q = np.eye(3) * process_noise
        self.R_noise = np.eye(3) * measurement_noise
        self.is_initialized = False
        self.reference_rot = np.eye(3)
        self.filtered_rot = np.eye(3)

    def _rotation_to_axis_angle(self, R):
        rvec, _ = cv2.Rodrigues(R)
        return rvec.flatten()

    def _axis_angle_to_rotation(self, rvec):
        R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        return R

    def update(self, rotation_matrix):
        if not self.is_initialized:
            self.reference_rot = rotation_matrix.copy()
            self.filtered_rot = rotation_matrix.copy()
            self.is_initialized = True
            return rotation_matrix

        R_relative = rotation_matrix @ self.reference_rot.T

        z_local = self._rotation_to_axis_angle(R_relative)

        self.covariance = self.covariance + self.Q
        S = self.covariance + self.R_noise
        K_gain = self.covariance @ inv(S)
        self.state = self.state + K_gain @ (z_local - self.state)
        self.covariance = (np.eye(3) - K_gain) @ self.covariance

        R_filtered_local = self._axis_angle_to_rotation(self.state)
        R_filtered_global = R_filtered_local @ self.reference_rot

        if np.linalg.norm(self.state) > 1.5:
            self.reference_rot = R_filtered_global.copy()
            self.state = np.zeros(3)

        self.filtered_rot = R_filtered_global
        return self.filtered_rot

    def reset(self):
        self.state = np.zeros(3)
        self.covariance = np.eye(3) * 0.1
        self.is_initialized = False
        self.reference_rot = np.eye(3)


class VectorLowPassFilter:
    """Simple low-pass filter for position."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.filtered_val = None

    def update(self, val):
        if self.filtered_val is None:
            self.filtered_val = val.copy()
            return self.filtered_val
        self.filtered_val = self.alpha * val + (1 - self.alpha) * self.filtered_val
        return self.filtered_val

    def reset(self):
        self.filtered_val = None


CORNER_FILTER_ALPHA = 1.0


class CornerEMAFilter:
    """Per-marker-ID corner EMA filter."""

    def __init__(self, alpha=CORNER_FILTER_ALPHA, max_age=5):
        self.alpha = alpha
        self.max_age = max_age
        self._state = {}
        self._age = {}

    def update(self, corners, ids):
        """Filter corners in-place and return (filtered_corners, ids)."""
        if ids is None or len(ids) == 0:
            for mid in list(self._age):
                self._age[mid] += 1
                if self._age[mid] > self.max_age:
                    del self._state[mid]
                    del self._age[mid]
            return corners, ids

        seen = set()
        filtered = []
        for i, mid in enumerate(ids.flatten()):
            mid = int(mid)
            seen.add(mid)
            pts = corners[i].reshape(4, 2).astype(np.float32)
            if mid in self._state:
                pts = self.alpha * pts + (1 - self.alpha) * self._state[mid]
            self._state[mid] = pts.copy()
            self._age[mid] = 0
            filtered.append(pts.reshape(1, 4, 2).astype(np.float32))

        for mid in list(self._age):
            if mid not in seen:
                self._age[mid] += 1
                if self._age[mid] > self.max_age:
                    del self._state[mid]
                    del self._age[mid]

        return filtered, ids

    def reset(self):
        self._state.clear()
        self._age.clear()


BACKLOG_LATENCY_S = 30.0e-3     # 30ms
BACKLOG_COUNT = 5                # consecutive slow grabs before flush
BACKLOG_MAX_FLUSH = 20           # safety cap on flush loop


class CubeWorldObserver:
    """Detects cube pose relative to world coordinate system defined by AprilTag."""

    def __init__(self, visualize=False, zmq_port=5555,
                 process_noise=0.01, measurement_noise=1.0, alpha=0.3,
                 world_sample_frames=WORLD_SAMPLE_FRAMES,
                 cube_config_path: str | None = None):
        self.visualize = visualize
        self._cube_config_path = cube_config_path or CUBE_CONFIG_FILE

        print("Initializing camera...")
        MvCamera.MV_CC_Initialize()
        deviceList = MV_CC_DEVICE_INFO_LIST()
        MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)

        if deviceList.nDeviceNum == 0:
            raise RuntimeError("No camera found!")

        self.cam = MvCamera()
        stDevice = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        self.cam.MV_CC_CreateHandle(stDevice)
        self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        self.cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_BayerGB8)
        setup_camera_capture(self.cam, _cam_cfg)
        setup_camera_roi(self.cam, _cam_cfg)
        self.cam.MV_CC_StartGrabbing()
        print("Camera ready!")

        self.apriltag_detector = AprilTagDetector(
            families="tag36h11", nthreads=4, quad_decimate=1.0,
            quad_sigma=0.0, decode_sharpening=0.25,
        )

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        # OpenCV 4.7 replaced the factory API with direct constructors.
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.aruco_detector = None
        else:
            aruco_params = cv2.aruco.DetectorParameters()
            aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, aruco_params)

        self._load_config()
        self._build_aruco_board()

        self.filter_R = SO3KalmanFilter(process_noise=process_noise, measurement_noise=measurement_noise)
        self.filter_t = VectorLowPassFilter(alpha=alpha)

        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.bind(f"tcp://*:{zmq_port}")
        print(f"ZMQ publisher on port {zmq_port}")

        self.stOutFrame = MV_FRAME_OUT()
        self.world_pose = None
        self.frame_count = 0
        self.last_print_time = 0
        self.last_frame_count = 0
        self._display_fps = 0.0
        self.filt_R = np.eye(3)
        self.filt_t = np.zeros(3)
        self.prev_quat = None
        self._R_cube_world = None
        self._t_cube_world = None
        self._dominant_face = None

        self._world_samples_R = []
        self._world_samples_t = []
        self._world_fixed = False
        self._world_sample_target = world_sample_frames

        _cfg = load_observer_config()
        self._reproj_threshold = float(_cfg['pnp']['reproj_threshold'])

        self._enable_clahe = bool(_cfg['preprocess']['enable_clahe'])
        _clip = float(_cfg['preprocess']['clahe_clip'])
        _tile = tuple(int(x) for x in _cfg['preprocess']['clahe_tile'])
        self._clahe = cv2.createCLAHE(clipLimit=_clip, tileGridSize=_tile) if self._enable_clahe else None

        self.corner_filter = CornerEMAFilter(alpha=CORNER_FILTER_ALPHA)

        self._ippe_locked_idx = 0
        self._lost_frames = 0
        self._prev_dominant_face = None
        self._active_faces = set()
        self._reproj_err = 0.0

        self._grab_slow_count = 0

    def _load_config(self):
        self._tag_map = None
        self._face_axes_cfg = None
        self._face_rotations = {'TOP': 0, 'BOTTOM': 0, 'FRONT': 0, 'BACK': 0, 'LEFT': 0, 'RIGHT': 0}

        if not os.path.exists(self._cube_config_path):
            raise FileNotFoundError(
                f"cube tags JSON not found: {self._cube_config_path}. "
                "Pass an existing path with --cube, or omit it for the "
                "default 54mm config."
            )

        try:
            with open(self._cube_config_path, 'r') as f:
                cfg = json.load(f)
            try:
                self._cube_size = float(cfg['cube_size'])
                self._tag_size = float(cfg['tag_size'])
                self._tag_offset = float(cfg['tag_center_offset'])
            except KeyError as e:
                raise KeyError(
                    f"{self._cube_config_path} is missing required key {e}; "
                    "cube_size, tag_size and tag_center_offset are not allowed to be defaulted."
                ) from None
            faces_cfg = cfg.get('faces_config', {})
            self._tag_map = {face: {int(k): v for k, v in tags.items()} for face, tags in faces_cfg.items()}
            self._face_axes_cfg = cfg.get('face_axes', None)
            for face, rot in cfg.get('face_rotations', {}).items():
                self._face_rotations[face] = rot
            print(f"Loaded cube config from {self._cube_config_path}")
            print(f"  cube_size={self._cube_size*1000:.1f}mm  "
                  f"tag_size={self._tag_size*1000:.2f}mm  "
                  f"tag_center_offset={self._tag_offset*1000:.2f}mm")
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse config JSON: {e}")

        self._tag_to_face = {}
        if self._tag_map:
            for face, tags in self._tag_map.items():
                for tid in tags.keys():
                    self._tag_to_face[tid] = face

    def _build_aruco_board(self):
        half = self._cube_size / 2
        ht = self._tag_size / 2
        off = self._tag_offset

        def rotate_corners(corners, rotation):
            n = (rotation // 90) % 4
            if n == 0: return corners
            elif n == 1: return np.array([corners[3], corners[0], corners[1], corners[2]])
            elif n == 2: return np.array([corners[2], corners[3], corners[0], corners[1]])
            else: return np.array([corners[1], corners[2], corners[3], corners[0]])

        def face_tags(face_center, u_axis, v_axis, rotation=0):
            tags = {}
            for pos, center in [('T', face_center + off * v_axis), ('B', face_center - off * v_axis),
                               ('L', face_center - off * u_axis), ('R', face_center + off * u_axis)]:
                corners = np.array([
                    center - ht * u_axis + ht * v_axis, center + ht * u_axis + ht * v_axis,
                    center + ht * u_axis - ht * v_axis, center - ht * u_axis - ht * v_axis,
                ], dtype=np.float32)
                tags[pos] = rotate_corners(corners, rotation)
            return tags

        if self._face_axes_cfg:
            faces = {name: (np.array(axes['center'], dtype=np.float64) * half,
                           np.array(axes['u'], dtype=np.float64),
                           np.array(axes['v'], dtype=np.float64))
                    for name, axes in self._face_axes_cfg.items()}
        else:
            X, Y, Z = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
            faces = {'TOP': (half*Z, X, Y), 'BOTTOM': (-half*Z, X, -Y), 'FRONT': (-half*Y, X, Z),
                    'BACK': (half*Y, -X, Z), 'LEFT': (-half*X, -Y, Z), 'RIGHT': (half*X, Y, Z)}

        tag_map = self._tag_map or {
            'TOP': {0:'L',1:'B',2:'T',3:'R'}, 'BOTTOM': {8:'R',9:'T',10:'B',11:'L'},
            'FRONT': {16:'R',17:'T',18:'B',19:'L'}, 'BACK': {20:'B',21:'R',22:'L',23:'T'},
            'LEFT': {4:'R',5:'T',6:'B',7:'L'}, 'RIGHT': {12:'B',13:'R',14:'L',15:'T'},
        }

        board_corners, board_ids = [], []
        for face_name, (center, u, v) in faces.items():
            tags = face_tags(center, u, v, self._face_rotations.get(face_name, 0))
            for tid, pos in tag_map[face_name].items():
                board_corners.append(tags[pos])
                board_ids.append([tid])

        sorted_idx = np.argsort([b[0] for b in board_ids])
        board_corners = [board_corners[i] for i in sorted_idx]
        board_ids = np.array([board_ids[i] for i in sorted_idx], dtype=np.int32)

        if hasattr(cv2.aruco, 'Board_create'):
            self.cube_board = cv2.aruco.Board_create(board_corners, self.aruco_dict, board_ids)
        else:
            self.cube_board = cv2.aruco.Board(board_corners, self.aruco_dict, board_ids)
        print(f"ArUco Board: {len(board_ids)} tags")

    def _match_image_points(self, corners, ids):
        if hasattr(self.cube_board, 'matchImagePoints'):
            return self.cube_board.matchImagePoints(corners, ids)
        else:
            obj_pts = []
            img_pts = []
            if ids is None or len(ids) == 0:
                return None, None

            board_ids_flat = self.cube_board.ids.flatten()
            for i, marker_id in enumerate(ids.flatten()):
                board_idx = np.where(board_ids_flat == marker_id)[0]
                if len(board_idx) > 0:
                    board_idx = board_idx[0]
                    obj_pts.extend(self.cube_board.objPoints[board_idx])
                    img_pts.extend(corners[i][0])

            if len(obj_pts) == 0:
                return None, None

            return np.array(obj_pts, dtype=np.float32), np.array(img_pts, dtype=np.float32)

    def detect_world_tag(self, gray):
        """Detect world AprilTag and return its pose in camera frame."""
        results = self.apriltag_detector.detect(
            gray, estimate_tag_pose=True,
            camera_params=(K[0, 0], K[1, 1], K[0, 2], K[1, 2]),
            tag_size=WORLD_TAG_SIZE
        )
        for r in results:
            if r.tag_id == WORLD_TAG_ID:
                return r.pose_R, r.pose_t.flatten(), r.corners
        return None, None, None

    def _average_rotations(self, rotations):
        if len(rotations) == 0:
            return np.eye(3)

        quats = []
        for R in rotations:
            rot = Rotation.from_matrix(R)
            q = rot.as_quat()  # (x, y, z, w)
            quats.append(q)

        quats = np.array(quats)

        for i in range(1, len(quats)):
            if np.dot(quats[i], quats[0]) < 0:
                quats[i] = -quats[i]

        avg_quat = quats.mean(axis=0)
        avg_quat /= np.linalg.norm(avg_quat)

        return Rotation.from_quat(avg_quat).as_matrix()

    def start_world_sampling(self):
        """Start/restart world frame sampling."""
        self._world_samples_R = []
        self._world_samples_t = []
        self._world_fixed = False
        self.world_pose = None
        print(f"\n[World Sampling] Starting... (collecting {self._world_sample_target} frames)")

    def _finalize_world_frame(self):
        if len(self._world_samples_R) < 10:
            print(f"[World Sampling] Failed: only {len(self._world_samples_R)} samples collected")
            return False

        avg_R = self._average_rotations(self._world_samples_R)

        avg_t = np.mean(self._world_samples_t, axis=0)

        if WORLD_FRAME_CORRECTION is not None:
            if isinstance(WORLD_FRAME_CORRECTION, str):
                correction_R = parse_axis_remap(WORLD_FRAME_CORRECTION)
                print(f"[World Sampling] Axis remap: {WORLD_FRAME_CORRECTION}")
            else:
                correction_R = np.array(WORLD_FRAME_CORRECTION)

            # avg_R maps AprilTag→camera and correction_R maps AprilTag→world,
            # so the corrected world→camera rotation is avg_R @ correction_R.T.
            avg_R = avg_R @ correction_R.T
            print(f"[World Sampling] Applied world frame correction (det={np.linalg.det(correction_R):.1f})")

        self.world_pose = (avg_R, avg_t)
        self._world_fixed = True

        print(f"[World Sampling] Complete! Averaged {len(self._world_samples_R)} samples")
        print(f"[World Sampling] World frame is now FIXED. Press 'w' to resample.")

        if not self.visualize:
            self._switch_to_fast_roi()

        return True

    def _switch_to_fast_roi(self):
        global K
        fast_roi = _cam_cfg.get('fast_roi')
        if fast_roi is None:
            return
        cur_roi = _cam_cfg['roi']
        if (fast_roi['width'] == cur_roi['width']
                and fast_roi['height'] == cur_roi['height']
                and fast_roi['offset_x'] == cur_roi['offset_x']
                and fast_roi['offset_y'] == cur_roi['offset_y']):
            return

        print(f"[Fast ROI] Switching to {fast_roi['width']}x{fast_roi['height']} "
              f"@ ({fast_roi['offset_x']}, {fast_roi['offset_y']}) ...")
        self.cam.MV_CC_StopGrabbing()
        self.cam.MV_CC_SetIntValueEx("OffsetX", 0)
        self.cam.MV_CC_SetIntValueEx("OffsetY", 0)
        self.cam.MV_CC_SetIntValueEx("Width", fast_roi['width'])
        self.cam.MV_CC_SetIntValueEx("Height", fast_roi['height'])
        self.cam.MV_CC_SetIntValueEx("OffsetX", fast_roi['offset_x'])
        self.cam.MV_CC_SetIntValueEx("OffsetY", fast_roi['offset_y'])
        self.cam.MV_CC_StartGrabbing()

        intr = _cam_cfg['intrinsics']
        K[0, 2] = intr['cx'] - fast_roi['offset_x']
        K[1, 2] = intr['cy'] - fast_roi['offset_y']
        print(f"[Fast ROI] Active. K updated: cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")

    def detect_cube_pose(self, corners, ids):
        """Detect cube pose via IPPE + ITERATIVE hybrid with dominant-face strategy."""
        if ids is None or len(ids) == 0:
            self._dominant_face = None
            self._lost_frames += 1
            return None, None, 0

        face_counts = {}
        for tid in ids.flatten():
            if int(tid) in self._tag_to_face:
                face = self._tag_to_face[int(tid)]
                face_counts[face] = face_counts.get(face, 0) + 1

        if not face_counts:
            self._dominant_face = None
            self._lost_frames += 1
            return None, None, 0

        best_face = max(face_counts, key=face_counts.get)
        if (self._prev_dominant_face is not None
                and self._prev_dominant_face in face_counts
                and face_counts.get(self._prev_dominant_face, 0) >= face_counts[best_face]):
            best_face = self._prev_dominant_face
        self._dominant_face = best_face
        self._prev_dominant_face = best_face
        self._active_faces = {best_face}

        valid_indices = [i for i, tid in enumerate(ids.flatten())
                         if int(tid) in self._tag_to_face and self._tag_to_face[int(tid)] == best_face]
        if valid_indices:
            corners = [corners[i] for i in valid_indices]
            ids = ids[valid_indices]

        obj_pts, img_pts = self._match_image_points(corners, ids)
        if obj_pts is None or len(obj_pts) < 4:
            self._lost_frames += 1
            return None, None, 0

        # IPPE returns both coplanar solutions; sol 0 has the lower reproj error.
        n_sol, rvecs_ippe, tvecs_ippe, reproj_errors = cv2.solvePnPGeneric(
            obj_pts, img_pts, K, DIST_COEFFS, flags=cv2.SOLVEPNP_IPPE)

        if n_sol == 0:
            self._lost_frames += 1
            return None, None, 0

        if n_sol == 1:
            best_idx = 0
        elif not self.filter_R.is_initialized or self._lost_frames > 0:
            best_idx = 0
        else:
            R_prev = self.filt_R
            dists = []
            for i in range(n_sol):
                R_i, _ = cv2.Rodrigues(rvecs_ippe[i])
                diff = cv2.Rodrigues(R_prev.T @ R_i)[0]
                dists.append(np.linalg.norm(diff))

            locked = self._ippe_locked_idx
            other = 1 - locked
            re_locked = reproj_errors[locked].item()
            re_other = reproj_errors[other].item()

            if (re_other < re_locked * 0.8) and (dists[other] < dists[locked] * 0.33):
                best_idx = other
            else:
                best_idx = locked

        self._ippe_locked_idx = best_idx
        pick_rvec, pick_tvec = rvecs_ippe[best_idx], tvecs_ippe[best_idx]

        success, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, DIST_COEFFS,
            rvec=pick_rvec.copy(), tvec=pick_tvec.copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            self._lost_frames += 1
            return None, None, 0

        reproj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, DIST_COEFFS)
        reproj_err = float(np.mean(np.linalg.norm(
            img_pts.reshape(-1, 2) - reproj_pts.reshape(-1, 2), axis=1)))
        self._reproj_err = reproj_err
        if reproj_err > self._reproj_threshold:
            self._lost_frames += 1
            return None, None, 0

        if self._lost_frames > 0:
            self.corner_filter.reset()
            self.filter_R.reset()
            self.filter_t.reset()
            self.prev_quat = None

        self._lost_frames = 0

        R, _ = cv2.Rodrigues(rvec)
        self.filt_R = self.filter_R.update(R)
        self.filt_t = self.filter_t.update(tvec.flatten())

        return self.filt_R, self.filt_t, len(ids)

    def transform_to_world_frame(self, R_cube_cam, t_cube_cam):
        """Transform cube pose from camera frame to world frame."""
        if self.world_pose is None:
            return None, None
        R_world_cam, t_world_cam = self.world_pose
        R_cam_world = R_world_cam.T
        t_cam_world = -R_cam_world @ t_world_cam
        R_cube_world = R_cam_world @ R_cube_cam
        t_cube_world = R_cam_world @ t_cube_cam + t_cam_world
        return R_cube_world, t_cube_world

    def _draw_world_axes(self, img, axis_length=0.033, line_width=5):
        if self.world_pose is None:
            return

        R_world, t_world = self.world_pose

        origin = t_world.reshape(3, 1)
        x_end = origin + axis_length * R_world[:, 0:1]
        y_end = origin + axis_length * R_world[:, 1:2]
        z_end = origin + axis_length * R_world[:, 2:3]

        origin_2d, _ = cv2.projectPoints(origin.T, np.zeros(3), np.zeros(3), K, DIST_COEFFS)
        x_2d, _ = cv2.projectPoints(x_end.T, np.zeros(3), np.zeros(3), K, DIST_COEFFS)
        y_2d, _ = cv2.projectPoints(y_end.T, np.zeros(3), np.zeros(3), K, DIST_COEFFS)
        z_2d, _ = cv2.projectPoints(z_end.T, np.zeros(3), np.zeros(3), K, DIST_COEFFS)

        origin_pt = tuple(origin_2d[0, 0].astype(int))
        x_pt = tuple(x_2d[0, 0].astype(int))
        y_pt = tuple(y_2d[0, 0].astype(int))
        z_pt = tuple(z_2d[0, 0].astype(int))

        # BGR, not RGB: X=red, Y=green, Z=blue.
        cv2.arrowedLine(img, origin_pt, x_pt, (0, 0, 255), line_width, tipLength=0.3)
        cv2.arrowedLine(img, origin_pt, y_pt, (0, 255, 0), line_width, tipLength=0.3)
        cv2.arrowedLine(img, origin_pt, z_pt, (255, 0, 0), line_width, tipLength=0.3)

        cv2.putText(img, "+X", x_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, "+Y", y_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, "+Z", z_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.circle(img, origin_pt, 5, (255, 255, 255), -1)

    def _draw_cube_axes_in_world(self, img, R_cube_world, t_cube_world, axis_length=0.025, line_width=2):
        if self.world_pose is None:
            return

        # world_pose = (R_world_cam, t_world_cam): R_world_cam's columns are the
        # world axes in camera frame, t_world_cam is the world origin there.
        R_world_cam, t_world_cam = self.world_pose

        t_cube_cam = R_world_cam @ t_cube_world + t_world_cam

        R_cube_cam = R_world_cam @ R_cube_world

        origin = t_cube_cam.reshape(3, 1)
        x_end = origin + axis_length * R_cube_cam[:, 0:1]
        y_end = origin + axis_length * R_cube_cam[:, 1:2]
        z_end = origin + axis_length * R_cube_cam[:, 2:3]

        origin_2d, _ = cv2.projectPoints(origin.T, np.zeros(3), np.zeros(3), K, DIST_COEFFS)
        x_2d, _ = cv2.projectPoints(x_end.T, np.zeros(3), np.zeros(3), K, DIST_COEFFS)
        y_2d, _ = cv2.projectPoints(y_end.T, np.zeros(3), np.zeros(3), K, DIST_COEFFS)
        z_2d, _ = cv2.projectPoints(z_end.T, np.zeros(3), np.zeros(3), K, DIST_COEFFS)

        origin_pt = tuple(origin_2d[0, 0].astype(int))
        x_pt = tuple(x_2d[0, 0].astype(int))
        y_pt = tuple(y_2d[0, 0].astype(int))
        z_pt = tuple(z_2d[0, 0].astype(int))

        cv2.arrowedLine(img, origin_pt, x_pt, (100, 100, 255), line_width, tipLength=0.3)
        cv2.arrowedLine(img, origin_pt, y_pt, (100, 255, 100), line_width, tipLength=0.3)
        cv2.arrowedLine(img, origin_pt, z_pt, (255, 100, 100), line_width, tipLength=0.3)

    def run(self):
        """Main detection loop."""
        print("\nCube World Observer running...")
        print("  Press 'q' to quit, 'r' to reset filters, 'w' to resample world frame\n")

        self.start_world_sampling()
        if not self.visualize:
            self._run_detection()
            return

        stop = threading.Event()
        commands = queue.SimpleQueue()
        latest = deque(maxlen=1)
        errors = []

        def capture():
            try:
                self._run_detection(stop, commands, latest)
            except BaseException as exc:
                errors.append(exc)
            finally:
                stop.set()

        worker = threading.Thread(target=capture, name="CubeWorldObserver-capture")
        worker.start()
        try:
            self._run_preview(stop, commands, latest)
        finally:
            stop.set()
            worker.join()
        if errors:
            raise errors[0]

    def _run_preview(self, stop, commands, latest):
        next_show = time.monotonic()
        shown_frame = None
        while not stop.is_set():
            delay = next_show - time.monotonic()
            if delay > 0:
                stop.wait(delay)
                continue
            if not latest and shown_frame is None:
                stop.wait(0.005)
                continue

            next_show = time.monotonic() + 1 / 30
            if latest:
                color, shown_frame = latest.pop()
                cv2.imshow('Cube World Observer', cv2.resize(color, (960, 768)))

            key = cv2.pollKey() & 0xFF
            if key == ord('q'):
                stop.set()
            elif key == ord('r') or key == ord('w'):
                commands.put(key)
            elif key == ord('s'):
                self._select_and_save_fast_roi(shown_frame)

    def _run_detection(self, stop=None, commands=None, latest=None):
        next_preview = time.monotonic() if self.visualize else None
        while stop is None or not stop.is_set():
            if commands is not None:
                try:
                    key = commands.get_nowait()
                except queue.Empty:
                    pass
                else:
                    if key == ord('r'):
                        self.filter_R.reset()
                        self.filter_t.reset()
                        self.prev_quat = None
                        print("Cube filters reset!")
                    elif key == ord('w'):
                        self.start_world_sampling()
                        self.filter_R.reset()
                        self.filter_t.reset()
                        self.prev_quat = None
            _tA = time.perf_counter()
            ret = self.cam.MV_CC_GetImageBuffer(self.stOutFrame, 100)
            if ret != 0:
                continue
            grab_dt = time.perf_counter() - _tA

            if grab_dt > BACKLOG_LATENCY_S:
                self._grab_slow_count += 1
                if self._grab_slow_count >= BACKLOG_COUNT:
                    self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)
                    flushed = 0
                    while flushed < BACKLOG_MAX_FLUSH:
                        r = self.cam.MV_CC_GetImageBuffer(self.stOutFrame, 1)
                        if r != 0:
                            break
                        self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)
                        flushed += 1
                    ret = self.cam.MV_CC_GetImageBuffer(self.stOutFrame, 100)
                    if ret != 0:
                        self._grab_slow_count = 0
                        continue
                    print(f"[FLUSH] buffer backlog detected (grab={grab_dt*1000:.1f}ms), "
                          f"drained {flushed} stale frames")
                    self._grab_slow_count = 0
            else:
                self._grab_slow_count = 0

            self.frame_count += 1
            nH = self.stOutFrame.stFrameInfo.nHeight
            nW = self.stOutFrame.stFrameInfo.nWidth
            data = string_at(self.stOutFrame.pBufAddr, self.stOutFrame.stFrameInfo.nFrameLen)
            bayer = np.frombuffer(data, dtype=np.uint8).reshape(nH, nW)

            bgr = cv2.cvtColor(bayer, cv2.COLOR_BayerGB2BGR)

            gray_min = np.minimum(np.minimum(bgr[:, :, 0], bgr[:, :, 1]), bgr[:, :, 2])

            if self._clahe is not None:
                gray = self._clahe.apply(gray_min)
            else:
                gray = gray_min

            color = bgr if self.visualize else None

            if not self._world_fixed:
                R_world, t_world, world_corners = self.detect_world_tag(gray)
                world_detected = R_world is not None
                if world_detected:
                    self._world_samples_R.append(R_world)
                    self._world_samples_t.append(t_world)
                    if len(self._world_samples_R) >= self._world_sample_target:
                        self._finalize_world_frame()
            else:
                world_detected = True
                world_corners = None

            fast_roi = _cam_cfg.get('fast_roi')
            _use_sw_roi = (self.visualize and self._world_fixed
                           and fast_roi is not None)
            if _use_sw_roi:
                rx, ry = fast_roi['offset_x'], fast_roi['offset_y']
                rw, rh = fast_roi['width'], fast_roi['height']
                gray_roi = gray[ry:ry+rh, rx:rx+rw]
            else:
                gray_roi = gray
                rx, ry = 0, 0

            if self.aruco_detector is None:
                corners, ids, _ = cv2.aruco.detectMarkers(gray_roi, self.aruco_dict, parameters=self.aruco_params)
            else:
                corners, ids, _ = self.aruco_detector.detectMarkers(gray_roi)

            if _use_sw_roi and ids is not None and len(ids) > 0:
                corners = [c + np.array([[[rx, ry]]], dtype=c.dtype) for c in corners]
            cube_quat_world = None
            cube_pos_world = None
            n_tags = 0

            if ids is not None and len(ids) > 0:
                mask = (ids.flatten() >= 0) & (ids.flatten() <= 23)
                if mask.any():
                    corners = [corners[i] for i in range(len(corners)) if mask[i]]
                    ids = ids[mask]
                else:
                    corners, ids = [], None

            if ids is not None and len(ids) > 0:
                corners, ids = self.corner_filter.update(corners, ids)
            else:
                self.corner_filter.update([], None)

            R_cube_cam = None
            if ids is not None and len(ids) > 0:
                R_cube_cam, t_cube_cam, n_tags = self.detect_cube_pose(corners, ids)
            else:
                # No cube markers this frame — still counts as a lost frame so the
                # IPPE disambiguation reset logic sees a fresh reacquire next time.
                self._lost_frames += 1

            if R_cube_cam is not None and self.world_pose is not None:
                R_cube_world, t_cube_world = self.transform_to_world_frame(R_cube_cam, t_cube_cam)
                if R_cube_world is not None:
                    if CUBE_FRAME_CORRECTION is not None:
                        if isinstance(CUBE_FRAME_CORRECTION, str):
                            cube_correction_R = parse_axis_remap(CUBE_FRAME_CORRECTION)
                        else:
                            cube_correction_R = np.array(CUBE_FRAME_CORRECTION)
                        R_cube_world = R_cube_world @ cube_correction_R.T

                    self._R_cube_world = R_cube_world
                    self._t_cube_world = t_cube_world

                    rot = Rotation.from_matrix(R_cube_world)
                    quat = rot.as_quat()

                    # Sign continuity: q and -q are the same rotation, so keep
                    # the one closest to the previous frame's quaternion.
                    if self.prev_quat is not None:
                        if np.dot(quat, self.prev_quat) < 0:
                            quat = -quat
                    self.prev_quat = quat.copy()

                    cube_quat_world = quat
                    cube_pos_world = t_cube_world

            # Published pose is in the wrist-tag frame, quaternion in scipy
            # (x, y, z, w) order; CubeReceiver converts to MuJoCo (w, x, y, z).
            if cube_quat_world is not None:
                msg = {
                    'timestamp': time.time(),
                    'frame': self.frame_count,
                    'world_detected': world_detected,
                    'world_fixed': self._world_fixed,
                    'cube_size': float(self._cube_size),
                    'cube1': {
                        'position': {'x': float(cube_pos_world[0]), 'y': float(cube_pos_world[1]), 'z': float(cube_pos_world[2])},
                        'orientation': {'x': float(cube_quat_world[0]), 'y': float(cube_quat_world[1]),
                                       'z': float(cube_quat_world[2]), 'w': float(cube_quat_world[3])},
                        'timestamp': time.time(),
                    }
                }
                self.zmq_socket.send_string(json.dumps(msg), flags=zmq.NOBLOCK)

            if self.visualize and color is not None and time.monotonic() >= next_preview:
                if self._world_fixed and self.world_pose is not None:
                    self._draw_world_axes(color)

                if world_corners is not None:
                    cv2.polylines(color, [world_corners.astype(int)], True, (255, 0, 255), 3)

                if _use_sw_roi:
                    cv2.rectangle(color, (rx, ry), (rx + rw, ry + rh), (0, 200, 200), 2)

                if ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedMarkers(color, corners, ids)

                if n_tags > 0 and self._R_cube_world is not None:
                    self._draw_cube_axes_in_world(color, self._R_cube_world, self._t_cube_world)

                if not self._world_fixed:
                    n_samples = len(self._world_samples_R)
                    progress = n_samples / self._world_sample_target
                    bar_width = 200
                    bar_height = 20
                    cv2.rectangle(color, (10, 10), (10 + bar_width, 10 + bar_height), (50, 50, 50), -1)
                    cv2.rectangle(color, (10, 10), (10 + int(bar_width * progress), 10 + bar_height), (0, 255, 255), -1)
                    cv2.rectangle(color, (10, 10), (10 + bar_width, 10 + bar_height), (255, 255, 255), 1)
                    cv2.putText(color, f"World Sampling: {n_samples}/{self._world_sample_target}",
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    cv2.putText(color, "WORLD FIXED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    if world_detected:
                        cv2.putText(color, "(tag visible)", (180, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

                cv2.putText(color, f"Tags: {n_tags}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if self._dominant_face and self._dominant_face in FACE_COLORS:
                    face_name = self._dominant_face
                    color_name, face_bgr = FACE_COLORS[face_name]
                    cv2.rectangle(color, (10, 75), (40, 105), face_bgr, -1)
                    cv2.rectangle(color, (10, 75), (40, 105), (255, 255, 255), 1)
                    cv2.putText(color, f"{face_name} ({color_name})", (50, 98),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_bgr, 2)

                if cube_quat_world is not None:
                    rpy = Rotation.from_quat(cube_quat_world).as_euler('xyz', degrees=True)
                    cv2.putText(color, f"RPY: ({rpy[0]:+.1f}, {rpy[1]:+.1f}, {rpy[2]:+.1f})", (10, 130),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                fps_color = (0, 255, 0) if self._display_fps >= 20 else (0, 165, 255)
                fps_text = f"FPS: {self._display_fps:.1f}"
                (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.putText(color, fps_text, (color.shape[1] - tw - 10, th + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)

                cv2.putText(color, "q:quit  r:reset  w:resample world  s:select ROI", (10, 755),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                latest.append((color, bgr))
                next_preview = time.monotonic() + 1 / 30

            self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)

            now = time.time()
            if now - self.last_print_time > 2.0:
                elapsed = now - self.last_print_time if self.last_print_time > 0 else 1.0
                fps = (self.frame_count - self.last_frame_count) / elapsed
                self._display_fps = fps
                self.last_frame_count = self.frame_count
                if not self._world_fixed:
                    n_samples = len(self._world_samples_R)
                    print(f"[{self.frame_count:6d}] FPS: {fps:5.1f} | World Sampling: {n_samples}/{self._world_sample_target}")
                elif cube_quat_world is not None:
                    rpy = Rotation.from_quat(cube_quat_world).as_euler('xyz', degrees=True)
                    # cube_quat_world is (x,y,z,w) from scipy
                    qx, qy, qz, qw = cube_quat_world
                    px, py, pz = cube_pos_world
                    print(f"[{self.frame_count:6d}] FPS: {fps:5.1f} | World: FIXED | Tags: {n_tags} | "
                          f"Pos: ({px:+.4f}, {py:+.4f}, {pz:+.4f}) | "
                          f"Quat(wxyz): ({qw:+.4f}, {qx:+.4f}, {qy:+.4f}, {qz:+.4f}) | "
                          f"Quat(xyzw): ({qx:+.4f}, {qy:+.4f}, {qz:+.4f}, {qw:+.4f}) | "
                          f"RPY: ({rpy[0]:+6.1f}, {rpy[1]:+6.1f}, {rpy[2]:+6.1f})")
                else:
                    print(f"[{self.frame_count:6d}] FPS: {fps:5.1f} | World: FIXED | Cube: NOT DETECTED")
                self.last_print_time = now


    def _select_and_save_fast_roi(self, current_frame):
        import os

        if not isinstance(current_frame, np.ndarray) or current_frame.ndim != 3:
            print(f"[ROI] ERROR: expected BGR image, got {type(current_frame).__name__}")
            return

        print("\n[ROI] Drag a rectangle on the frame. ENTER/SPACE to confirm, C to cancel.")
        display_size = (960, 768)
        display = cv2.resize(current_frame, display_size)
        scale_x = current_frame.shape[1] / display_size[0]
        scale_y = current_frame.shape[0] / display_size[1]

        x, y, w, h = cv2.selectROI(
            "Select ROI (ENTER/SPACE=confirm, C=cancel)", display,
            showCrosshair=True, fromCenter=False,
        )
        cv2.destroyWindow("Select ROI (ENTER/SPACE=confirm, C=cancel)")

        if w == 0 or h == 0:
            print("[ROI] Selection cancelled.")
            return

        offset_x = int(round(x * scale_x))
        offset_y = int(round(y * scale_y))
        width = int(round(w * scale_x))
        height = int(round(h * scale_y))

        # Some Hikvision cameras require width/height to be multiples of 4 or 8
        width = (width // 8) * 8
        height = (height // 8) * 8
        offset_x = (offset_x // 8) * 8
        offset_y = (offset_y // 8) * 8
        if width < 64 or height < 64:
            print(f"[ROI] Selection too small ({width}x{height}), ignored.")
            return

        print(f"[ROI] New fast_roi: offset=({offset_x},{offset_y}) size={width}x{height}")

        yaml_path = os.path.join(ROOT_DIR, "config", "camera.yaml")
        try:
            with open(yaml_path, "r") as f:
                cfg = yaml.safe_load(f)
            cfg["fast_roi"] = {
                "offset_x": offset_x,
                "offset_y": offset_y,
                "width": width,
                "height": height,
            }
            tmp_path = yaml_path + ".tmp"
            with open(tmp_path, "w") as f:
                yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
            os.replace(tmp_path, yaml_path)
            print(f"[ROI] Saved to {yaml_path}")
        except Exception as exc:
            print(f"[ROI] Failed to save: {exc}")
            try:
                os.remove(yaml_path + ".tmp")
            except OSError:
                pass
            return

        _cam_cfg["fast_roi"] = {
            "offset_x": offset_x,
            "offset_y": offset_y,
            "width": width,
            "height": height,
        }
        print("[ROI] Applied. Next frames will use the new fast_roi.")

    def cleanup(self):
        """Release resources."""
        if self.visualize:
            cv2.destroyAllWindows()
        self.cam.MV_CC_StopGrabbing()
        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()
        self.zmq_socket.close()
        self.zmq_context.term()
        print("Cleanup done.")


def main():
    import argparse

    cfg = load_observer_config()

    parser = argparse.ArgumentParser(description="Cube World Observer")
    parser.add_argument('--preview', action='store_true', help="Show preview window")
    parser.add_argument('--port', type=int, default=None, help="ZMQ port (override config)")
    parser.add_argument('--process-noise', type=float, default=None, help="SO3 Kalman Q (override config)")
    parser.add_argument('--measurement-noise', type=float, default=None, help="SO3 Kalman R (override config)")
    parser.add_argument('--alpha', type=float, default=None, help="Position LP alpha (override config)")
    parser.add_argument('--world-samples', type=int, default=WORLD_SAMPLE_FRAMES,
                        help=f"Number of frames to sample for world frame (default: {WORLD_SAMPLE_FRAMES})")
    parser.add_argument('--cube', type=str, default=None,
                        help="Cube tags config path (absolute or relative to cwd). "
                             "Default: config/cube_tags.json (54mm).")
    args = parser.parse_args()
    cube_config_path = resolve_cube_config_path(args.cube)

    port = args.port if args.port is not None else cube_port()
    process_noise = args.process_noise if args.process_noise is not None else cfg['rotation_filter']['process_noise']
    measurement_noise = args.measurement_noise if args.measurement_noise is not None else cfg['rotation_filter']['measurement_noise']
    alpha = args.alpha if args.alpha is not None else cfg['position_filter']['alpha']

    print(f"Filter params: process_noise={process_noise}, measurement_noise={measurement_noise}, alpha={alpha}")
    print(f"World frame sampling: {args.world_samples} frames")

    observer = CubeWorldObserver(
        visualize=args.preview,
        zmq_port=port,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        alpha=alpha,
        world_sample_frames=args.world_samples,
        cube_config_path=cube_config_path,
    )
    try:
        observer.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        observer.cleanup()


if __name__ == "__main__":
    # OPENCV_LOG_LEVEL is only read while cv2 initialises, so setting it after
    # `import cv2` is a no-op; use the runtime API at the process entry point.
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    main()
