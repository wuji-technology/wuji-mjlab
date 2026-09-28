# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Software detection ROI; camera capture and calibration stay unchanged."""
from __future__ import annotations

import os
import tempfile

import cv2
import yaml

from lib.camera_config import CONFIG_FILE, load_camera_config


def frame_roi(cfg, shape):
    """Convert sensor-coordinate fast_roi to this captured frame, clipping bounds."""
    height, width = shape[:2]
    full = (0, 0, width, height)
    roi = cfg.get("fast_roi")
    if not roi:
        return full
    capture = cfg["roi"]
    x = int(roi["offset_x"]) - int(capture["offset_x"])
    y = int(roi["offset_y"]) - int(capture["offset_y"])
    right = min(width, x + int(roi["width"]))
    bottom = min(height, y + int(roi["height"]))
    x, y = max(0, x), max(0, y)
    if right <= x or bottom <= y:
        return full
    return x, y, right - x, bottom - y


def aligned_selection(rect, shape):
    """Expand a drag to 8-pixel boundaries and at least 64 pixels per side."""
    x, y, width, height = rect
    if width <= 0 or height <= 0:
        return None
    result = []
    for start, size, limit in ((x, width, shape[1]), (y, height, shape[0])):
        limit = (int(limit) // 8) * 8
        if limit < 64:
            raise ValueError("ROI selection needs a frame of at least 64 x 64 pixels")
        left = max(0, min(int(start) // 8 * 8, limit - 64))
        right = min(limit, max(left + 64, (int(start + size) + 7) // 8 * 8))
        result.append((left, right - left))
    return result[0][0], result[1][0], result[0][1], result[1][1]


def save_selection(cfg, rect, shape, config_file=None):
    """Atomically update only fast_roi in the latest YAML; return saved ROI."""
    rect = aligned_selection(rect, shape)
    if rect is None:
        return None
    x, y, width, height = rect
    roi = dict(offset_x=x + int(cfg["roi"]["offset_x"]),
               offset_y=y + int(cfg["roi"]["offset_y"]),
               width=width, height=height)
    path = os.path.abspath(config_file or CONFIG_FILE)
    latest = load_camera_config(path)
    latest["fast_roi"] = roi
    fd, temporary = tempfile.mkstemp(prefix=".camera-", suffix=".yaml", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as stream:
            yaml.safe_dump(latest, stream, sort_keys=False, allow_unicode=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, os.stat(path).st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return roi


class RoiSelector:
    """Nonblocking drag state: keep publishing using the old ROI while editing."""

    def __init__(self):
        self.editing = False
        self.dragging = False
        self.start = self.end = None

    def begin(self):
        self.editing = True
        self.dragging = False
        self.start = self.end = None

    def cancel(self):
        self.editing = self.dragging = False
        self.start = self.end = None

    def mouse(self, event, x, y, flags, param):
        if not self.editing:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = self.end = (x, y)
            self.dragging = True
        elif self.dragging and event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP):
            self.end = (x, y)
            if event == cv2.EVENT_LBUTTONUP:
                self.dragging = False

    @property
    def rect(self):
        if self.start is None or self.end is None:
            return None
        x, y = min(self.start[0], self.end[0]), min(self.start[1], self.end[1])
        return x, y, abs(self.start[0] - self.end[0]), abs(self.start[1] - self.end[1])

    def draw(self, image, roi):
        x, y, width, height = roi
        cv2.rectangle(image, (x, y), (x + width - 1, y + height - 1), (255, 200, 0), 1)
        message = "s: select pen ROI | f: full frame | w: resample | q: quit"
        if self.editing:
            message = "Drag pen workspace; Enter/Space: save; c/Esc: cancel"
            if self.rect:
                x, y, width, height = self.rect
                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 255), 2)
        cv2.putText(image, message, (10, image.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 200, 0), 1)
