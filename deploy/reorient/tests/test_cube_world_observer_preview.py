# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

import ctypes
import importlib.util
import json
from pathlib import Path
import threading
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml
import wuji_reorient_deploy.mvs_sdk as mvs_sdk

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cube_world_observer.py"


def load_observer(monkeypatch, path=_SCRIPT):
    package = ModuleType("MvImport")
    package.__path__ = []
    bindings = ModuleType("MvImport.MvCameraControl_class")

    class MvCamera:
        @staticmethod
        def MV_CC_Finalize():
            pass

    bindings.MvCamera = MvCamera
    bindings.MV_FRAME_OUT = lambda: SimpleNamespace(stFrameInfo=SimpleNamespace(), pBufAddr=None)
    with monkeypatch.context() as patch:
        patch.setattr(mvs_sdk, "ensure_mvs_importable", lambda: None)
        patch.setitem(sys.modules, "MvImport", package)
        patch.setitem(sys.modules, bindings.__name__, bindings)
        spec = importlib.util.spec_from_file_location("offline_cube_world_observer", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class CameraDone(RuntimeError):
    pass


class FakeCamera:
    def __init__(self, duration=None, fail=False):
        self.duration = duration
        self.fail = fail
        self.started = time.monotonic()
        self.next_frame = self.started
        self.frames = 0
        self.closed = False
        self.roi_writes = []
        self.buffer = ctypes.create_string_buffer(128 * 128)

    def MV_CC_GetImageBuffer(self, frame, timeout):
        if self.duration is not None and time.monotonic() - self.started >= self.duration:
            raise CameraDone("fake camera stopped")
        delay = self.next_frame - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.next_frame = max(self.next_frame + 1 / 90, time.monotonic())
        self.frames += 1
        if self.fail and self.frames == 12:
            raise CameraDone("fake camera failed")
        ctypes.memset(self.buffer, self.frames % 256, len(self.buffer))
        frame.pBufAddr = ctypes.cast(self.buffer, ctypes.POINTER(ctypes.c_ubyte))
        frame.stFrameInfo.nWidth = 128
        frame.stFrameInfo.nHeight = 128
        frame.stFrameInfo.nFrameLen = len(self.buffer)
        return 0

    def MV_CC_FreeImageBuffer(self, frame):
        pass

    def MV_CC_StopGrabbing(self):
        pass

    def MV_CC_SetIntValueEx(self, key, value):
        self.roi_writes.append((key, value))

    def MV_CC_StartGrabbing(self):
        pass


    def MV_CC_CloseDevice(self):
        self.closed = True

    def MV_CC_DestroyHandle(self):
        pass


class Publisher:
    def __init__(self):
        self.frames = []

    def send_string(self, text, flags):
        self.frames.append(json.loads(text)["frame"])

    def close(self):
        pass


class Context:
    def term(self):
        pass


def make_observer(module, camera, preview):
    observer = object.__new__(module.CubeWorldObserver)
    observer.cam = camera
    observer.stOutFrame = module.MV_FRAME_OUT()
    observer.visualize = preview
    observer._grab_slow_count = 0
    observer.frame_count = observer.last_frame_count = 0
    observer.last_print_time = time.time()
    observer._display_fps = 0.0
    observer._world_fixed = True
    observer.world_pose = (np.eye(3), np.array([0.0, 0.0, 1.0]))
    observer._world_samples_R = []
    observer._world_sample_target = 10
    observer._clahe = None
    observer.aruco_detector = SimpleNamespace(detectMarkers=lambda image: (
        [np.array([[[20., 20.], [40., 20.], [40., 40.], [20., 40.]]], dtype=np.float32)],
        np.array([[0]], dtype=np.int32), []))
    observer.corner_filter = SimpleNamespace(update=lambda corners, ids: (corners, ids))
    observer.detect_cube_pose = lambda corners, ids: (np.eye(3), np.array([0., 0., 1.]), 1)
    observer.transform_to_world_frame = lambda rotation, translation: (rotation, translation)
    observer._R_cube_world = observer._t_cube_world = observer._dominant_face = None
    observer.prev_quat = None
    observer._cube_size = 0.054
    observer.zmq_socket = Publisher()
    observer.zmq_context = Context()
    observer.start_world_sampling = lambda: None
    observer._draw_world_axes = lambda image: None
    observer._draw_cube_axes_in_world = lambda image, rotation, translation: None
    observer.filter_R = SimpleNamespace(reset=lambda: None)
    observer.filter_t = SimpleNamespace(reset=lambda: None)
    return observer


@pytest.fixture
def offline(monkeypatch):
    module = load_observer(monkeypatch)
    monkeypatch.setattr(module, "_cam_cfg", {
        "roi": {"offset_x": 0, "offset_y": 0, "width": 128, "height": 128},
        "fast_roi": {"offset_x": 0, "offset_y": 0, "width": 128, "height": 128},
    })
    monkeypatch.setattr(module.cv2, "destroyAllWindows", lambda: None)
    return module


def run_offline(module, camera, preview):
    observer = make_observer(module, camera, preview)
    start = time.monotonic()
    try:
        with pytest.raises(CameraDone, match="fake camera stopped"):
            observer.run()
    finally:
        observer.cleanup()
    elapsed = time.monotonic() - start
    assert camera.closed
    assert observer.zmq_socket.frames == list(range(1, camera.frames + 1))
    return camera.frames / elapsed


@pytest.mark.slow
def test_slow_gui_keeps_publication_at_camera_rate(offline, monkeypatch):
    displays = []
    shown = []
    preview_camera = None

    def slow_gui(_title, image):
        assert threading.current_thread() is threading.main_thread()
        shown.append((int(image[600, 600, 0]), preview_camera.frames))
        time.sleep(0.018)
        displays.append(time.monotonic())

    def slow_poll_key():
        assert threading.current_thread() is threading.main_thread()
        time.sleep(0.018)
        return -1

    monkeypatch.setattr(offline.cv2, "imshow", slow_gui)
    monkeypatch.setattr(offline.cv2, "pollKey", slow_poll_key)
    baseline = run_offline(offline, FakeCamera(duration=2.2), preview=False)
    displays.clear()
    preview_camera = FakeCamera(duration=2.2)
    preview = run_offline(offline, preview_camera, preview=True)
    refresh = (len(displays) - 1) / (displays[-1] - displays[0])
    print(f"offline no-preview={baseline:.2f} fps preview={preview:.2f} fps gui={refresh:.2f} hz")
    assert preview >= 0.95 * baseline
    assert refresh <= 30
    assert 3 < len(displays) < 66
    assert all(0 <= produced - displayed <= 15 for displayed, produced in shown)


@pytest.mark.parametrize("exit_path", ["q", "camera_error", "keyboard_interrupt"])
def test_preview_exits_and_closes_camera(offline, monkeypatch, exit_path):
    camera = FakeCamera(fail=exit_path == "camera_error")
    observer = make_observer(offline, camera, preview=True)
    keys = iter([-1, -1, ord("q")])
    monkeypatch.setattr(offline.cv2, "imshow", lambda *args: None)

    def poll_key():
        if exit_path == "keyboard_interrupt":
            raise KeyboardInterrupt
        if exit_path == "camera_error":
            return -1
        return next(keys, -1)

    monkeypatch.setattr(offline.cv2, "pollKey", poll_key)
    try:
        if exit_path == "camera_error":
            with pytest.raises(CameraDone, match="fake camera failed"):
                observer.run()
        elif exit_path == "keyboard_interrupt":
            with pytest.raises(KeyboardInterrupt):
                observer.run()
        else:
            observer.run()
    finally:
        observer.cleanup()
    assert camera.closed
    assert not any(thread.name == "CubeWorldObserver-capture" for thread in threading.enumerate())


def test_preview_keys_keep_roi_selection_on_gui_and_apply_new_software_roi(offline, monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    camera_file = config_dir / "camera.yaml"
    camera_file.write_text(yaml.safe_dump({"roi": {"width": 128}, "fast_roi": {"width": 128}}))
    monkeypatch.setattr(offline, "ROOT_DIR", str(tmp_path))
    camera = FakeCamera()
    observer = make_observer(offline, camera, preview=True)
    selected_on = []
    shapes = []
    original_detector = observer.aruco_detector
    observer.aruco_detector = SimpleNamespace(detectMarkers=lambda image: (
        shapes.append(image.shape) or original_detector.detectMarkers(image)))
    observer.filter_R = SimpleNamespace(reset=lambda: selected_on.append("r"))
    observer.filter_t = SimpleNamespace(reset=lambda: None)
    observer.start_world_sampling = lambda: selected_on.append("w")
    monkeypatch.setattr(offline.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(offline.cv2, "destroyWindow", lambda *args: None)
    monkeypatch.setattr(offline.cv2, "selectROI", lambda *args, **kwargs: (
        selected_on.append(threading.current_thread().name) or (96, 96, 768, 576)))
    keys = iter([ord("s"), ord("r"), ord("w"), ord("q")])
    monkeypatch.setattr(offline.cv2, "pollKey", lambda: next(keys, -1))
    try:
        observer.run()
    finally:
        observer.cleanup()
    assert camera.closed
    assert selected_on.count("MainThread") == 1
    assert selected_on.count("r") == 2
    assert selected_on.count("w") == 2
    assert yaml.safe_load(camera_file.read_text())["fast_roi"] == {
        "offset_x": 8, "offset_y": 16, "width": 96, "height": 96,
    }
    assert (96, 96) in shapes

def test_preview_world_fix_never_downloads_fast_roi_to_camera(offline):
    offline._cam_cfg["fast_roi"] = {
        "offset_x": 8, "offset_y": 16, "width": 96, "height": 96,
    }
    camera = FakeCamera()
    observer = make_observer(offline, camera, preview=True)
    observer._world_samples_R = [np.eye(3)] * 10
    observer._world_samples_t = [np.array([0., 0., 1.])] * 10
    assert observer._finalize_world_frame()
    assert camera.roi_writes == []
