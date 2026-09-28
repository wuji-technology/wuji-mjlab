#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Publish camera-based pen poses in the calibrated wrist-tag frame over ZMQ.

The pen uses its own 18-pattern codebook. Single-face poses are flagged as
coplanar; --min-faces 2 rejects them. Use --preview to inspect alignment.

Run from the repository root:
    pixi run -e pen-spin-deploy pen-observer --preview"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from functools import lru_cache

import numpy as np
import zmq
from lib.camera_config import (
  get_camera_matrix,
  get_dist_coeffs,
  load_camera_config,
)
from lib.math_utils import matrix_from_quat, quat_from_matrix
from lib.tag_frame import (
  WORLD_TAG_ID,
  WORLD_TAG_SIZE_M,
  WRIST_AXES_IN_OPENCV_TAG,
)
from lib.zmq_bridge import PEN_PORT


class WorldFrame:
  """Camera -> wrist-tag transform, averaged over the first N good frames.

  Held FIXED after that: the tag is on the wrist mount, and re-estimating it
  every frame would inject the tag's own detection noise into every pen pose.
  Press 'w' in the preview window to resample.
  """

  def __init__(self, sample_frames: int = 100):
    if sample_frames < 1:
      raise ValueError("sample_frames must be positive")
    self.sample_frames = sample_frames
    self._R: list[np.ndarray] = []
    self._t: list[np.ndarray] = []
    self.R_cw: np.ndarray | None = None    # camera -> world(tag)
    self.t_cw: np.ndarray | None = None

  @property
  def fixed(self) -> bool:
    return self.R_cw is not None

  @property
  def samples(self) -> int:
    """Good wrist-tag detections collected so far (for the operator's readout)."""
    return len(self._t)

  def reset(self) -> None:
    self._R.clear()
    self._t.clear()
    self.R_cw = self.t_cw = None

  def add(self, R_wc: np.ndarray, t_wc: np.ndarray) -> None:
    if self.fixed:
      return
    self._R.append(R_wc)
    self._t.append(t_wc)
    if len(self._R) >= self.sample_frames:
      self._finalise()

  def _finalise(self) -> None:
    import cv2
    # Average rotations via their quaternions (sign-aligned), not matrices.
    qs = np.array([quat_from_matrix(R) for R in self._R])
    ref = qs[0]
    qs[np.einsum("ij,j->i", qs, ref) < 0] *= -1.0
    q = qs.mean(0)
    q /= np.linalg.norm(q)
    R_wc = matrix_from_quat(q)
    t_wc = np.median(np.array(self._t), axis=0)
    self.R_cw = R_wc.T
    self.t_cw = -R_wc.T @ t_wc
    del cv2

  def to_world(self, R_cam, t_cam):
    """Pose given in the camera frame -> the wrist-tag frame."""
    return self.R_cw @ R_cam, self.R_cw @ t_cam + self.t_cw


@lru_cache(maxsize=1)
def _world_tag_detectors():
  import cv2
  dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
  coarse = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
  params = cv2.aruco.DetectorParameters()
  params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
  return coarse, cv2.aruco.ArucoDetector(dictionary, params)


def _detect_world_tag(gray, K, dist):
  """36h11 pose in the calibrated wrist convention, not raw OpenCV axes."""
  import cv2
  coarse, precise = _world_tag_detectors()
  corners, ids, _ = coarse.detectMarkers(gray)
  if ids is None or WORLD_TAG_ID not in ids.flatten():
    return None
  quad = corners[list(ids.flatten()).index(WORLD_TAG_ID)][0]
  # Integer coarse corners jumped by 1–3 px on a stationary real tag, causing
  # ~5-degree IPPE jumps. Fit AprilTag edges locally: full-image AprilTag
  # processing is expensive on the textured glove/table. The crop is located
  # anew every frame, not tracked or temporally smoothed.
  pad = max(8., .15 * np.max(np.ptp(quad, axis=0)))
  lo = np.maximum(np.floor(quad.min(axis=0) - pad).astype(int), 0)
  hi = np.minimum(np.ceil(quad.max(axis=0) + pad).astype(int), gray.shape[::-1])
  refined, refined_ids, _ = precise.detectMarkers(
      np.ascontiguousarray(gray[lo[1]:hi[1], lo[0]:hi[0]]))
  if refined_ids is None or WORLD_TAG_ID not in refined_ids.flatten():
    return None  # do not reintroduce coarse-corner jumps on failed refinement
  quad = refined[list(refined_ids.flatten()).index(WORLD_TAG_ID)][0].astype(np.float64) + lo
  h = WORLD_TAG_SIZE_M / 2.0
  obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
  ok, rvec, tvec = cv2.solvePnP(obj, quad.astype(np.float64), K, dist,
                                flags=cv2.SOLVEPNP_IPPE_SQUARE)
  if not ok:
    return None
  R, _ = cv2.Rodrigues(rvec)
  return R @ WRIST_AXES_IN_OPENCV_TAG, tvec.reshape(3)


_ARTWORK_CACHE: dict[tuple, np.ndarray] = {}


def _tag_artwork(tid: int, cell: int = 40, layout=None) -> np.ndarray:
  """One tag as it is physically printed: 8x8 cells, only the data inverted.

  Layout, outermost first: 1 black ring (the ArUco border), 1 white ring,
  then the 4x4 data with every bit flipped. Verified cell-by-cell against
  ``textures/outward_A_top_front.png``.
  """
  from lib.pen_pose import PenTagLayout
  layout = layout if layout is not None else PenTagLayout()
  key = (layout.dictionary.bytesList.tobytes(), tid, cell)
  if key in _ARTWORK_CACHE:
    return _ARTWORK_CACHE[key]
  import cv2
  d4 = layout.dictionary
  core = cv2.aruco.generateImageMarker(d4, tid, 6 * cell)   # 6x6: border+data
  img = np.zeros((8 * cell, 8 * cell), np.uint8)            # black quiet ring
  img[cell:7 * cell, cell:7 * cell] = 255                   # white ring
  data = core[cell:5 * cell, cell:5 * cell]                 # the 4x4 data
  img[2 * cell:6 * cell, 2 * cell:6 * cell] = 255 - data    # inverted
  _ARTWORK_CACHE[key] = img
  return img


def _pen_wireframe(layout):
  """Edges of the pen's two heads + the shaft axis, in the pen body frame.

  Drawn from the SAME numbers the solver uses (config/pen_tags.json), so a
  wireframe that does not sit on the real pen means the layout is wrong —
  not that the drawing is decorative.
  """
  pts, edges = [], []
  for box in ("A_top", "A_bottom"):
    half = layout.box_half_sizes[box]
    center = layout.box_centers[box]
    base = len(pts)
    for sx in (-1, 1):
      for sy in (-1, 1):
        for sz in (-1, 1):
          pts.append(center + np.array([sx, sy, sz]) * half)
    # 12 edges of a box over the 8 corners just appended
    for a, b in ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                 (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)):
      edges.append((base + a, base + b))
  base = len(pts)
  pts += [[0, 0, -layout.shaft_half_length], [0, 0, layout.shaft_half_length]]
  edges.append((base, base + 1))
  return np.asarray(pts, dtype=np.float64), edges


def _draw_pose(vis, pose, K, dist, layout, cv2):
  """Project the pen model and a body-frame axis triad through the solved pose."""
  rvec, tvec = pose.rvec.reshape(3, 1), pose.tvec.reshape(3, 1)
  colour = (0, 165, 255) if pose.coplanar else (0, 255, 0)

  pts, edges = _pen_wireframe(layout)
  uv, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
  uv = uv.reshape(-1, 2)
  if np.all(np.isfinite(uv)):
    for a, b in edges:
      cv2.line(vis, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), colour, 1)

  # Axis triad at the pen origin (shaft midpoint): X red, Y green, Z blue.
  # +Z points at A_top (local source-pattern 17 on the default 1x2 pen).
  L = 0.03
  axes = np.float64([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]])
  a, _ = cv2.projectPoints(axes, rvec, tvec, K, dist)
  a = a.reshape(-1, 2)
  if np.all(np.isfinite(a)):
    o = tuple(a[0].astype(int))
    for i, c in ((1, (0, 0, 255)), (2, (0, 255, 0)), (3, (255, 0, 0))):
      cv2.arrowedLine(vis, o, tuple(a[i].astype(int)), c, 2, tipLength=0.25)
    cv2.circle(vis, o, 4, (255, 255, 255), -1)


def _draw_world_tag(vis, transform, K, dist, cv2, *, fixed=False):
  """Draw the detected tag or the fixed calibration used by the publisher.

  The cached frame stays visible when the tag is occluded. A separate live
  outline makes camera/mount motion after calibration visible to the operator.
  """
  R, t = transform
  if not np.isfinite(R).all() or not np.isfinite(t).all() or t[2] <= 0:
    return
  rvec = cv2.Rodrigues(R)[0]
  h = WORLD_TAG_SIZE_M / 2
  corners = np.float64([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
  uv = cv2.projectPoints(corners, rvec, t, K, dist)[0].reshape(4, 2)
  if not np.isfinite(uv).all():
    return
  colour = (255, 255, 0) if fixed else (0, 255, 255)
  cv2.polylines(vis, [uv.astype(int)], True, colour, 2)
  label = "world FIXED" if fixed else f"base tag {WORLD_TAG_ID} LIVE"
  anchor = (int(uv[:, 0].min()),
            int(uv[:, 1].max() + 18 if fixed else uv[:, 1].min() - 8))
  cv2.putText(vis, label, anchor,
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
  cv2.putText(vis, label, anchor,
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
  # drawFrameAxes warns every frame if any endpoint leaves the image.
  # Omit the triad in that case; the tag outline/label remain visible.
  axes = np.float64([[0, 0, 0], [.04, 0, 0], [0, .04, 0], [0, 0, .04]])
  pixels = cv2.projectPoints(axes, rvec, t, K, dist)[0].reshape(-1, 2)
  height, width = vis.shape[:2]
  if (np.all((R @ axes.T).T[:, 2] + np.asarray(t).reshape(3)[2] > 0)
      and np.isfinite(pixels).all()
      and np.all(pixels >= 0)
      and np.all(pixels < [width, height])):
    cv2.drawFrameAxes(vis, K, dist, rvec, t, 0.04, 2)


def _synthetic_frames(K, dist, hz, layout=None):
  """Render the layout through a moving pose — no camera, no MuJoCo."""
  import cv2
  from lib.pen_pose import PenTagLayout
  lay = layout if layout is not None else PenTagLayout()
  W, H = 1280, 1024
  t0 = time.monotonic()
  while True:
    t = time.monotonic() - t0
    ax = np.array([math.cos(t * 0.7), math.sin(t * 0.5), 0.4])
    ax /= np.linalg.norm(ax)
    R, _ = cv2.Rodrigues(ax * (0.9 + 0.5 * math.sin(t)))
    tv = np.array([0.02 * math.cos(t), 0.02 * math.sin(t), 0.32])
    canvas = np.full((H, W), 255, np.uint8)
    draw = []
    for tid, C in lay.corners.items():
      Pc = (R @ C.T).T + tv
      n = np.cross(Pc[1] - Pc[0], Pc[2] - Pc[1])
      n /= np.linalg.norm(n)
      c = Pc.mean(0)
      # Canonical clockwise corners have an inward cross-product normal.
      if float(-n @ (-c / np.linalg.norm(c))) < 0.25:
        continue
      px, _ = cv2.projectPoints(C, cv2.Rodrigues(R)[0], tv, K, dist)
      draw.append((c[2], tid, px.reshape(4, 2).astype(np.float32)))
    for _, tid, uv in sorted(draw, reverse=True):
      # The custom codebook inverts data cells, leaving both border rings unchanged.
      src_img = _tag_artwork(tid, layout=lay)
      s_px = src_img.shape[0]
      cell = s_px // 8
      # corners_m describes the WHITE-BORDER OUTER EDGE: cells 1..7 of 8.
      lo, hi = cell, s_px - cell
      src = np.float32([[lo, lo], [hi, lo], [hi, hi], [lo, hi]])
      Hm = cv2.getPerspectiveTransform(src, uv)
      warp = cv2.warpPerspective(src_img, Hm, (W, H), borderValue=255)
      mask = cv2.warpPerspective(np.full(src_img.shape, 255, np.uint8), Hm,
                                 (W, H), borderValue=0)
      canvas[mask > 128] = warp[mask > 128]
    yield canvas, (R, tv)
    time.sleep(max(0.0, 1.0 / hz))


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--source", choices=("camera", "video", "synthetic"),
                  default="camera")
  ap.add_argument("--video", help="path/device for --source video")
  ap.add_argument("--port", type=int, default=PEN_PORT)
  ap.add_argument("--rate-hz", type=float, default=90.0)
  ap.add_argument("--preview", action="store_true")
  ap.add_argument("--exposure-us", type=float,
                  help="Camera exposure time in microseconds for this run only; defaults to camera.yaml")
  ap.add_argument("--tag-layout", help="pen tag JSON; default: config/pen_tags.json (1x2, 250 mm)")
  ap.add_argument("--min-faces", type=int, default=1,
                  help="refuse poses spanning fewer cube faces than this "
                       "(2 rejects the ill-conditioned single-face solve)")
  ap.add_argument("--world-frames", type=int, default=100,
                  help="frames averaged to fix the wrist-tag world frame")
  ap.add_argument("--no-world-tag", action="store_true",
                  help="publish in the CAMERA frame; the policy side expects "
                       "the wrist-tag frame, so this is for bench work only")
  args = ap.parse_args()
  if args.exposure_us is not None and (not np.isfinite(args.exposure_us) or args.exposure_us <= 0):
    ap.error("--exposure-us must be finite and positive")
  if args.world_frames < 1:
    ap.error("--world-frames must be positive")

  import cv2
  from lib.detection_roi import RoiSelector, frame_roi, save_selection
  from lib.pen_pose import PenPoseEstimator, PenTagLayout
  if args.preview:
    from lib.preview_ui import configure_qt_fonts
    configure_qt_fonts()

  cfg = load_camera_config()
  if args.exposure_us is not None:
    cfg['capture']['exposure_time'] = args.exposure_us
  K, dist = get_camera_matrix(cfg), get_dist_coeffs(cfg)
  est = PenPoseEstimator(K, dist, layout=PenTagLayout(args.tag_layout), min_tags=2)
  world = WorldFrame(args.world_frames)
  selector = RoiSelector()
  preview_initialized = False

  ctx = zmq.Context.instance()
  pub = ctx.socket(zmq.PUB)
  pub.bind(f"tcp://*:{args.port}")
  print(f"[pen observer] source={args.source} publishing on :{args.port}")
  print(f"[pen observer] {len(est.layout)} tags, {est.layout.dictionary_name}, "
        f"inverted={est.layout.inverted}, size={est.layout.tag_size * 1000:.2f} mm")
  print(f"[pen observer] pen length={est.layout.total_length * 1000:.1f} mm, "
        f"exposed shaft={est.layout.shaft_half_length * 2000:.1f} mm")
  print("[pen observer] raw PnP; every solved frame is published as measured, "
        "nothing is smoothed, gated or held")
  print("[pen observer] visibility checks enabled; planar candidates use continuity only for similar pixel errors")
  if args.no_world_tag:
    print("[pen observer] WARNING: publishing in the CAMERA frame "
          "(--no-world-tag); the policy expects the wrist-tag frame.")

  if args.source == "synthetic":
    frames = _synthetic_frames(K, dist, args.rate_hz, est.layout)
  elif args.source == "video":
    cap = cv2.VideoCapture(args.video if args.video else 0)
    if not cap.isOpened():
      print(f"[pen observer] cannot open video source {args.video!r}")
      return 2
    frames = _video_frames(cap)
  else:
    frames = _camera_frames(cfg)

  n = ok_n = 0
  total_n = total_ok = 0
  t_report = time.monotonic()
  # Observation throughput over the last second, including capture wait,
  # detection, and the previous frame's preview time.
  frame_times = deque()
  observer_fps = None
  phase_ms = dict(grab=0., base=0., detect=0., solve=0., output=0., preview=0.)
  consensus_at_report = 0
  fast_at_report = full_at_report = 0
  try:
    previous_frame_end = time.perf_counter()
    for gray, _truth in frames:
      frame_start = time.perf_counter()
      # The first frame includes camera initialization; exclude it from steady-state timing.
      if total_n:
        phase_ms['grab'] += (frame_start - previous_frame_end) * 1000.
      detection_roi = frame_roi(cfg, gray.shape)
      n += 1
      total_n += 1
      # Keep the preview responsive while the operator positions the wrist tag.
      world_seen = False
      wt = None
      # Stop detecting the base after calibration, including in previews and
      # recordings; press w to sample it again.
      if not args.no_world_tag and not world.fixed:
        wt = _detect_world_tag(gray, K, dist)
        world_seen = wt is not None
        if world_seen:
          world.add(*wt)

      base_end = time.perf_counter()
      phase_ms['base'] += (base_end - frame_start) * 1000.
      frame_fixed = args.no_world_tag or world.fixed
      pose = est.estimate(gray, roi=detection_roi) if frame_fixed else None
      solve_end = time.perf_counter()
      pen_ms = (solve_end - base_end) * 1000.
      detect_ms = min(pen_ms, getattr(est, 'last_detect_ms', 0.)) if frame_fixed else 0.
      phase_ms['detect'] += detect_ms
      phase_ms['solve'] += pen_ms - detect_ms
      if pose is not None:
        faces = {est.layout.face_normal[t] for t in pose.tag_ids}
        if len(faces) < args.min_faces:
          pose = None
      if pose is not None:
        ok_n += 1
        total_ok += 1
      # Every frame is published; a failed frame carries no pen. No hold and no
      # jump gate: a gate rejects exactly the fast motion a spinning pen
      # consists of. lib/zmq_bridge.PenReceiver keeps the last valid pose and
      # ages it.
      if pose is None:
        _publish(pub, None, frame_fixed)
      else:
        # Cache camera-frame PnP directly. Preview and publisher use the same
        # sample, with only the existing wrist-frame transform on the wire.
        if args.no_world_tag:
          pos, quat = pose.pos, pose.quat
        else:
          R, pos = world.to_world(matrix_from_quat(pose.quat), pose.pos)
          quat = quat_from_matrix(R)
        _publish(pub, (pos, quat), frame_fixed)

      preview_start = time.perf_counter()
      phase_ms['output'] += (preview_start - solve_end) * 1000.
      frame_now = time.monotonic()
      frame_times.append(frame_now)
      while len(frame_times) > 2 and frame_times[1] < frame_now - 1.0:
        frame_times.popleft()
      span = frame_now - frame_times[0]
      observer_fps = (len(frame_times) - 1) / span if span > 0 else None

      if args.preview:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if not args.no_world_tag:
          if world.fixed:
            R_wc = world.R_cw.T
            _draw_world_tag(vis, (R_wc, -R_wc @ world.t_cw), K, dist, cv2,
                            fixed=True)
          elif wt is not None:
            _draw_world_tag(vis, wt, K, dist, cv2)
        detections = getattr(est, 'last_detection', None) if frame_fixed else None
        if detections is None:
          # During world-frame sampling the pen pose is not solved yet; detect once for preview only.
          detections = est.detect(gray, roi=detection_roi)
        inlier_indices = set(getattr(est, 'last_inlier_indices', ())) if pose is not None else set()
        for index, (tid, quad) in enumerate(zip(*detections, strict=True)):
          accepted = index in inlier_indices
          colour = (0, 255, 0) if accepted else (0, 165, 255)
          cv2.polylines(vis, [quad.astype(int)], True, colour, 1)
          cv2.putText(vis, str(tid), tuple(quad[0].astype(int)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        if pose is not None:
          _draw_pose(vis, pose, K, dist, est.layout, cv2)
          faces = sorted({est.layout.face_normal[t] for t in pose.tag_ids})
          d_mm = float(np.linalg.norm(pose.pos)) * 1000.0
          cv2.putText(vis, f"{pose.num_tags} tags / {len(faces)} faces {faces}  "
                           f"reproj {pose.reproj_px:.2f}px  {d_mm:.0f}mm"
                           f"{'  PLANAR AMBIGUOUS' if pose.ambiguous else '  COPLANAR' if pose.coplanar else ''}",
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                      (0, 165, 255) if pose.coplanar else (0, 255, 0), 2)
        else:
          reason = getattr(est, "last_rejection", "")
          cv2.putText(vis, f"no pose: {reason}" if reason else "no pose", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        ignored = getattr(est, "last_ignored_ids", ())
        if pose is not None and ignored:
          cv2.putText(vis, f"ignored inconsistent tags: {list(ignored)}", (10, 86),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        if not args.no_world_tag:
          if world.fixed:
            txt = "world frame FIXED | base detection OFF (w = resample)"
            col = (0, 255, 0)
          else:
            got, need = world.samples, world.sample_frames
            txt = (f"sampling world frame {got}/{need} | "
                   f"{'base tag SEEN' if world_seen else 'show the wrist AprilTag id 0'}")
            col = (0, 255, 255) if world_seen else (0, 165, 255)
          cv2.putText(vis, txt, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        selector.draw(vis, detection_roi)
        cv2.putText(vis, 'RAW PnP', (10, vis.shape[0]-38), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255, 200, 0), 1)
        stats = f"Observer FPS: {observer_fps:.1f}" if observer_fps is not None else "Observer FPS: --"
        if args.source == 'camera':
          stats += f" | Exposure set: {cfg['capture']['exposure_time']/1000.:g} ms"
        cv2.putText(vis, stats, (10, vis.shape[0]-62), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (0, 0, 0), 3)
        cv2.putText(vis, stats, (10, vis.shape[0]-62), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (255, 255, 255), 1)
        cv2.imshow("pen observer", vis)
        if not preview_initialized:
          cv2.setMouseCallback("pen observer", selector.mouse)
          preview_initialized = True
        k = cv2.pollKey() & 0xFF
        if k == ord("q"):
          break
        if k == ord("w"):
          world.reset()
          print("[pen observer] resampling the world frame")
        if k == ord("s"):
          selector.begin()
        if k in (ord("c"), 27):
          selector.cancel()
        selection = None
        if selector.editing and not selector.dragging and k in (10, 13, 32):
          selection = selector.rect
        elif k == ord("f"):
          selection = (0, 0, gray.shape[1], gray.shape[0])
        if selection is not None:
          try:
            saved = save_selection(cfg, selection, gray.shape)
            if saved is not None:
              cfg["fast_roi"] = saved
              selector.cancel()
              est.reset()
              print(f"[pen observer] saved and applied fast_roi={saved}; world frame unchanged")
          except (OSError, ValueError) as exc:
            print(f"[pen observer] ROI not changed: {exc}")

      phase_ms['preview'] += (time.perf_counter() - preview_start) * 1000.
      now = time.monotonic()
      if now - t_report >= 2.0:
        print(f"[pen observer] {n} frames, {ok_n} poses "
              f"({n / (now - t_report):.1f} FPS), "
              f"({100.0 * ok_n / max(1, n):.0f}%), "
              f"reproj-rejects={est.rejected_reproj}, "
              f"visibility-rejects={getattr(est, 'rejected_visibility', 0)}, "
              f"appearance-rejects={getattr(est, 'rejected_appearance', 0)}, "
              f"contour-fallbacks={getattr(est, 'contour_fallbacks', 0)}, "
              f"coplanar={est.coplanar_solves}")
        consensus_total = getattr(est, 'consensus_calls', 0)
        fast_total = getattr(est, 'consensus_fast', 0)
        full_total = getattr(est, 'consensus_full', 0)
        phases = ' '.join(f'{name}={elapsed/max(n, 1):.1f}' for name, elapsed in phase_ms.items())
        print(f"[pen timing] mean ms/frame: {phases} | "
              f"consensus={consensus_total-consensus_at_report}/{n} frames "
              f"fast={fast_total-fast_at_report} full={full_total-full_at_report} "
              f"ignored-last={list(getattr(est, 'last_ignored_ids', ()))}", flush=True)
        consensus_at_report = consensus_total
        fast_at_report, full_at_report = fast_total, full_total
        phase_ms = dict.fromkeys(phase_ms, 0.)
        n = ok_n = 0
        t_report = now
      previous_frame_end = time.perf_counter()
  except KeyboardInterrupt:
    print("\n[pen observer] stopped")
  finally:
    # Always say what happened. A finite source (--source video) can end
    # before the 2 s report tick, and exiting in silence tells the operator
    # nothing about whether anything was ever detected.
    print(f"[pen observer] final: {total_n} frames, {total_ok} poses, "
          f"contour-fallbacks={getattr(est, 'contour_fallbacks', 0)}, "
          f"reproj-rejects={est.rejected_reproj}, "
          f"visibility-rejects={getattr(est, 'rejected_visibility', 0)}, "
          f"coplanar={est.coplanar_solves}")
    pub.close(linger=0)
    if args.preview:
      cv2.destroyAllWindows()
  return 0


def _publish(pub, pose, world_fixed: bool) -> None:
  """Wire schema — identical to what lib/zmq_bridge.PenReceiver parses."""
  if pose is None:
    msg = {"world_fixed": bool(world_fixed)}
  else:
    pos, q = pose
    msg = {"world_fixed": bool(world_fixed),
           "pen": {"position": {"x": float(pos[0]), "y": float(pos[1]),
                                "z": float(pos[2])},
                   # scipy xyzw on the wire; zmq_bridge converts to wxyz
                   "orientation": {"x": float(q[1]), "y": float(q[2]),
                                   "z": float(q[3]), "w": float(q[0])}}}
  try:
    pub.send_string(json.dumps(msg), flags=zmq.NOBLOCK)
  except zmq.Again:
    pass


def _video_frames(cap):
  import cv2
  while True:
    ok, frame = cap.read()
    if not ok:
      break
    yield (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
           if frame.ndim == 3 else frame), None


def _camera_frames(cfg):
  """Hikvision MVS grab loop — copied from the cube observer that ran on this rig.

  Kept line-for-line rather than rewritten, because every step here was paid
  for on real hardware:

  * ``MvCamera.MV_CC_Initialize()`` before enumerating, ``MV_CC_Finalize()``
    on the way out. Skipping the first one is how you get a process that
    opens the device, prints its settings, and then hangs with no image.
  * ``TriggerMode = OFF`` — a camera left in trigger mode waits forever for a
    trigger nobody sends.
  * ``PixelFormat = BayerGB8`` read with ``string_at`` and demosaiced. The
    buffer is a raw Bayer plane, not gray.
  * min-channel gray rather than ``COLOR_BGR2GRAY``: white stays 255 while
    any colour collapses to ~0, which is what separates dark tags cleanly.
  """
  mvs = os.environ.get("MVS_PYTHON_PATH", "/opt/MVS/Samples/64/Python")
  if not os.path.isdir(os.path.join(mvs, "MvImport")):
    raise RuntimeError(
        f"MvImport not found at {mvs}/MvImport. Install the Hikvision MVS SDK "
        "(https://www.hikrobotics.com) or set MVS_PYTHON_PATH. For bench work "
        "without a camera use --source video <file>.")
  sys.path.insert(0, mvs)
  from ctypes import POINTER, cast, string_at

  import cv2
  from lib.camera_config import setup_camera_capture, setup_camera_roi
  from MvImport.MvCameraControl_class import (  # type: ignore
    MV_CC_DEVICE_INFO,
    MV_CC_DEVICE_INFO_LIST,
    MV_FRAME_OUT,
    MV_GIGE_DEVICE,
    MV_TRIGGER_MODE_OFF,
    MV_USB_DEVICE,
    MV_ACCESS_Exclusive,
    MvCamera,
    PixelType_Gvsp_BayerGB8,
  )

  print("[pen observer] initializing camera...")
  MvCamera.MV_CC_Initialize()
  devs = MV_CC_DEVICE_INFO_LIST()
  MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, devs)
  if devs.nDeviceNum == 0:
    MvCamera.MV_CC_Finalize()
    raise RuntimeError("No camera found!")

  cam = MvCamera()
  dev = cast(devs.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
  cam.MV_CC_CreateHandle(dev)
  cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
  cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
  cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_BayerGB8)
  setup_camera_capture(cam, cfg)
  setup_camera_roi(cam, cfg)
  cam.MV_CC_StartGrabbing()
  print("[pen observer] camera ready!")

  out = MV_FRAME_OUT()
  misses = 0
  try:
    while True:
      ret = cam.MV_CC_GetImageBuffer(out, 100)
      if ret != 0:
        # The cube observer just `continue`d here. Say something instead: a
        # silent retry loop makes a dead camera look exactly like a camera
        # that is simply not pointed at the pen.
        misses += 1
        if misses in (10, 50) or misses % 200 == 0:
          print(f"[pen observer] no frame yet ({misses} timeouts, "
                f"last ret=0x{ret & 0xffffffff:x})", flush=True)
        continue
      if misses:
        print(f"[pen observer] frames flowing after {misses} timeouts", flush=True)
        misses = 0
      nH = out.stFrameInfo.nHeight
      nW = out.stFrameInfo.nWidth
      data = string_at(out.pBufAddr, out.stFrameInfo.nFrameLen)
      bayer = np.frombuffer(data, dtype=np.uint8).reshape(nH, nW)
      bgr = cv2.cvtColor(bayer, cv2.COLOR_BayerGB2BGR)
      gray = np.minimum(np.minimum(bgr[:, :, 0], bgr[:, :, 1]), bgr[:, :, 2])
      cam.MV_CC_FreeImageBuffer(out)
      yield gray.copy(), None
  finally:
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    MvCamera.MV_CC_Finalize()


if __name__ == "__main__":
  raise SystemExit(main())
