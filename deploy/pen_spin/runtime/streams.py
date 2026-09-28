# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Real-time inputs: the camera-fed pen pose.

The camera observer publishes over ZMQ (lib.zmq_bridge); this module lifts the
pose into the palm frame the policy observes (lib.tag_frame), in one place so
the viewer cannot drift from what the policy sees.

There is no staleness policy: the loop consumes the newest sample whatever its
age, and the age travels with it as telemetry.
"""

from __future__ import annotations

import numpy as np
from lib.math_utils import quat_apply
from lib.tag_frame import pen_in_palm
from lib.zmq_bridge import PEN_PORT, PenReceiver

# Body frames an observer may publish the pen in. Every layout under
# config/pen_tags*.json is source-z (+Z shaft); policies and clips are +X.
PEN_SOURCE_FRAMES = ("tracking-x", "source-z")


class ObjectPose:
    """Camera ZMQ -> palm-local pen pose.

    The camera observer publishes in the wrist-tag frame and the tag is bolted
    to the wrist, so tag->palm is a constant and ONE stream is enough.
    """

    def __init__(self, receiver: PenReceiver, *, source_frame: str,
                 axis_offset_m: float = 0.):
        # No default: a silently wrong frame is a 90 deg pen. The one default
        # lives on --pen-source-frame (runtime/cli.py).
        if source_frame not in PEN_SOURCE_FRAMES:
            raise ValueError(f"unknown pen frame: {source_frame}")
        self.source_frame = source_frame
        self.axis_offset_m = float(axis_offset_m)
        if not np.isfinite(self.axis_offset_m):
            raise ValueError("pen axis offset must be finite")
        self.receiver = receiver

    def snapshot(self) -> dict:
        pos, quat, age = self.receiver.latest()
        return {"pen_pos": pos, "pen_quat": quat, "pen_age_s": age}

    def has_pose(self, snap: dict) -> bool:
        """True once ANY valid pen pose has arrived. Age is not a veto — it is
        reported alongside so callers can display it."""
        del snap  # the receiver owns the flag; the snapshot is for the values
        return self.receiver.seen

    def local(self, snap: dict) -> tuple[np.ndarray, np.ndarray]:
        """Pen pose in the palm frame — exactly what obs consumes."""
        pos, quat = pen_in_palm(snap["pen_pos"], snap["pen_quat"])
        # Source +Z and tracking +X both point toward A_top; apply the offset
        # before the basis conversion.
        if self.axis_offset_m:
            axis = ([0., 0., self.axis_offset_m] if self.source_frame == "source-z"
                    else [self.axis_offset_m, 0., 0.])
            pos = pos + quat_apply(quat, np.asarray(axis))
        if self.source_frame == "source-z":
            from lib.pen_frame import source_to_tracking
            quat = source_to_tracking(quat)
        return pos, quat


def open_camera(port: int = PEN_PORT, host: str = "localhost") -> PenReceiver:
    return PenReceiver(port=port, host=host)
