#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""ZMQ Communication Bridge for Wuji Hand Deploy.

Provides unified ZMQ pub/sub classes for communication between processes:
- GoalPublisher: Publish goal orientation (toreal_viewer.py -> play_real.py)
- GoalReceiver: Receive goal orientation (play_real.py <- toreal_viewer.py)
- CubeReceiver: Receive cube pose (play_real.py <- cube_world_observer.py)

Example:
    >>> from zmq_bridge import GoalPublisher, CubeReceiver
    >>> publisher = GoalPublisher(port=5556)
    >>> publisher.publish(goal_quat)
    >>>
    >>> receiver = CubeReceiver(port=5555)
    >>> quat, pos, valid = receiver.get_pose()
"""
from __future__ import annotations

import json
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import zmq

if TYPE_CHECKING:
    pass

DEFAULT_CUBE_PORT: int = 5555
"""Default ZMQ port for cube pose (cube_world_observer.py)."""

DEFAULT_GOAL_PORT: int = 5556
"""Default ZMQ port for goal orientation (toreal_viewer.py)."""

_RECEIVER_POLL_INTERVAL: float = 0.001
"""Polling interval for receiver threads in seconds."""


class GoalPublisher:
    """Publish goal orientation via ZMQ.

    Publishes goal quaternion in MuJoCo convention (w, x, y, z) to subscribers.
    Used by toreal_viewer.py to send goals to play_real.py.

    Attributes:
        port: ZMQ port number.

    Example:
        >>> publisher = GoalPublisher(port=5556)
        >>> goal_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
        >>> publisher.publish(goal_quat)
        >>> publisher.close()
    """

    def __init__(self, port: int = DEFAULT_GOAL_PORT) -> None:
        """Initialize goal publisher.

        Args:
            port: ZMQ port to bind to.
        """
        self.port = port
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(f"tcp://*:{port}")
        print(f"GoalPublisher started on port {port}")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"GoalPublisher(port={self.port})"

    def publish(self, goal_quat: np.ndarray) -> None:
        """Publish goal quaternion.

        Args:
            goal_quat: Goal orientation quaternion (w, x, y, z).
        """
        msg = {
            'timestamp': time.time(),
            'goal': {
                'orientation': {
                    'w': float(goal_quat[0]),
                    'x': float(goal_quat[1]),
                    'y': float(goal_quat[2]),
                    'z': float(goal_quat[3]),
                }
            }
        }
        self._socket.send_string(json.dumps(msg))

    def close(self) -> None:
        """Close the publisher and release resources."""
        self._socket.close()
        self._context.term()


class GoalReceiver:
    """Receive goal orientation from toreal_viewer.py via ZMQ.

    Runs a background thread to continuously receive goal updates.
    Returns the latest received goal quaternion in MuJoCo convention (w, x, y, z).

    Attributes:
        port: ZMQ port number.
        goal_count: Number of goals received.

    Example:
        >>> receiver = GoalReceiver(port=5556)
        >>> goal_quat = receiver.get_goal()  # Returns (w, x, y, z)
        >>> receiver.close()
    """

    def __init__(self, port: int = DEFAULT_GOAL_PORT, host: str = "localhost") -> None:
        """Initialize goal receiver.

        Args:
            port: ZMQ port to connect to.
            host: Host address to connect to.
        """
        self.port = port
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{host}:{port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.CONFLATE, 1)

        # State
        self._latest_goal = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
        self.goal_count = 0
        self._running = True
        self._lock = threading.Lock()

        # Start receiver thread
        self._thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._thread.start()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"GoalReceiver(port={self.port}, count={self.goal_count})"

    def get_goal(self) -> np.ndarray:
        """Get the latest goal quaternion.

        Returns:
            Goal quaternion (w, x, y, z).
        """
        with self._lock:
            return self._latest_goal.copy()

    def latest(self) -> np.ndarray:
        """Return latest goal quaternion (wxyz). Identity if no goal received yet.

        Mirrors the ABI consumed by RealHandEnv.step() and play_real so
        downstream consumers no longer need a per-script adapter.

        Returns:
            Goal quaternion (w, x, y, z), identity ``[1, 0, 0, 0]`` until a
            goal arrives over ZMQ.
        """
        with self._lock:
            return self._latest_goal.copy()

    def close(self) -> None:
        """Close the receiver and release resources."""
        self._running = False
        self._socket.close()
        self._context.term()

    def _receiver_loop(self) -> None:
        """Background thread to receive goals."""
        while self._running:
            try:
                data = self._socket.recv(zmq.NOBLOCK)
                msg = json.loads(data.decode('utf-8'))

                if 'goal' in msg:
                    q = msg['goal']['orientation']
                    with self._lock:
                        self._latest_goal = np.array([q['w'], q['x'], q['y'], q['z']])
                        self.goal_count += 1

            except zmq.Again:
                pass
            except (json.JSONDecodeError, KeyError):
                pass
            time.sleep(_RECEIVER_POLL_INTERVAL)


class CubeReceiver:
    """Receive cube pose from cube_world_observer.py via ZMQ.

    Runs a background thread to continuously receive cube pose updates.
    Returns quaternion in MuJoCo convention (w, x, y, z) and position.

    Attributes:
        port: ZMQ port number.
        cube_count: Number of cube poses received.
        world_fixed: Whether the world frame is calibrated.

    Example:
        >>> receiver = CubeReceiver(port=5555)
        >>> quat, pos, valid = receiver.get_pose()
        >>> if valid:
        ...     print(f"Cube at {pos}, orientation {quat}")
        >>> receiver.close()
    """

    def __init__(self, port: int = DEFAULT_CUBE_PORT, host: str = "localhost") -> None:
        """Initialize cube receiver.

        Args:
            port: ZMQ port to connect to.
            host: Host address to connect to.
        """
        self.port = port
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{host}:{port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.CONFLATE, 1)

        # State (w, x, y, z format for MuJoCo)
        self._latest_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._latest_pos = np.zeros(3)
        # Cached last *valid* sample (world_fixed=True). Drives .latest()
        # so consumers keep seeing the last good pose if the publisher
        # temporarily drops world_fixed back to False. Defaults match the
        # no-ZMQ fallback in RealHandEnv.step() — single fallback contract.
        self._cached_valid_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._cached_valid_pos = np.zeros(3)
        self.cube_count = 0
        self.world_fixed = False
        # Latest cube_size announced by the publisher (meters, full edge length).
        # None until the first message containing the field arrives. Lets
        # consumers (e.g. play_real.py) auto-rescale the sim cube to match the
        # vision-side config without a duplicated CLI knob.
        self.cube_size: float | None = None
        self._running = True
        self._lock = threading.Lock()

        # Start receiver thread
        self._thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._thread.start()
        print(f"CubeReceiver started on port {port}")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"CubeReceiver(port={self.port}, count={self.cube_count}, world_fixed={self.world_fixed})"

    def get_pose(self) -> tuple[np.ndarray, np.ndarray, bool]:
        """Get the latest cube pose.

        Returns:
            Tuple of (quaternion, position, valid) where:
            - quaternion: Cube orientation (w, x, y, z)
            - position: Cube position (x, y, z)
            - valid: Whether the pose is valid (world frame calibrated)
        """
        with self._lock:
            valid = self.cube_count > 0 and self.world_fixed
            return self._latest_quat.copy(), self._latest_pos.copy(), valid

    def latest(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (cube_pos, cube_quat_wxyz). Caches last valid sample.

        Matches the ABI consumed by RealHandEnv.step() and play_real so
        downstream consumers no longer need a per-script adapter. The cache
        only updates when ``world_fixed=True`` (i.e. a valid sample is in
        flight); before any valid sample arrives (or while world_fixed has
        dropped back to False) the previously cached pose — or the
        (zeros, identity) defaults that match the no-ZMQ fallback in
        RealHandEnv — is returned.

        Returns:
            Tuple of (cube_pos, cube_quat_wxyz), each ``(3,)`` / ``(4,)``.
        """
        with self._lock:
            if self.cube_count > 0 and self.world_fixed:
                self._cached_valid_pos = self._latest_pos.copy()
                self._cached_valid_quat = self._latest_quat.copy()
            return self._cached_valid_pos.copy(), self._cached_valid_quat.copy()

    def close(self) -> None:
        """Close the receiver and release resources."""
        self._running = False
        self._socket.close()
        self._context.term()

    def _receiver_loop(self) -> None:
        """Background thread to receive cube pose."""
        while self._running:
            try:
                data = self._socket.recv(zmq.NOBLOCK)
                msg = json.loads(data.decode('utf-8'))

                if 'cube1' in msg:
                    cube = msg['cube1']
                    q = cube['orientation']  # (x, y, z, w) from scipy
                    p = cube['position']
                    with self._lock:
                        # Convert to (w, x, y, z) for MuJoCo
                        self._latest_quat = np.array([q['w'], q['x'], q['y'], q['z']])
                        self._latest_pos = np.array([p['x'], p['y'], p['z']])
                        # world_fixed means world frame is calibrated
                        self.world_fixed = msg.get('world_fixed', False)
                        # Optional: cube_size (publisher started carrying it after
                        # the cube_tags refactor; keep backward-compat)
                        cs = msg.get('cube_size')
                        if cs is not None:
                            self.cube_size = float(cs)
                        self.cube_count += 1

            except zmq.Again:
                pass
            except (json.JSONDecodeError, KeyError):
                pass
            time.sleep(_RECEIVER_POLL_INTERVAL)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ZMQ Bridge Test')
    parser.add_argument('--mode', choices=['pub', 'sub'], required=True)
    parser.add_argument('--type', choices=['goal', 'cube'], default='goal')
    args = parser.parse_args()

    if args.mode == 'pub' and args.type == 'goal':
        pub = GoalPublisher()
        try:
            i = 0
            while True:
                quat = np.array([1.0, 0.0, 0.0, 0.0])
                pub.publish(quat)
                if i % 20 == 0:
                    print(f"Published goal {i}")
                i += 1
                time.sleep(0.05)
        except KeyboardInterrupt:
            pub.close()

    elif args.mode == 'sub' and args.type == 'goal':
        recv = GoalReceiver()
        try:
            while True:
                goal = recv.get_goal()
                print(f"Goal: {goal}, count={recv.goal_count}")
                time.sleep(0.5)
        except KeyboardInterrupt:
            recv.close()

    elif args.mode == 'sub' and args.type == 'cube':
        recv = CubeReceiver()
        try:
            while True:
                quat, pos, valid = recv.get_pose()
                print(f"Cube: pos={pos}, quat={quat}, valid={valid}")
                time.sleep(0.5)
        except KeyboardInterrupt:
            recv.close()
