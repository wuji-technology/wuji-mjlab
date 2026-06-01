#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Joint Data Shared Memory for Wuji Hand Deploy.

Provides shared memory communication for real-time joint data visualization.

Data layout per timestep:
    - time: 1 float (seconds since start)
    - action: 20 floats (policy output)
    - actual: 20 floats (actual joint positions)
    - target: 20 floats (motor target positions)

Example:
    >>> # Writer
    >>> from joint_data import JointDataWriter
    >>> writer = JointDataWriter("/tmp/toreal_joint_data.bin")
    >>> writer.write(action, actual, target)
    >>> writer.close()
    >>>
    >>> # Reader
    >>> from joint_data import JointDataReader
    >>> reader = JointDataReader("/tmp/toreal_joint_data.bin")
    >>> if reader.connect():
    ...     times, actions, actuals, targets = reader.read()
"""
from __future__ import annotations

import mmap
import struct
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

DEFAULT_MAX_HISTORY: int = 500
"""Default number of timesteps to store in circular buffer."""

DEFAULT_N_JOINTS: int = 20
"""Default number of joints (5 fingers × 4 DOF)."""

JOINT_DATA_FILE_REAL: str = "/tmp/toreal_joint_data.bin"
"""Default data file path for real hand control."""


class JointDataWriter:
    """Write joint data to shared memory file for real-time plotting.

    Creates a memory-mapped file that can be read by JointDataReader
    for real-time visualization of joint trajectories.

    The data is stored in a circular buffer with a fixed maximum history.

    Attributes:
        data_file: Path to the shared memory file.
        n_joints: Number of joints.
        max_history: Maximum number of timesteps stored.

    Example:
        >>> writer = JointDataWriter("/tmp/joint_data.bin")
        >>> action = np.zeros(20)
        >>> actual = np.zeros(20)
        >>> target = np.zeros(20)
        >>> writer.write(action, actual, target)
        >>> writer.close()
    """

    def __init__(
        self,
        data_file: str,
        n_joints: int = DEFAULT_N_JOINTS,
        max_history: int = DEFAULT_MAX_HISTORY,
    ) -> None:
        """Initialize joint data writer.

        Args:
            data_file: Path to create the shared memory file.
            n_joints: Number of joints to store.
            max_history: Maximum number of timesteps in circular buffer.
        """
        self.data_file = data_file
        self.n_joints = n_joints
        self.max_history = max_history

        # Data layout: time(1) + action(n) + actual(n) + target(n) = 1 + 3n floats
        self._floats_per_step = 1 + 3 * n_joints
        self._bytes_per_step = self._floats_per_step * 4
        self._header_size = 8  # write_idx(4) + total_steps(4)
        self._total_size = self._header_size + max_history * self._bytes_per_step

        # Create shared file
        with open(data_file, 'wb') as f:
            f.write(b'\x00' * self._total_size)

        self._file = open(data_file, 'r+b')
        self._mmap = mmap.mmap(self._file.fileno(), self._total_size)
        self._write_idx = 0
        self._total_steps = 0
        self._start_time = time.time()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"JointDataWriter({self.data_file!r}, steps={self._total_steps})"

    def write(self, action: np.ndarray, actual: np.ndarray, target: np.ndarray) -> None:
        """Write one timestep of joint data.

        Args:
            action: Policy action output (n_joints,).
            actual: Actual joint positions (n_joints,).
            target: Motor target positions (n_joints,).
        """
        t = time.time() - self._start_time
        data = struct.pack(
            f'{self._floats_per_step}f',
            t,
            *action[:self.n_joints],
            *actual[:self.n_joints],
            *target[:self.n_joints],
        )

        offset = self._header_size + self._write_idx * self._bytes_per_step
        self._mmap[offset:offset + self._bytes_per_step] = data

        self._write_idx = (self._write_idx + 1) % self.max_history
        self._total_steps += 1
        self._mmap[0:8] = struct.pack('II', self._write_idx, self._total_steps)

    def reset(self) -> None:
        """Reset buffer to empty state."""
        self._write_idx = 0
        self._total_steps = 0
        self._start_time = time.time()
        self._mmap[0:8] = struct.pack('II', 0, 0)

    def close(self) -> None:
        """Close the writer and release resources."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None


class JointDataReader:
    """Read joint data from shared memory file.

    Connects to a memory-mapped file created by JointDataWriter
    and reads the circular buffer of joint data.

    Attributes:
        data_file: Path to the shared memory file.
        n_joints: Number of joints.
        max_history: Maximum number of timesteps stored.

    Example:
        >>> reader = JointDataReader("/tmp/joint_data.bin")
        >>> if reader.connect():
        ...     times, actions, actuals, targets = reader.read()
        ...     print(f"Got {len(times)} samples")
        >>> reader.close()
    """

    def __init__(
        self,
        data_file: str,
        n_joints: int = DEFAULT_N_JOINTS,
        max_history: int = DEFAULT_MAX_HISTORY,
    ) -> None:
        """Initialize joint data reader.

        Args:
            data_file: Path to the shared memory file.
            n_joints: Number of joints expected.
            max_history: Maximum number of timesteps in buffer.
        """
        self.data_file = data_file
        self.n_joints = n_joints
        self.max_history = max_history

        # Data layout must match JointDataWriter
        self._floats_per_step = 1 + 3 * n_joints
        self._bytes_per_step = self._floats_per_step * 4
        self._header_size = 8
        self._total_size = self._header_size + max_history * self._bytes_per_step

        self._file = None
        self._mmap = None

    def __repr__(self) -> str:
        """Return string representation."""
        connected = self._mmap is not None
        return f"JointDataReader({self.data_file!r}, connected={connected})"

    def connect(self) -> bool:
        """Try to connect to shared memory file.

        Returns:
            True if connection successful, False otherwise.
        """
        if not Path(self.data_file).exists():
            return False
        try:
            self._file = open(self.data_file, 'rb')
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def read(self) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Read all available data from buffer.

        Returns:
            Tuple of (times, actions, actuals, targets) arrays,
            or (None, None, None, None) if no data available.
        """
        if self._mmap is None:
            return None, None, None, None

        # Read header
        header = self._mmap[0:8]
        write_idx, total_steps = struct.unpack('II', header)

        if total_steps == 0:
            return None, None, None, None

        # Calculate how many samples to read
        n_samples = min(total_steps, self.max_history)

        # Read data in order (oldest first)
        if total_steps >= self.max_history:
            # Buffer is full, start from write_idx (oldest)
            start_idx = write_idx
        else:
            # Buffer not full yet, start from 0
            start_idx = 0

        times = np.zeros(n_samples)
        actions = np.zeros((n_samples, self.n_joints))
        actuals = np.zeros((n_samples, self.n_joints))
        targets = np.zeros((n_samples, self.n_joints))

        for i in range(n_samples):
            idx = (start_idx + i) % self.max_history
            offset = self._header_size + idx * self._bytes_per_step
            data = struct.unpack(
                f'{self._floats_per_step}f',
                self._mmap[offset:offset + self._bytes_per_step]
            )
            times[i] = data[0]
            actions[i] = data[1:1 + self.n_joints]
            actuals[i] = data[1 + self.n_joints:1 + 2 * self.n_joints]
            targets[i] = data[1 + 2 * self.n_joints:]

        return times, actions, actuals, targets

    def close(self) -> None:
        """Close the reader and release resources."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Joint Data Test')
    parser.add_argument('--mode', choices=['write', 'read'], required=True)
    parser.add_argument('--file', default=JOINT_DATA_FILE_REAL)
    args = parser.parse_args()

    if args.mode == 'write':
        writer = JointDataWriter(args.file)
        print(f"Writing to {args.file}")
        try:
            i = 0
            while True:
                action = np.sin(np.arange(20) * 0.1 + i * 0.1) * 0.5
                actual = action * 0.9
                target = action
                writer.write(action, actual, target)
                if i % 20 == 0:
                    print(f"Written step {i}")
                i += 1
                time.sleep(0.05)
        except KeyboardInterrupt:
            writer.close()
            print("Done")

    elif args.mode == 'read':
        reader = JointDataReader(args.file)
        if not reader.connect():
            print(f"Could not connect to {args.file}")
        else:
            print(f"Connected to {args.file}")
            try:
                while True:
                    times, actions, actuals, targets = reader.read()
                    if times is not None:
                        print(f"Read {len(times)} samples, latest time={times[-1]:.2f}s")
                    time.sleep(0.5)
            except KeyboardInterrupt:
                reader.close()
                print("Done")
