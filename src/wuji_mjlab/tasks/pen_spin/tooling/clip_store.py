# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Validated access to the task-local pen-spin clip catalog."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_INDEX_SCHEMA = "wuji_pen_spin_clips:1"
_DEFAULT_CLIPS_ROOT = Path(__file__).resolve().parents[1] / "data/rawdata/clips"
_INDEX_FIELDS = {
  "schema",
  "group_id",
  "group_name",
  "frame_dt",
  "clip_count",
  "clips",
}
_CLIP_FIELDS = {"clip_id", "file", "sha256"}


@dataclass(frozen=True)
class TrackingClip:
  """One trajectory in the task-local clip catalog."""

  clip_id: str
  group_id: str
  npz_path: Path


@dataclass(frozen=True)
class ClipGroup:
  """One named group from ``rawdata/clips``."""

  group_id: str
  group_name: str
  frame_dt: float
  clips: tuple[TrackingClip, ...]


@dataclass(frozen=True)
class ResolvedClipSet:
  """The recipe-selected trajectories consumed by the tracking command."""

  frame_dt: float
  clips: tuple[TrackingClip, ...]


def discover_clip_groups(
  clips_root: str | Path | None = None,
  *,
  verify_hashes: bool = False,
) -> tuple[ClipGroup, ...]:
  root = (_DEFAULT_CLIPS_ROOT if clips_root is None else Path(clips_root)).expanduser()
  if root.is_symlink() or not root.is_dir():
    raise ValueError(f"clip root must be a real directory: {root}")

  groups = tuple(
    _read_group(path, verify_hashes=verify_hashes)
    for path in _sorted_directories(root, "clip group")
  )
  if not groups:
    raise ValueError(f"clip root contains no groups: {root}")
  return groups


def _read_group(group_root: Path, *, verify_hashes: bool) -> ClipGroup:
  payload = _read_json(group_root / "index.json", "clip group index")
  require_fields(payload, _INDEX_FIELDS, set(), group_root / "index.json")

  group_id = require_component(payload["group_id"], "group ID")
  if group_id != group_root.name:
    raise ValueError(
      f"clip group ID {group_id!r} does not match directory {group_root.name!r}"
    )
  group_name = payload["group_name"]
  if not isinstance(group_name, str) or not group_name:
    raise ValueError(f"clip group name must be a non-empty string: {group_root}")
  frame_dt = payload["frame_dt"]
  if (
    isinstance(frame_dt, bool)
    or not isinstance(frame_dt, int | float)
    or not math.isfinite(frame_dt)
    or frame_dt <= 0.0
  ):
    raise ValueError(f"clip group frame_dt must be finite and positive: {group_root}")

  entries = payload["clips"]
  if not isinstance(entries, list) or not entries:
    raise ValueError(f"clip group must contain a non-empty clips array: {group_root}")
  if payload["clip_count"] != len(entries):
    raise ValueError(f"clip_count does not match clips array: {group_root}")

  clips: list[TrackingClip] = []
  indexed_files: set[str] = set()
  for index, entry in enumerate(entries):
    if not isinstance(entry, dict):
      raise ValueError(f"clip entry {index} must be an object: {group_root}")
    require_fields(entry, _CLIP_FIELDS, set(), f"{group_root}: clip {index}")
    local_id = require_component(entry["clip_id"], "clip ID")
    filename = require_component(entry["file"], "clip filename")
    if filename != f"{local_id}.npz":
      raise ValueError(f"clip file must equal <clip_id>.npz: {group_root / filename}")
    expected_hash = entry["sha256"]
    if (
      not isinstance(expected_hash, str)
      or len(expected_hash) != 64
      or any(char not in "0123456789abcdef" for char in expected_hash)
    ):
      raise ValueError(
        f"clip sha256 must be lowercase hexadecimal: {group_root / filename}"
      )
    path = _require_regular_file(group_root / filename, "clip file")
    if verify_hashes and _sha256(path) != expected_hash:
      raise ValueError(f"clip SHA-256 mismatch: {path}")
    if filename in indexed_files:
      raise ValueError(f"duplicate clip file in index: {group_root / filename}")
    indexed_files.add(filename)
    clips.append(
      TrackingClip(
        clip_id=f"{group_id}/{local_id}",
        group_id=group_id,
        npz_path=path,
      )
    )

  actual_files = {
    path.name for path in _sorted_regular_files(group_root, suffix=".npz")
  }
  if indexed_files != actual_files:
    missing = sorted(indexed_files - actual_files)
    unindexed = sorted(actual_files - indexed_files)
    raise ValueError(
      f"clip index/files mismatch in {group_root}: missing={missing}, unindexed={unindexed}"
    )
  return ClipGroup(
    group_id=group_id,
    group_name=group_name,
    frame_dt=float(frame_dt),
    clips=tuple(clips),
  )


def parse_catalog_clip_id(value: object, description: str) -> str:
  if not isinstance(value, str):
    raise ValueError(f"{description} must use group/clip form")
  path = PurePosixPath(value)
  if path.is_absolute() or len(path.parts) != 2 or "\\" in value:
    raise ValueError(f"{description} must use group/clip form")
  group_id, clip_id = (require_component(part, description) for part in path.parts)
  if clip_id.endswith(".npz"):
    raise ValueError(f"{description} must omit the .npz suffix")
  return f"{group_id}/{clip_id}"


def require_fields(
  payload: dict,
  required: set[str],
  optional: set[str],
  description: object,
) -> None:
  unknown = set(payload) - required - optional
  missing = required - set(payload)
  if unknown:
    raise ValueError(f"{description} has unknown fields: {sorted(unknown)}")
  if missing:
    raise ValueError(f"{description} is missing fields: {sorted(missing)}")


def require_component(value: object, description: str) -> str:
  if (
    not isinstance(value, str)
    or not value
    or value in {".", ".."}
    or "/" in value
    or "\\" in value
    or "\x00" in value
  ):
    raise ValueError(f"{description} must be a non-empty path-safe string")
  return value


def _sorted_directories(root: Path, description: str) -> tuple[Path, ...]:
  entries: list[Path] = []
  for path in root.iterdir():
    if path.is_symlink():
      raise ValueError(f"{description} cannot be a symlink: {path}")
    if path.is_dir():
      entries.append(path)
  return tuple(sorted(entries, key=lambda path: path.name.encode("utf-8")))


def _sorted_regular_files(root: Path, *, suffix: str) -> tuple[Path, ...]:
  files: list[Path] = []
  for path in root.iterdir():
    if path.is_symlink():
      raise ValueError(f"clip catalog content cannot be a symlink: {path}")
    if path.is_file() and path.suffix == suffix:
      files.append(path)
  return tuple(sorted(files, key=lambda path: path.name.encode("utf-8")))


def _require_regular_file(path: Path, description: str) -> Path:
  if path.is_symlink() or not path.is_file():
    raise ValueError(f"{description} is missing or not a regular file: {path}")
  return path.resolve()


def _read_json(path: Path, description: str) -> dict:
  path = _require_regular_file(path, description)
  try:
    payload = json.loads(path.read_text())
  except (OSError, json.JSONDecodeError) as exc:
    raise ValueError(f"invalid {description}: {path}") from exc
  if not isinstance(payload, dict):
    raise ValueError(f"{description} must contain a JSON object: {path}")
  return payload


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()
