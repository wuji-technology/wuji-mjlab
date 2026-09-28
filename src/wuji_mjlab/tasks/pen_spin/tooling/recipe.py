# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Resolve pen-spin recipes against the task-local clip catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from wuji_mjlab.tasks.pen_spin.tooling.clip_store import (
  ClipGroup,
  ResolvedClipSet,
  TrackingClip,
  discover_clip_groups,
  parse_catalog_clip_id,
  require_component,
  require_fields,
)

_SCHEMA = "wuji_pen_spin_recipe:1"
_RECIPE_FIELDS = {"schema"}
_RECIPE_OPTIONAL_FIELDS = {"groups", "clips", "exclude_clips"}


@dataclass(frozen=True)
class ResolvedRecipe:
  recipe_path: Path
  clips_root: Path
  clip_sets: tuple[ResolvedClipSet, ...]


def resolve_recipe(
  recipe_path: str | Path,
  *,
  clips_root: str | Path | None = None,
  verify_clip_hashes: bool = False,
) -> ResolvedRecipe:
  path = Path(recipe_path)
  payload = _load_recipe(path)
  groups_filter, clips_filter, excluded_filter = _parse_recipe(path, payload)
  inventory = discover_clip_groups(clips_root, verify_hashes=verify_clip_hashes)

  available_groups = {group.group_id for group in inventory}
  available_clips = {clip.clip_id for group in inventory for clip in group.clips}
  _require_available(path, "group", groups_filter, available_groups)
  _require_available(path, "clip", clips_filter, available_clips)
  _require_available(path, "excluded clip", excluded_filter, available_clips)

  selected = tuple(
    clip
    for group in inventory
    for clip in group.clips
    if (groups_filter is None or group.group_id in groups_filter)
    and (clips_filter is None or clip.clip_id in clips_filter)
    and (excluded_filter is None or clip.clip_id not in excluded_filter)
  )
  if not selected:
    raise ValueError(f"{path}: recipe resolved to zero clips")
  frame_dt = _common_frame_dt(inventory, selected)
  resolved_root = selected[0].npz_path.parents[1]
  return ResolvedRecipe(
    recipe_path=path.resolve(),
    clips_root=resolved_root,
    clip_sets=(ResolvedClipSet(frame_dt=frame_dt, clips=selected),),
  )


def _load_recipe(path: Path) -> dict:
  if not path.is_file():
    raise FileNotFoundError(f"recipe not found: {path}")
  try:
    payload = json.loads(path.read_text())
  except (OSError, json.JSONDecodeError) as exc:
    raise ValueError(f"invalid recipe: {path}") from exc
  if not isinstance(payload, dict):
    raise ValueError(f"{path}: recipe must contain a JSON object")
  return payload


def _parse_recipe(
  path: Path,
  payload: dict,
) -> tuple[set[str] | None, set[str] | None, set[str] | None]:
  require_fields(payload, _RECIPE_FIELDS, _RECIPE_OPTIONAL_FIELDS, f"{path}: recipe")
  if payload["schema"] != _SCHEMA:
    raise ValueError(f"{path}: schema must be {_SCHEMA!r}")

  groups = _parse_groups(path, payload.get("groups"))
  clips = _parse_clips(path, "clips", payload.get("clips"))
  excluded = _parse_clips(path, "exclude_clips", payload.get("exclude_clips"))
  if groups is None and clips is None:
    raise ValueError(f"{path}: recipe must select groups or clips")
  if clips is not None and excluded is not None:
    raise ValueError(f"{path}: clips and exclude_clips are mutually exclusive")
  return groups, clips, excluded


def _parse_groups(path: Path, value: object) -> set[str] | None:
  if value is None:
    return None
  if not isinstance(value, list) or not value:
    raise ValueError(f"{path}: groups must be a non-empty array")
  parsed = tuple(require_component(item, f"{path}: group") for item in value)
  if len(set(parsed)) != len(parsed):
    raise ValueError(f"{path}: groups contains duplicate IDs")
  return set(parsed)


def _parse_clips(path: Path, field: str, value: object) -> set[str] | None:
  if value is None:
    return None
  if not isinstance(value, list) or not value:
    raise ValueError(f"{path}: {field} must be a non-empty array")
  parsed = tuple(parse_catalog_clip_id(item, f"{path}: {field}") for item in value)
  if len(set(parsed)) != len(parsed):
    raise ValueError(f"{path}: {field} contains duplicate IDs")
  return set(parsed)


def _require_available(
  path: Path,
  description: str,
  requested: set[str] | None,
  available: set[str],
) -> None:
  if requested is None:
    return
  missing = requested - available
  if missing:
    raise ValueError(f"{path}: unknown {description} IDs: {sorted(missing)}")


def _common_frame_dt(
  inventory: tuple[ClipGroup, ...],
  selected: tuple[TrackingClip, ...],
) -> float:
  selected_groups = {clip.group_id for clip in selected}
  frame_dts = {
    group.frame_dt for group in inventory if group.group_id in selected_groups
  }
  if len(frame_dts) != 1:
    raise ValueError(
      f"selected clip groups must use one frame_dt, got {sorted(frame_dts)}"
    )
  return frame_dts.pop()
