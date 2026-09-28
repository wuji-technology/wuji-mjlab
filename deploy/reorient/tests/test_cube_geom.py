# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import os

import pytest
from wuji_reorient_deploy.cube_geom import (
  DEFAULT_CUBE_CONFIG_FILE,
  resolve_cube_config_path,
)


def test_default_arg_resolves_to_the_checked_in_baseline():
  for arg in (None, "", "default"):
    assert resolve_cube_config_path(arg) == DEFAULT_CUBE_CONFIG_FILE


def test_default_arg_raises_if_the_baseline_file_is_missing(monkeypatch, tmp_path):
  missing = tmp_path / "cube_tags.json"
  monkeypatch.setattr(
    "wuji_reorient_deploy.cube_geom.DEFAULT_CUBE_CONFIG_FILE", str(missing)
  )
  with pytest.raises(FileNotFoundError):
    resolve_cube_config_path(None)


def test_explicit_existing_path_is_used_as_is(tmp_path):
  custom = tmp_path / "custom_tags.json"
  custom.write_text("{}")
  assert resolve_cube_config_path(str(custom)) == os.path.abspath(str(custom))


def test_nonexistent_explicit_path_raises_file_not_found(tmp_path):
  missing = tmp_path / "missing_cube_tags.json"
  with pytest.raises(FileNotFoundError):
    resolve_cube_config_path(str(missing))
