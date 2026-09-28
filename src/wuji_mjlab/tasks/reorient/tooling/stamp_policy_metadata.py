# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Stamp a verified semantic identity onto a reorient ONNX policy."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import onnx
from wuji_reorient_deploy.constants import joint_names, load_constants
from wuji_reorient_deploy.onnx_policy import (
  POLICY_IDENTITY_FIELDS,
  POLICY_SCHEMA,
)

from wuji_mjlab.tasks.reorient.tooling.onnx_export_core import (
  _build_config,
  _load_yaml,
  _set_policy_identity_metadata,
)


def _identity_from_run_dir(run_dir: Path) -> dict[str, Any] | None:
  env_path = run_dir / "params" / "env.yaml"
  if not env_path.is_file():
    return None
  env_cfg = _load_yaml(str(env_path))
  if env_cfg is None:
    raise ValueError(f"Cannot parse existing environment config {env_path}")
  config = _build_config(str(run_dir), env_cfg)
  return {field: config[field] for field in POLICY_IDENTITY_FIELDS}


def _identity_from_constants(gen: int, hand: str) -> dict[str, Any]:
  constants = load_constants(gen, hand)
  hand_asset = "wuji_hand2" if gen == 2 else "wuji_hand"
  return {
    "policy_schema": POLICY_SCHEMA,
    "hand_gen": gen,
    "robot_mjcf": str(
      Path("src")
      / "wuji_mjlab"
      / "assets"
      / "robots"
      / hand_asset
      / "mjcf"
      / f"{hand}_mjlab.xml"
    ),
    "joint_order": joint_names(hand),
    "default_joint_pos": constants.default_joint_pos.tolist(),
    "default_joint_pos_source": "load_constants(gen, hand).default_joint_pos",
    "obs_term_names": [name for name, _ in constants.obs_terms],
  }


def stamp_policy_metadata(
  onnx_path: str | Path, identity: dict[str, Any]
) -> tuple[Path, Path]:
  onnx_path = Path(onnx_path)
  if not onnx_path.is_file():
    raise FileNotFoundError(f"ONNX policy does not exist: {onnx_path}")

  missing = [field for field in POLICY_IDENTITY_FIELDS if field not in identity]
  if missing:
    raise ValueError(f"Policy identity is missing fields: {missing}")

  candidates = (Path(f"{onnx_path}.config.json"), onnx_path.parent / "config.json")
  config_path = next((path for path in candidates if path.is_file()), None)
  if config_path is None:
    raise FileNotFoundError(
      f"Cannot stamp {onnx_path}: no sidecar found at {candidates[0]} or {candidates[1]}"
    )
  try:
    config = json.loads(config_path.read_text())
  except json.JSONDecodeError as exc:
    raise ValueError(f"Cannot parse {config_path}: {exc}") from exc
  if not isinstance(config, dict):
    raise ValueError(f"Cannot update {config_path}: expected a JSON object")

  backup_path = Path(f"{onnx_path}.bak")
  if backup_path.exists():
    raise FileExistsError(f"Refusing to overwrite existing backup {backup_path}")

  model = onnx.load(onnx_path, load_external_data=True)
  identity = {field: identity[field] for field in POLICY_IDENTITY_FIELDS}
  _set_policy_identity_metadata(model, identity)
  config.update(identity)

  shutil.copy2(onnx_path, backup_path)
  onnx.save(model, onnx_path, save_as_external_data=False)
  config_path.write_text(json.dumps(config, indent=2) + "\n")
  return backup_path, config_path


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Stamp verified robot identity into a legacy reorient ONNX policy"
  )
  parser.add_argument("onnx_path", type=Path, help="legacy .onnx policy to update")
  parser.add_argument(
    "--gen", type=int, choices=(1, 2), required=True, help="verified hand generation"
  )
  parser.add_argument("--hand", choices=("right",), default="right")
  parser.add_argument(
    "--run-dir",
    type=Path,
    default=None,
    help="training run directory (default: directory containing the ONNX model)",
  )
  verification = parser.add_mutually_exclusive_group()
  verification.add_argument("--i-verified-this-is-gen1", action="store_true")
  verification.add_argument("--i-verified-this-is-gen2", action="store_true")
  return parser


def main(argv: Sequence[str] | None = None) -> None:
  parser = _build_parser()
  args = parser.parse_args(argv)
  run_dir = args.run_dir or args.onnx_path.parent

  try:
    identity = _identity_from_run_dir(run_dir)
  except (OSError, ValueError) as exc:
    parser.error(str(exc))

  if identity is not None:
    print(f"Inferred policy identity from {run_dir / 'params' / 'env.yaml'}:")
    print(json.dumps(identity, indent=2), flush=True)
    if identity["hand_gen"] != args.gen:
      parser.error(
        f"--gen {args.gen} disagrees with the inferred hand_gen "
        f"{identity['hand_gen']}; refusing to stamp"
      )
  else:
    verified = (
      args.i_verified_this_is_gen1 if args.gen == 1 else args.i_verified_this_is_gen2
    )
    if not verified:
      parser.error(
        f"{run_dir / 'params' / 'env.yaml'} is unavailable; --gen {args.gen} "
        f"requires --i-verified-this-is-gen{args.gen}"
      )
    identity = _identity_from_constants(args.gen, args.hand)
    print(
      "No params/env.yaml was found; using deploy constants after explicit "
      "operator verification. Identity to be written:"
    )
    print(json.dumps(identity, indent=2), flush=True)

  if args.i_verified_this_is_gen1 and args.gen != 1:
    parser.error("--i-verified-this-is-gen1 cannot be used with --gen 2")
  if args.i_verified_this_is_gen2 and args.gen != 2:
    parser.error("--i-verified-this-is-gen2 cannot be used with --gen 1")

  try:
    backup_path, config_path = stamp_policy_metadata(args.onnx_path, identity)
  except (OSError, ValueError) as exc:
    parser.error(str(exc))
  print(f"Backed up original ONNX: {backup_path}")
  print(f"Stamped ONNX metadata: {args.onnx_path}")
  print(f"Updated sidecar config: {config_path}")
