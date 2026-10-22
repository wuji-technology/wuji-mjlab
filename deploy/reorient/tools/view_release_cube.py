#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Open an interactive viewer of the release-bundled printable cube."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
  parser = argparse.ArgumentParser(
    description="View the release-bundled printable cube mesh."
  )
  parser.add_argument(
    "--cube",
    type=Path,
    default=Path("release-assets/reorient/hardware/cube.obj"),
    help=(
      "Path to cube.obj. Defaults to "
      "release-assets/reorient/hardware/cube.obj relative to cwd "
      "(the layout produced by the unzip + rename step in "
      "docs/external/en/introduction.mdx). "
      "Sibling cube.mtl and cube.png must be "
      "alongside."
    ),
  )
  args = parser.parse_args()

  if not args.cube.is_file():
    print(
      f"cube file not found: {args.cube}\n"
      "Download the release asset zip per "
      "docs/external/en/introduction.mdx\n"
      "  https://github.com/wuji-technology/wuji-mjlab/releases/latest\n"
      "and unzip + rename so 'release-assets/reorient/hardware/cube.obj'\n"
      f"lives at {args.cube.resolve()},\n"
      "or pass --cube <path> explicitly.",
      file=sys.stderr,
    )
    return 1

  import trimesh

  mesh = trimesh.load(str(args.cube))
  mesh.show()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
