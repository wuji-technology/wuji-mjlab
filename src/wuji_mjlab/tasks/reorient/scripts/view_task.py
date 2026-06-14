# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Visualize a task with a dummy policy for model/collision inspection.

Examples:
  pixi run python src/wuji_mjlab/tasks/reorient/scripts/view_task.py WujiHand_Reorient --num-envs 1 --viewer viser
  pixi run python src/wuji_mjlab/tasks/reorient/scripts/view_task.py WujiHand_Reorient --num-envs 1 --viewer native --play
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

import torch
import tyro
import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class ViewTaskConfig:
  agent: Literal["zero", "random"] = "zero"
  play: bool = False
  num_envs: int = 1
  device: str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"


def run_view(task_id: str, cfg: ViewTaskConfig) -> None:
  import mjlab.tasks  # noqa: F401

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task_id, play=cfg.play)
  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=None)

  action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore
  if cfg.agent == "zero":

    class PolicyZero:
      def __call__(self, obs) -> torch.Tensor:
        del obs
        return torch.zeros(action_shape, device=env.unwrapped.device)

    policy = PolicyZero()
  else:

    class PolicyRandom:
      def __call__(self, obs) -> torch.Tensor:
        del obs
        return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

    policy = PolicyRandom()

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    viewer = "native" if has_display else "viser"
  else:
    viewer = cfg.viewer

  if viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif viewer == "viser":
    print(
      "[INFO] In Viser, open the 'Geoms' tab and enable Group 3 to inspect collision geometry."
    )
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {viewer}")

  env.close()


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()

  # Handle --help / -h before the first tyro pass, since the positional
  # task-id parser uses `add_help=False` and would reject `--help` as an
  # invalid choice otherwise.
  if any(a in ("-h", "--help") for a in sys.argv[1:]):
    print("Usage: view_task.py TASK_ID [OPTIONS]\n")
    print("Available TASK_ID choices:")
    for t in all_tasks:
      print(f"  {t}")
    print("\nOptions (after TASK_ID):")
    tyro.cli(
      ViewTaskConfig,
      args=["--help"],
      default=ViewTaskConfig(),
      prog=sys.argv[0] + " TASK_ID",
      config=(
        tyro.conf.AvoidSubcommands,
      ),
    )
    sys.exit(0)

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  args = tyro.cli(
    ViewTaskConfig,
    args=remaining_args,
    default=ViewTaskConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
    ),
  )

  run_view(chosen_task, args)
