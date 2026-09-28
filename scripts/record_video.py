# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Headless video recording — runs N steps then exits."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from wuji_mjlab.utils.cli_override_utils import (
  DEFAULT_TASK_ALIASES,
  resolve_task,
  split_special_args,
)
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", type=str, default="reorient")
  parser.add_argument("--checkpoint-file", type=str, required=True)
  parser.add_argument("--num-envs", type=int, default=None)
  parser.add_argument("--video-length", type=int, default=1000)
  parser.add_argument("--video-dir", type=str, default=None)
  known, remaining = parser.parse_known_args()

  _, _, overrides = split_special_args(remaining)
  from mjlab.tasks.registry import list_tasks
  task_id = resolve_task(known.task, task_aliases=DEFAULT_TASK_ALIASES, all_tasks=list_tasks())

  env_cfg, agent_cfg = prepare_task_cfgs(task_id, overrides, play=True)

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  # Only override num_envs if user explicitly passed --num-envs.
  # The play=True branch in env_cfgs.py already sets num_envs=1, and stacking
  # multiple envs at the same world origin (no env_spacing applied without a
  # terrain) breaks the rgb_array renderer — hands become invisible.
  if known.num_envs is not None:
    env_cfg.scene.num_envs = known.num_envs
  env_cfg.viewer.height = 480
  env_cfg.viewer.width = 640

  ckpt = Path(known.checkpoint_file)
  if not ckpt.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

  video_dir = Path(known.video_dir) if known.video_dir else ckpt.parent / "videos" / "play"

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  env = VideoRecorder(
    env,
    video_folder=video_dir,
    step_trigger=lambda step: step == 0,
    video_length=known.video_length,
    disable_logger=True,
  )
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  obs = env.get_observations()
  # Circular history buffers reject grad-tracked inputs during in-place writes.
  with torch.inference_mode():
    for _ in range(known.video_length + 10):
      action = policy(obs)
      obs, _, _, _ = env.step(action)

  env.close()
  print(f"[INFO] Video saved to: {video_dir}")


if __name__ == "__main__":
  main()
