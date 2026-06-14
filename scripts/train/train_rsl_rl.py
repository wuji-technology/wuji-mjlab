# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Training wrapper with IsaacLab-style env/agent overrides."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import mjlab
import tyro
import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.registry import list_tasks, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder
from wuji_mjlab.utils.cli_override_utils import (
  DEFAULT_TASK_ALIASES,
  resolve_task,
  split_special_args,
)
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs


def _cuda_device_count() -> int:
  import torch

  try:
    return int(torch.cuda.device_count())
  except Exception:
    return 0


def _maybe_force_cpu_when_cuda_unavailable(
  cfg: TrainConfig,
) -> tuple[bool, TrainConfig, str | None]:
  if cfg.gpu_ids is None:
    return False, cfg, None

  if _cuda_device_count() > 0:
    return False, cfg, None

  updated_cfg = replace(cfg, gpu_ids=None)
  return (
    True,
    updated_cfg,
    "[WARN] CUDA requested but no CUDA devices are available; falling back to CPU.",
  )


def _run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent

  resume_path: Path | None = None
  if cfg.agent.resume:
    if cfg.wandb_run_path is not None:
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      if rank == 0:
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
          f"(run: {run_id}, {cached_str})"
        )
    else:
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)
  env_cfg["task_id"] = task_id

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner = runner_cls(env, agent_cfg, str(log_dir), device)

  add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def _launch_training(task_id: str, cfg: TrainConfig) -> None:
  log_root_path = Path("logs") / "rsl_rl" / cfg.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if cfg.agent.run_name:
    log_dir_name += f"_{cfg.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  selected_gpus, num_gpus = select_gpus(cfg.gpu_ids)

  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    _run_train(task_id, cfg, log_dir)
  else:
    import torchrunx

    logging.basicConfig(level=logging.INFO)

    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if cfg.torchrunx_log_dir is not None:
        os.environ["TORCHRUNX_LOG_DIR"] = cfg.torchrunx_log_dir
      else:
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(_run_train, task_id, cfg, log_dir)


def main() -> None:
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--task", type=str, default=None)
  known, remaining = parser.parse_known_args()

  task_from_equals, tyro_args, overrides = split_special_args(remaining)
  chosen_task = known.task or task_from_equals or "reorient"
  task_id = resolve_task(
    chosen_task,
    task_aliases=DEFAULT_TASK_ALIASES,
    all_tasks=list_tasks(),
  )
  prepared_env_cfg, prepared_agent_cfg = prepare_task_cfgs(task_id, overrides, play=False)

  cfg = tyro.cli(
    TrainConfig,
    args=tyro_args,
    default=replace(
      TrainConfig.from_task(task_id),
      env=prepared_env_cfg,
      agent=prepared_agent_cfg,
    ),
    prog=sys.argv[0] + f" --task {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )

  changed, cfg, message = _maybe_force_cpu_when_cuda_unavailable(cfg)
  if changed and message is not None:
    print(message)
  _launch_training(task_id=task_id, cfg=cfg)


if __name__ == "__main__":
  main()
