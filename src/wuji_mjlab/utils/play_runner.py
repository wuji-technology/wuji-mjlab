# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Thin local play adapter that accepts already-prepared cfg objects."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts.play import PlayConfig
from mjlab.tasks.registry import load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


def _resolve_downloaded_motion_path(download_dir: Path) -> Path:
  preferred_names = ("trajectories_dyn.npz", "trajectories_kin.npz", "motion.npz")
  for name in preferred_names:
    candidate = download_dir / name
    if candidate.exists():
      return candidate

  motion_files = sorted(
    list(download_dir.glob("*.pt")) + list(download_dir.glob("*.npz"))
  )
  if len(motion_files) == 1:
    return motion_files[0]
  if len(motion_files) > 1:
    raise RuntimeError(
      f"Multiple motion files found under artifact download: {download_dir}"
    )
  raise FileNotFoundError(
    f"No motion file found under artifact download: {download_dir}"
  )


def _get_motion_path_cfg(motion_cmd: Any) -> tuple[str | None, str]:
  motion_files = getattr(motion_cmd, "motion_files", None)
  if motion_files is not None:
    return motion_files[0] if motion_files else None, "motion_files"

  motion_file = getattr(motion_cmd, "motion_file", None)
  if motion_file is not None:
    return motion_file, "motion_file"

  dataset_cfg = getattr(motion_cmd, "dataset_cfg", None)
  if dataset_cfg is not None:
    return getattr(dataset_cfg, "data_path", None), "dataset_cfg.data_path"

  raise AttributeError(
    "Unsupported tracking command cfg: missing motion_files, motion_file, and dataset_cfg.data_path"
  )


def _set_motion_path_cfg(motion_cmd: Any, motion_path: str) -> None:
  # motion_files is our plural API (used by the NPZ loader); motion_file is
  # the mjlab base-class compat. Keep both in sync.
  set_any = False
  if hasattr(motion_cmd, "motion_files"):
    motion_cmd.motion_files = (motion_path,)
    set_any = True
  if hasattr(motion_cmd, "motion_file"):
    motion_cmd.motion_file = motion_path
    set_any = True
  if set_any:
    return

  dataset_cfg = getattr(motion_cmd, "dataset_cfg", None)
  if dataset_cfg is not None and hasattr(dataset_cfg, "data_path"):
    dataset_cfg.data_path = motion_path
    return

  raise AttributeError(
    "Unsupported tracking command cfg: missing motion_files, motion_file, and dataset_cfg.data_path"
  )


def run_play_with_cfg(task_id: str, cfg: PlayConfig, env_cfg: Any, agent_cfg: Any):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    bundled_motion_file, motion_field = _get_motion_path_cfg(motion_cmd)
    bundled_path = Path(bundled_motion_file) if bundled_motion_file else None
    has_bundled_motion_file = bundled_path is not None and bundled_path.exists()
    motion_path = Path(cfg.motion_file) if cfg.motion_file is not None else None

    if motion_path is not None:
      if not motion_path.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")
      print(f"[INFO]: Using local motion file: {motion_path}")
      _set_motion_path_cfg(motion_cmd, str(motion_path))
    elif has_bundled_motion_file:
      print(f"[INFO]: Using bundled motion file from {motion_field}: {bundled_path}")
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion-file (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      _set_motion_path_cfg(
        motion_cmd,
        str(_resolve_downloaded_motion_path(Path(artifact.download()))),
      )
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        _set_motion_path_cfg(motion_cmd, cfg.motion_file)
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          _set_motion_path_cfg(
            motion_cmd,
            str(_resolve_downloaded_motion_path(Path(art.download()))),
          )

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      assert resume_path is not None
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    assert resume_path is not None
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  env.max_common_steps = agent_cfg.max_iterations * agent_cfg.num_steps_per_env

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
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
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()
