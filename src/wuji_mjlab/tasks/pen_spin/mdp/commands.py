# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  combine_frame_transforms,
  quat_apply,
  quat_inv,
  quat_mul,
  subtract_frame_transforms,
)

from wuji_mjlab.tasks.pen_spin.tooling.clip_store import ResolvedClipSet, TrackingClip
from wuji_mjlab.tasks.pen_spin.tooling.recipe import resolve_recipe

from .observations import (
  bounded_object_pose_to_points_wrist_local,
  make_axis_points,
  object_pose_to_points_wrist_local,
  world_points_to_wrist_local,
)


@dataclass(frozen=True)
class TrajectoryGroup:
  """Command-owned trajectories for exactly one object asset."""

  frame_dt: float
  trajectory_end: torch.Tensor
  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  wrist_pose_w: torch.Tensor
  object_pose: torch.Tensor
  link_pos: torch.Tensor

  @classmethod
  def from_reader_data(cls, data: dict[str, object]) -> TrajectoryGroup:
    trajectory_lengths = torch.as_tensor(data["trajectory_lengths"], dtype=torch.long)
    return cls(
      frame_dt=data["frame_dt"],
      trajectory_end=torch.cumsum(trajectory_lengths, dim=0),
      joint_pos=data["joint_pos"],
      joint_vel=data["joint_vel"],
      wrist_pose_w=data["wrist_pose_w"],
      object_pose=data["object_pose"],
      link_pos=data["link_pos"],
    )

  @property
  def num_trajectories(self) -> int:
    return self.trajectory_end.numel()


@dataclass(frozen=True, kw_only=True)
class SingleHandTrackingVisualizationCfg:
  hand_color: tuple[float, float, float] = (0.45, 0.75, 1.0)
  hand_alpha: float = 0.38
  object_alpha: float = 0.42
  actual_frame_scale: float = 0.12
  reference_frame_scale: float = 0.15
  wrist_frame_scale: float = 0.12
  reference_frame_radius: float = 0.004
  reference_frame_alpha: float = 0.45
  link_radius: float = 0.0045
  link_color: tuple[float, float, float, float] = (0.15, 0.55, 1.0, 0.8)
  # Recording enhancement, disabled by default: reference-object tint and future trajectory.
  reference_object_color: tuple[float, float, float] | None = None
  reference_trace_seconds: float = 0.0
  reference_trace_stride: int = 10
  reference_trace_color: tuple[float, float, float] = (0.1, 0.85, 0.25)


@dataclass(kw_only=True)
class SingleHandTrackingCommandCfg(CommandTermCfg):
  recipe_path: str = ""
  axis_point_radius_m: float = 0.10
  object_position_tanh_scale_m: float = 0.15
  trajectory_sampling_mode: str = "uniform"
  adaptive_sampling_ema_decay: float = 0.99
  adaptive_sampling_uniform_mix: float = 0.20
  adaptive_sampling_bin_seconds: float = 1.0
  adaptive_sampling_update_steps: int = 40
  adaptive_sampling_kernel_size: int = 1
  adaptive_sampling_kernel_decay: float = 0.8
  # Warmup: this share of resets holds the clip's first frame for a duration drawn
  # from the range, as deploy does until the operator starts the clip.
  first_frame_hold_prob: float = 0.0
  first_frame_hold_s: tuple[float, float] = (0.0, 0.0)
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  joint_names: tuple[str, ...] = ()
  link_names: tuple[str, ...] = ()
  fixed_wrist_pose_w: tuple[float, float, float, float, float, float, float] | None = (
    None
  )
  viz: SingleHandTrackingVisualizationCfg = SingleHandTrackingVisualizationCfg()

  def build(self, env) -> SingleHandTrackingCommand:
    return SingleHandTrackingCommand(self, env)


class SingleHandTrackingCommand(CommandTerm):
  cfg: SingleHandTrackingCommandCfg

  def __init__(self, cfg: SingleHandTrackingCommandCfg, env):
    super().__init__(cfg, env)
    if not cfg.joint_names:
      raise ValueError("joint_names must be configured by the concrete robot config")
    if not cfg.link_names:
      raise ValueError("link_names must be configured by the concrete robot config")
    if cfg.axis_point_radius_m <= 0:
      raise ValueError("axis_point_radius_m must be positive")
    if not (
      math.isfinite(cfg.object_position_tanh_scale_m)
      and cfg.object_position_tanh_scale_m > 0.0
    ):
      raise ValueError("bounded object mapping scale must be finite and positive")
    if cfg.trajectory_sampling_mode not in ("uniform", "adaptive"):
      raise ValueError(
        f"trajectory_sampling_mode must be 'uniform' or 'adaptive', got {cfg.trajectory_sampling_mode!r}"
      )
    if not (
      math.isfinite(cfg.adaptive_sampling_ema_decay)
      and 0.0 <= cfg.adaptive_sampling_ema_decay < 1.0
    ):
      raise ValueError("adaptive_sampling_ema_decay must be in [0, 1)")
    if not (
      math.isfinite(cfg.adaptive_sampling_uniform_mix)
      and 0.0 <= cfg.adaptive_sampling_uniform_mix <= 1.0
    ):
      raise ValueError("adaptive_sampling_uniform_mix must be in [0, 1]")
    if not (
      math.isfinite(cfg.adaptive_sampling_bin_seconds)
      and cfg.adaptive_sampling_bin_seconds > 0.0
    ):
      raise ValueError("adaptive_sampling_bin_seconds must be finite and positive")
    if cfg.adaptive_sampling_update_steps < 1:
      raise ValueError("adaptive_sampling_update_steps must be at least 1")
    if (
      not isinstance(cfg.adaptive_sampling_kernel_size, int)
      or isinstance(cfg.adaptive_sampling_kernel_size, bool)
      or cfg.adaptive_sampling_kernel_size <= 0
    ):
      raise ValueError("adaptive_sampling_kernel_size must be a positive integer")
    if not (
      math.isfinite(cfg.adaptive_sampling_kernel_decay)
      and 0.0 <= cfg.adaptive_sampling_kernel_decay <= 1.0
    ):
      raise ValueError("adaptive_sampling_kernel_decay must be in [0, 1]")

    if not (
      math.isfinite(cfg.first_frame_hold_prob)
      and 0.0 <= cfg.first_frame_hold_prob <= 1.0
    ):
      raise ValueError("first_frame_hold_prob must be in [0, 1]")
    hold_min_s, hold_max_s = cfg.first_frame_hold_s
    if not (math.isfinite(hold_max_s) and 0.0 <= hold_min_s <= hold_max_s):
      raise ValueError(
        "first_frame_hold_s must be a finite (min, max) with 0 <= min <= max"
      )

    (clip_set,) = resolve_recipe(cfg.recipe_path, verify_clip_hashes=True).clip_sets
    self.group = TrajectoryGroup.from_reader_data(
      SingleHandTrackingReader(clip_set, device=self.device).read()
    )
    self._validate_groups()

    self._trajectory_ends = self.group.trajectory_end
    self._trajectory_starts = torch.cat(
      (self._trajectory_ends.new_zeros(1), self._trajectory_ends[:-1])
    )
    self._configure_adaptive_sampling_bins()
    self._frames = {
      field: getattr(self.group, field)
      for field in (
        "joint_pos",
        "joint_vel",
        "wrist_pose_w",
        "object_pose",
        "link_pos",
      )
    }
    self._configure_fixed_wrist_reference()
    self.trajectory_ids = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.time_steps = torch.zeros_like(self.trajectory_ids)
    self._episode_start_steps = torch.zeros_like(self.trajectory_ids)
    self._reset_prepared = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.bin_failure_scores = torch.zeros(
      self._bin_trajectory_ids.numel(),
      dtype=torch.float32,
      device=self.device,
    )
    # Episode outcomes since the last score update, [trials; failures] per bin.
    self._pending_bin_outcomes = torch.zeros(
      (2, self._bin_trajectory_ids.numel()),
      dtype=torch.float32,
      device=self.device,
    )
    self._steps_since_score_update = 0
    self._sampling_probability = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device
    )
    self._sampled_bin_failure_score = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device
    )
    self._sampling_entropy = torch.zeros_like(self._sampling_probability)
    self._sampling_top1_prob = torch.zeros_like(self._sampling_probability)
    self._sampling_top1_bin = torch.zeros_like(self._sampling_probability)
    self.metrics["sampling_probability"] = self._sampling_probability.clone()
    self.metrics["bin_failure_score"] = self._sampled_bin_failure_score.clone()
    self.metrics["sampling_entropy"] = self._sampling_entropy.clone()
    self.metrics["sampling_top1_prob"] = self._sampling_top1_prob.clone()
    self.metrics["sampling_top1_bin"] = self._sampling_top1_bin.clone()

    dtype = self._frames["object_pose"].dtype
    self.axis_points = make_axis_points(
      cfg.axis_point_radius_m, device=self.device, dtype=dtype
    )
    self._object_axis_points = object_pose_to_points_wrist_local(
      self._frames["object_pose"], self.axis_points
    ).flatten(1)
    self._bounded_object_axis_points = bounded_object_pose_to_points_wrist_local(
      self._frames["object_pose"],
      self.axis_points,
      position_tanh_scale_m=cfg.object_position_tanh_scale_m,
    ).flatten(1)

  def _configure_adaptive_sampling_bins(self) -> None:
    """Build boundary-safe local start-frame bins for every trajectory."""
    trajectory_lengths = self._trajectory_ends - self._trajectory_starts
    bin_size_frames = max(
      1,
      round(self.cfg.adaptive_sampling_bin_seconds / self.group.frame_dt),
    )
    self._adaptive_sampling_bin_size_frames = bin_size_frames

    start_slots = (trajectory_lengths - 1).clamp_min(1)
    self._trajectory_bin_counts = torch.div(
      start_slots + bin_size_frames - 1,
      bin_size_frames,
      rounding_mode="floor",
    )
    self._trajectory_bin_offsets = torch.cat(
      (
        self._trajectory_bin_counts.new_zeros(1),
        self._trajectory_bin_counts.cumsum(0)[:-1],
      )
    )

    trajectory_ids = torch.arange(
      trajectory_lengths.numel(), device=self.device, dtype=torch.long
    )
    self._bin_trajectory_ids = torch.repeat_interleave(
      trajectory_ids, self._trajectory_bin_counts
    )
    bin_ids = torch.arange(
      self._bin_trajectory_ids.numel(), device=self.device, dtype=torch.long
    )
    local_bin_ids = bin_ids - self._trajectory_bin_offsets[self._bin_trajectory_ids]
    self._bin_start_steps = local_bin_ids * bin_size_frames
    self._bin_stop_steps = torch.minimum(
      self._bin_start_steps + bin_size_frames,
      start_slots[self._bin_trajectory_ids],
    )

    kernel_offsets = torch.arange(
      self.cfg.adaptive_sampling_kernel_size,
      device=self.device,
      dtype=torch.long,
    )
    final_bin_ids = (
      self._trajectory_bin_offsets[self._bin_trajectory_ids]
      + self._trajectory_bin_counts[self._bin_trajectory_ids]
      - 1
    )
    self._bin_kernel_indices = torch.minimum(
      bin_ids[:, None] + kernel_offsets[None, :],
      final_bin_ids[:, None],
    )
    kernel = self.cfg.adaptive_sampling_kernel_decay ** kernel_offsets.to(torch.float32)
    self._adaptive_sampling_kernel = kernel / kernel.sum()

  def _configure_fixed_wrist_reference(self) -> None:
    """Rebase every wrist-local trajectory onto one physical wrist mount."""
    fixed_pose = self.cfg.fixed_wrist_pose_w
    if fixed_pose is None:
      raise ValueError(
        "fixed_wrist_pose_w must be configured by the concrete robot config"
      )
    dtype = self._frames["wrist_pose_w"].dtype
    pose = torch.as_tensor(
      fixed_pose,
      device=self.device,
      dtype=dtype,
    )
    if pose.shape != (7,) or not torch.isfinite(pose).all():
      raise ValueError("fixed_wrist_pose_w must contain seven finite values")
    quat_norm = torch.linalg.vector_norm(pose[3:])
    if quat_norm <= 0:
      raise ValueError("fixed_wrist_pose_w must contain a non-zero quaternion")
    pose = pose.clone()
    pose[3:] /= quat_norm
    wrist_pose_w = (
      pose.unsqueeze(0).expand(self._frames["wrist_pose_w"].shape[0], -1).clone()
    )
    object_pose = self._frames["object_pose"]
    object_pos_w, object_quat_w = combine_frame_transforms(
      wrist_pose_w[:, :3],
      wrist_pose_w[:, 3:],
      object_pose[:, :3],
      object_pose[:, 3:],
    )
    object_pose_w = torch.cat((object_pos_w, object_quat_w), dim=-1)
    object_linvel_w = torch.zeros_like(object_pos_w)
    object_angvel_w = torch.zeros_like(object_pos_w)
    frame_dt = self.group.frame_dt
    for start, end in zip(
      self._trajectory_starts.tolist(),
      self._trajectory_ends.tolist(),
      strict=True,
    ):
      linear, angular = object_velocity_from_pose(
        object_pose_w[start:end],
        frame_dt=frame_dt,
      )
      object_linvel_w[start:end] = linear
      object_angvel_w[start:end] = angular
    self._frames.update(
      wrist_pose_w=wrist_pose_w,
      object_linvel_w=object_linvel_w,
      object_angvel_w=object_angvel_w,
    )

  def _validate_control_frequency(self, frame_dt: float) -> None:
    """The command advances one clip frame per control step, so frame_dt must equal step_dt."""
    control_dt = float(self._env.step_dt)
    frame_dt = float(frame_dt)
    if abs(frame_dt - control_dt) > 1e-3 * control_dt:
      raise ValueError(
        f"clip fps {1.0 / frame_dt:.4g} Hz (frame_dt={frame_dt:.6g}s) must match "
        f"the control frequency {1.0 / control_dt:.4g} Hz (step_dt={control_dt:.6g}s): "
        "a mismatch time-warps the reference."
      )

  def _validate_groups(self) -> None:
    group = self.group
    self._validate_control_frequency(group.frame_dt)
    expected_joints = len(self.cfg.joint_names)
    expected_links = tuple(self.cfg.link_names)
    if len(set(expected_links)) != len(expected_links):
      raise ValueError("link_names must be unique")
    if group.joint_pos.ndim != 2 or group.joint_pos.shape[1] != expected_joints:
      raise ValueError(
        "tracking joint width must match configured joint_names; "
        f"expected {expected_joints}, got {tuple(group.joint_pos.shape)}"
      )
    if group.link_pos.ndim != 3 or group.link_pos.shape[1:] != (
      len(expected_links),
      3,
    ):
      raise ValueError(
        "tracking link shape must match configured link_names; "
        f"expected (*, {len(expected_links)}, 3), got {tuple(group.link_pos.shape)}"
      )

  @property
  def joint_names(self) -> tuple[str, ...]:
    return tuple(self.cfg.joint_names)

  @property
  def trajectory_starts(self) -> torch.Tensor:
    return self._trajectory_starts[self.trajectory_ids]

  @property
  def trajectory_ends(self) -> torch.Tensor:
    return self._trajectory_ends[self.trajectory_ids]

  @property
  def trajectory_lengths(self) -> torch.Tensor:
    return self.trajectory_ends - self.trajectory_starts

  @property
  def frame_ids(self) -> torch.Tensor:
    elapsed = self.time_steps.clamp_min(0).minimum(self.trajectory_lengths - 1)
    return self.trajectory_starts + elapsed

  def _current(self, field: str) -> torch.Tensor:
    return self._frames[field][self.frame_ids]

  @property
  def joint_pos(self) -> torch.Tensor:
    return self._current("joint_pos")

  @property
  def holding_first_frame(self) -> torch.Tensor:
    """True while the warmup holds the first frame; ``time_steps`` counts up to 0 there."""
    return self.time_steps < 0

  def _current_velocity(self, field: str) -> torch.Tensor:
    """A held frame does not move, whatever velocity the clip starts with."""
    return self._current(field) * ~self.holding_first_frame.unsqueeze(-1)

  @property
  def joint_vel(self) -> torch.Tensor:
    return self._current_velocity("joint_vel")

  @property
  def wrist_pose_w(self) -> torch.Tensor:
    pose = self._current("wrist_pose_w")
    pose[:, :3] += self._env.scene.env_origins
    return pose

  @property
  def object_pose(self) -> torch.Tensor:
    return self._current("object_pose")

  @property
  def object_linvel_w(self) -> torch.Tensor:
    """World-frame object linear velocity for the reset injection (events.apply_reference_state)."""
    return self._current_velocity("object_linvel_w")

  @property
  def object_angvel_w(self) -> torch.Tensor:
    """World-frame object angular velocity for the reset injection (events.apply_reference_state)."""
    return self._current_velocity("object_angvel_w")

  @property
  def link_pos(self) -> torch.Tensor:
    return self._current("link_pos")

  @property
  def object_axis_points(self) -> torch.Tensor:
    return self._object_axis_points[self.frame_ids]

  @property
  def bounded_object_axis_points(self) -> torch.Tensor:
    return self._bounded_object_axis_points[self.frame_ids]

  @property
  def command(self) -> torch.Tensor:
    """Current-frame reference: finger qpos, bounded pen axis points, link positions."""
    return torch.cat(
      (self.joint_pos, self.bounded_object_axis_points, self.link_pos.flatten(1)),
      dim=-1,
    )

  def _bin_ids_at(
    self, env_ids: torch.Tensor, local_steps: torch.Tensor
  ) -> torch.Tensor:
    trajectory_ids = self.trajectory_ids[env_ids]
    trajectory_lengths = (
      self._trajectory_ends[trajectory_ids] - self._trajectory_starts[trajectory_ids]
    )
    final_start_steps = (trajectory_lengths - 2).clamp_min(0)
    local_steps = local_steps.clamp_min(0).minimum(final_start_steps)
    local_bin_ids = torch.div(
      local_steps,
      self._adaptive_sampling_bin_size_frames,
      rounding_mode="floor",
    ).minimum(self._trajectory_bin_counts[trajectory_ids] - 1)
    return self._trajectory_bin_offsets[trajectory_ids] + local_bin_ids

  def _current_bin_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
    return self._bin_ids_at(env_ids, self.time_steps[env_ids])

  def _record_bin_outcomes(self, env_ids: torch.Tensor) -> None:
    if self.cfg.trajectory_sampling_mode != "adaptive":
      return
    termination_manager = self._env.termination_manager
    terminated = termination_manager.terminated[env_ids]
    timed_out = termination_manager.time_outs[env_ids]
    unstable = termination_manager.get_term("unstable")[env_ids]
    failed = terminated & ~unstable
    succeeded = timed_out & ~terminated
    has_outcome = (
      (failed | succeeded) & ~unstable & (self._env.episode_length_buf[env_ids] > 0)
    )
    completed_env_ids = env_ids[has_outcome]
    if len(completed_env_ids) == 0:
      return

    # An episode is one trial in every bin it entered: it survived each bin
    # before the last, and the last one takes the outcome. Scoring only the
    # final bin never records survivals, and the sampler flattens to uniform.
    final_bin_ids = self._current_bin_ids(completed_env_ids)
    first_bin_ids = self._bin_ids_at(
      completed_env_ids, self._episode_start_steps[completed_env_ids]
    )
    outcomes = failed[has_outcome].to(self.bin_failure_scores.dtype)
    num_bins = self.bin_failure_scores.numel()
    entered = torch.bincount(first_bin_ids, minlength=num_bins + 1)
    entered -= torch.bincount(final_bin_ids + 1, minlength=num_bins + 1)
    counts = entered.cumsum(0)[:num_bins]
    failure_sums = torch.bincount(
      final_bin_ids,
      weights=outcomes,
      minlength=num_bins,
    )
    self._pending_bin_outcomes[0] += counts
    self._pending_bin_outcomes[1] += failure_sums

  def _update_bin_failure_scores(self) -> None:
    """Fold the pending outcomes of every rank into the scores.

    Runs on a fixed step count, never on a reset: ranks reset at different
    times, and a collective that only some ranks reach hangs the run.
    """
    self._steps_since_score_update += 1
    if self._steps_since_score_update < self.cfg.adaptive_sampling_update_steps:
      return
    self._steps_since_score_update = 0
    if dist.is_initialized():
      dist.all_reduce(self._pending_bin_outcomes)
    counts, failure_sums = self._pending_bin_outcomes
    updated = counts > 0
    batch_failure_rates = failure_sums[updated] / counts[updated]
    decay_powers = self.cfg.adaptive_sampling_ema_decay ** counts[updated]
    self.bin_failure_scores[updated] = (
      decay_powers * self.bin_failure_scores[updated]
      + (1.0 - decay_powers) * batch_failure_rates
    )
    self._pending_bin_outcomes.zero_()

  def _smoothed_bin_failure_scores(self) -> torch.Tensor:
    gathered = self.bin_failure_scores[self._bin_kernel_indices]
    return (gathered * self._adaptive_sampling_kernel).sum(dim=-1)

  def _adaptive_sampling_probabilities(self) -> torch.Tensor:
    scores = self._smoothed_bin_failure_scores()
    uniform = torch.full_like(scores, 1.0 / scores.numel())
    score_sum = scores.sum()
    proportional = torch.where(
      score_sum > 0.0,
      scores / score_sum.clamp_min(torch.finfo(scores.dtype).eps),
      uniform,
    )
    mix = self.cfg.adaptive_sampling_uniform_mix
    return mix * uniform + (1.0 - mix) * proportional

  def _set_sampling_distribution_metrics(
    self,
    selected: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    adaptive: bool,
  ) -> None:
    if probabilities.numel() > 1:
      entropy = -(
        probabilities * probabilities.clamp_min(1.0e-12).log()
      ).sum() / math.log(probabilities.numel())
    else:
      entropy = probabilities.new_tensor(1.0)
    top_probability, top_index = probabilities.max(dim=0)
    self._sampling_entropy[selected] = entropy
    self._sampling_top1_prob[selected] = top_probability
    self._sampling_top1_bin[selected] = (
      top_index.to(probabilities.dtype) / probabilities.numel() if adaptive else 0.0
    )

  def prepare_reset(self, env_ids: torch.Tensor) -> None:
    self._record_bin_outcomes(env_ids)
    if self.cfg.trajectory_sampling_mode == "adaptive":
      probabilities = self._adaptive_sampling_probabilities()
      sampled_bins = torch.multinomial(probabilities, len(env_ids), replacement=True)
      self.trajectory_ids[env_ids] = self._bin_trajectory_ids[sampled_bins]
      starts = self._bin_start_steps[sampled_bins]
      widths = self._bin_stop_steps[sampled_bins] - starts
      random_offsets = (
        torch.rand((len(env_ids),), device=self.device) * widths.to(torch.float32)
      ).to(torch.long)
      self.time_steps[env_ids] = starts + random_offsets
      self._sampling_probability[env_ids] = probabilities[sampled_bins]
      self._sampled_bin_failure_score[env_ids] = self.bin_failure_scores[sampled_bins]
      self._set_sampling_distribution_metrics(env_ids, probabilities, adaptive=True)
    else:
      num_trajectories = self.group.num_trajectories
      probabilities = torch.full(
        (num_trajectories,),
        1.0 / num_trajectories,
        dtype=self.bin_failure_scores.dtype,
        device=self.device,
      )
      sampled_trajectories = torch.randint(
        num_trajectories, (len(env_ids),), device=self.device
      )
      self.trajectory_ids[env_ids] = sampled_trajectories
      self.time_steps[env_ids] = 0
      self._sampling_probability[env_ids] = probabilities[sampled_trajectories]
      self._sampled_bin_failure_score[env_ids] = 0.0
      self._set_sampling_distribution_metrics(env_ids, probabilities, adaptive=False)
    self._hold_first_frame(env_ids)
    self._episode_start_steps[env_ids] = self.time_steps[env_ids]
    self._reset_prepared[env_ids] = True

  def _hold_first_frame(self, env_ids: torch.Tensor) -> None:
    """Send a share of the resets to a held first frame (negative ``time_steps``)."""
    if self.cfg.first_frame_hold_prob <= 0.0:
      return
    held = env_ids[
      torch.rand(len(env_ids), device=self.device) < self.cfg.first_frame_hold_prob
    ]
    if len(held) == 0:
      return
    frame_dt = self.group.frame_dt
    hold_min, hold_max = (
      round(seconds / frame_dt) for seconds in self.cfg.first_frame_hold_s
    )
    self.time_steps[held] = -torch.randint(
      hold_min, hold_max + 1, (len(held),), device=self.device
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    pending = env_ids[~self._reset_prepared[env_ids]]
    if len(pending):
      self.prepare_reset(pending)
    self._reset_prepared[env_ids] = False

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    # env_ids is None on the per-step update and the reset ids on reset();
    # a partial reset must not advance the other envs a second time.
    if env_ids is None:
      self.time_steps += 1
      if self.cfg.trajectory_sampling_mode == "adaptive":
        self._update_bin_failure_scores()
    else:
      self.time_steps[env_ids] += 1

  def _update_metrics(self) -> None:
    self.metrics["sampling_probability"].copy_(self._sampling_probability)
    self.metrics["bin_failure_score"].copy_(self._sampled_bin_failure_score)
    self.metrics["sampling_entropy"].copy_(self._sampling_entropy)
    self.metrics["sampling_top1_prob"].copy_(self._sampling_top1_prob)
    self.metrics["sampling_top1_bin"].copy_(self._sampling_top1_bin)

  def _debug_vis_impl(self, visualizer) -> None:
    from .visualization import draw_reference_visualization

    draw_reference_visualization(self, visualizer)


# Reference trajectory loading helpers.


_ARRAY_KEYS = (
  "reference_qpos",
  "wrist_pose_w",
  "object_pose_w",
  "link_pos_w",
)


class SingleHandTrackingReader:
  """Read the ordered clips selected by one pen-spin recipe."""

  def __init__(
    self,
    clip_set: ResolvedClipSet,
    *,
    device: str | torch.device = "cpu",
  ) -> None:
    if not isinstance(clip_set, ResolvedClipSet):
      raise TypeError("clip_set must be a ResolvedClipSet")
    if not clip_set.clips:
      raise ValueError("ResolvedClipSet must contain at least one clip")
    if not np.isfinite(clip_set.frame_dt) or clip_set.frame_dt <= 0:
      raise ValueError("ResolvedClipSet frame_dt must be finite and positive")
    self._clip_set = clip_set
    self._device = torch.device(device)

  def read(self) -> dict[str, object]:
    trajectories = [
      self._read_trajectory(clip, self._clip_set.frame_dt)
      for clip in self._clip_set.clips
    ]
    lengths = torch.tensor(
      [trajectory["joint_pos"].shape[0] for trajectory in trajectories],
      dtype=torch.long,
      device=self._device,
    )

    def concatenate(field: str) -> torch.Tensor:
      return torch.cat([trajectory[field] for trajectory in trajectories]).to(
        self._device
      )

    return {
      "frame_dt": self._clip_set.frame_dt,
      "trajectory_lengths": lengths,
      "joint_pos": concatenate("joint_pos"),
      "joint_vel": concatenate("joint_vel"),
      "wrist_pose_w": concatenate("wrist_pose_w"),
      "object_pose": concatenate("object_pose"),
      "link_pos": concatenate("link_pos"),
    }

  def _read_trajectory(self, clip: TrackingClip, frame_dt: float) -> dict:
    clip_id = clip.clip_id
    motion_path = clip.npz_path
    if not motion_path.is_file():
      raise FileNotFoundError(f"clip {clip_id!r} motion does not exist: {motion_path}")
    with np.load(motion_path, allow_pickle=False) as data:
      for key in _ARRAY_KEYS:
        if key not in data:
          raise KeyError(f"clip {clip_id!r} is missing required array {key!r}")
      joint_pos = torch.as_tensor(data["reference_qpos"], dtype=torch.float32)
      wrist_pose_w = torch.as_tensor(data["wrist_pose_w"], dtype=torch.float32)
      object_pose_w = torch.as_tensor(data["object_pose_w"], dtype=torch.float32)
      link_pos_w = torch.as_tensor(data["link_pos_w"], dtype=torch.float32)

    frames = joint_pos.shape[0] if joint_pos.ndim else 0
    frame_arrays = {
      "reference_qpos": (joint_pos, (frames, 20)),
      "wrist_pose_w": (wrist_pose_w, (frames, 7)),
      "object_pose_w": (object_pose_w, (frames, 7)),
      "link_pos_w": (link_pos_w, (frames, 21, 3)),
    }
    for name, (value, expected) in frame_arrays.items():
      if tuple(value.shape) != expected:
        raise ValueError(
          f"{name} in clip {clip_id!r} must have shape {expected}, got {tuple(value.shape)}"
        )
      if value.dtype != torch.bool and not torch.isfinite(value).all():
        raise ValueError(f"{name} in clip {clip_id!r} contains non-finite values")

    wrist_quat = _quat(wrist_pose_w[:, 3:], f"wrist_pose_w in clip {clip_id!r}")
    object_quat_w = _quat(object_pose_w[:, 3:], f"object_pose_w in clip {clip_id!r}")
    wrist_pose_w = torch.cat((wrist_pose_w[:, :3], wrist_quat), dim=-1)
    object_pose_w = torch.cat((object_pose_w[:, :3], object_quat_w), dim=-1)
    object_pos, object_quat = subtract_frame_transforms(
      wrist_pose_w[:, :3],
      wrist_quat,
      object_pose_w[:, :3],
      object_quat_w,
    )
    object_pose = torch.cat((object_pos, object_quat), dim=-1)
    link_pos = world_points_to_wrist_local(
      link_pos_w,
      wrist_pos=wrist_pose_w[:, :3],
      wrist_quat=wrist_quat,
    )
    return {
      "joint_pos": joint_pos,
      "joint_vel": _diff(joint_pos, frame_dt),
      "wrist_pose_w": wrist_pose_w,
      "object_pose": object_pose,
      "link_pos": link_pos,
    }


def _quat(value: torch.Tensor, name: str) -> torch.Tensor:
  """Normalize a validated quaternion tensor at the data boundary."""
  if value.shape[-1] != 4:
    raise ValueError(f"{name} contains an invalid quaternion")
  norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
  if torch.any(norm <= 0):
    raise ValueError(f"{name} contains a zero-length quaternion")
  return value / norm


def wrist_local_points_to_world(points, *, wrist_pos, wrist_quat):
  """Transform wrist-local points using an already-normalized wrist quaternion."""
  wrist_quat = wrist_quat.unsqueeze(1).expand(*points.shape[:-1], 4)
  return quat_apply(wrist_quat, points) + wrist_pos.unsqueeze(1)


def object_velocity_from_pose(object_pose, *, frame_dt: float):
  """Derive velocities from a pose sequence with normalized quaternions.

  Used with the world-frame object pose to produce the absolute velocities that
  the reset injects into the simulation.
  """
  if frame_dt <= 0 or object_pose.ndim != 2 or object_pose.shape[-1] != 7:
    raise ValueError(
      "object_pose must be rank 2 with last dimension 7 and positive frame_dt"
    )
  quat = object_pose[:, 3:]
  angvel = torch.zeros_like(object_pose[:, :3])
  if len(quat) > 1:
    delta = _quat(quat_mul(quat[1:], quat_inv(quat[:-1])), "object_delta_quat")
    delta = torch.where(delta[:, :1] < 0, -delta, delta)
    sin_half = torch.linalg.vector_norm(delta[:, 1:], dim=-1)
    axis = delta[:, 1:] / sin_half.clamp_min(1e-8).unsqueeze(-1)
    step = axis * (
      2 * torch.atan2(sin_half, delta[:, 0].clamp(-1, 1)) / frame_dt
    ).unsqueeze(-1)
    angvel[:-1], angvel[-1] = step, step[-1]
  return _diff(object_pose[:, :3], frame_dt), angvel


def _diff(value: torch.Tensor, dt: float) -> torch.Tensor:
  result = torch.zeros_like(value)
  if len(value) > 1:
    result[:-1] = (value[1:] - value[:-1]) / dt
    result[-1] = result[-2]
  return result
