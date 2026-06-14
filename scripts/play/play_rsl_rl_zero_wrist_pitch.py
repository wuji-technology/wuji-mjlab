# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Play wrapper that fixes the reorient robot reset pitch at zero.

This script intentionally mirrors ``scripts/play/play_rsl_rl.py`` and only
changes the play environment config before the runner starts. It is useful for
demoing Revo3 reorient checkpoints without the reset-time wrist pitch sweep.
"""

from __future__ import annotations

import argparse
import math
import sys
from contextlib import contextmanager
from dataclasses import replace
from types import MethodType
from typing import Any, Iterator

import mjlab
import torch
import tyro
import wuji_mjlab.tasks  # noqa: F401
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scripts.play import PlayConfig
from mjlab.tasks.registry import list_tasks
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_mul
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from wuji_mjlab.tasks.reorient.config.revo3_hand.env_cfgs import (
  REVO3_CUBE_OFFSET_IN_PALM,
)
from wuji_mjlab.tasks.reorient.mdp.commands import InHandReorientCommand
from wuji_mjlab.tasks.reorient.mdp.event_utils import resolve_env_ids
from wuji_mjlab.utils.cli_override_utils import (
  DEFAULT_TASK_ALIASES,
  resolve_task,
  split_special_args,
)
from wuji_mjlab.utils.play_runner import run_play_with_cfg
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")
_DEFAULT_ROBOT_PALM_CFG = SceneEntityCfg("robot", body_names=("right_palm",))
_FIXED_OBJECT_ROT_WXYZ = (1.0, 0.0, 0.0, 0.0)
_DEFAULT_REVO3_FIXED_OBJECT_OFFSET_IN_PALM = (0.023, 0.005, 0.075)
_SINGLE_GOAL_SWITCH_DELAY_STEPS = 1_000_000_000
_SINGLE_GOAL_MIN_INTERVAL_S = 1.0e6
_DEFAULT_FIXED_GOAL_AXIS = "z"
_DEFAULT_FIXED_GOAL_DEG = 90.0


def _parse_bool_arg(raw: str) -> bool:
  lowered = raw.lower()
  if lowered in {"1", "true", "t", "yes", "y"}:
    return True
  if lowered in {"0", "false", "f", "no", "n"}:
    return False
  raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{raw}'.")


def _as_offset_tuple(values: Any) -> tuple[float, float, float] | None:
  if values is None:
    return None
  if len(values) != 3:
    raise ValueError(f"Expected 3 cube-offset values, got {values}.")
  return (float(values[0]), float(values[1]), float(values[2]))


def _offset_close(lhs: Any, rhs: tuple[float, float, float]) -> bool:
  lhs_tuple = _as_offset_tuple(lhs)
  if lhs_tuple is None:
    return False
  return all(abs(a - b) <= 1.0e-9 for a, b in zip(lhs_tuple, rhs, strict=True))


def _axis_angle_quat_wxyz(
  axis: str,
  deg: float,
) -> tuple[float, float, float, float]:
  axis = axis.lower()
  if axis not in {"x", "y", "z"}:
    raise ValueError(f"Expected axis to be one of x, y, z. Got '{axis}'.")

  half_angle = math.radians(float(deg)) * 0.5
  sin_half = math.sin(half_angle)
  if axis == "x":
    return (math.cos(half_angle), sin_half, 0.0, 0.0)
  if axis == "y":
    return (math.cos(half_angle), 0.0, sin_half, 0.0)
  return (math.cos(half_angle), 0.0, 0.0, sin_half)


def _force_zero_robot_pitch_reset(env_cfg: Any) -> None:
  event = env_cfg.events.get("reset_robot_pose")
  if event is None:
    raise ValueError("Expected reset_robot_pose event in this reorient play cfg.")
  pose_range = dict(event.params.get("pose_range", {}))
  pose_range["pitch"] = (0.0, 0.0)
  event.params["pose_range"] = pose_range


def _resolve_body_id(asset: Any, cfg: SceneEntityCfg) -> int:
  body_ids = cfg.body_ids
  if isinstance(body_ids, slice):
    resolved, _ = asset.find_bodies(cfg.body_names)
    if not resolved:
      raise ValueError(
        f"No body matched pattern {cfg.body_names} on entity '{cfg.name}'."
      )
    return int(resolved[0])
  return int(body_ids[0])


def _reset_object_pose_in_palm_fixed_rot(
  env: Any,
  env_ids: Any,
  cube_offset_in_palm: tuple[float, float, float],
  object_rot: tuple[float, float, float, float] = _FIXED_OBJECT_ROT_WXYZ,
  pos_noise: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_PALM_CFG,
) -> None:
  """Reset object at a palm-local position with a deterministic orientation."""
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  asset = env.scene[asset_cfg.name]
  robot = env.scene[robot_cfg.name]
  default_state = asset.data.default_root_state[env_ids].clone()

  palm_id = _resolve_body_id(robot, robot_cfg)
  palm_pose_w = robot.data.body_link_pose_w[env_ids, palm_id, :]
  palm_pos_w = palm_pose_w[:, :3]
  palm_quat_w = palm_pose_w[:, 3:7]

  offset = torch.tensor(
    cube_offset_in_palm, device=env.device, dtype=palm_pos_w.dtype
  ).expand(env_ids.numel(), 3)
  if pos_noise > 0.0:
    offset = offset + (
      torch.rand(env_ids.numel(), 3, device=env.device, dtype=palm_pos_w.dtype)
      * 2.0
      - 1.0
    ) * float(pos_noise)

  positions = palm_pos_w + quat_apply(palm_quat_w, offset)
  object_quat = torch.tensor(
    object_rot, device=env.device, dtype=palm_pos_w.dtype
  ).expand(env_ids.numel(), 4)
  default_state[:, 7:] = 0.0

  pose = torch.cat([positions, object_quat], dim=-1)
  asset.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(default_state[:, 7:], env_ids=env_ids)


def _resolve_fixed_object_offset(
  task_id: str,
  env_cfg: Any,
  requested_offset: tuple[float, float, float] | None,
) -> tuple[float, float, float] | None:
  if requested_offset is not None:
    return requested_offset
  if not task_id.startswith("Revo3RightHand_"):
    return None

  event = env_cfg.events.get("reset_object_pose")
  if event is None:
    return None
  current_offset = event.params.get("cube_offset_in_palm")
  if _offset_close(current_offset, REVO3_CUBE_OFFSET_IN_PALM):
    return _DEFAULT_REVO3_FIXED_OBJECT_OFFSET_IN_PALM
  return None


def _force_fixed_object_init(
  env_cfg: Any,
  cube_offset_in_palm: tuple[float, float, float] | None = None,
) -> None:
  event = env_cfg.events.get("reset_object_pose")
  if event is None:
    raise ValueError("Expected reset_object_pose event in this reorient play cfg.")
  if "cube_offset_in_palm" not in event.params:
    raise ValueError(
      "Expected reset_object_pose params to include cube_offset_in_palm."
    )
  if cube_offset_in_palm is not None:
    event.params["cube_offset_in_palm"] = cube_offset_in_palm
  event.func = _reset_object_pose_in_palm_fixed_rot
  event.params["object_rot"] = _FIXED_OBJECT_ROT_WXYZ
  event.params["pos_noise"] = 0.0
  event.params.pop("pos_noise_curriculum_term", None)


def _force_single_reorient_goal(env_cfg: Any) -> None:
  command = env_cfg.commands.get("reorient_command")
  if command is None:
    raise ValueError("Expected reorient_command in this reorient play cfg.")
  command.goal_switch_delay = _SINGLE_GOAL_SWITCH_DELAY_STEPS
  command.min_goal_interval = _SINGLE_GOAL_MIN_INTERVAL_S


class _FixedGoalInHandReorientCommand(InHandReorientCommand):
  """Reorient command variant with a deterministic tag-frame goal."""

  def __init__(
    self,
    cfg: Any,
    env: Any,
    fixed_goal_quat_tag: tuple[float, float, float, float],
  ) -> None:
    self._fixed_goal_quat_tag = fixed_goal_quat_tag
    super().__init__(cfg, env)

  def _sample_goal_in_world(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    palm_quat_w = self.robot.data.body_link_pose_w[env_ids, self.palm_body_id, 3:7]
    tag_in_palm = torch.tensor(
      self.cfg.tag_in_palm_quat, device=self.device, dtype=palm_quat_w.dtype
    ).expand(n, 4)
    tag_rel_goal = torch.tensor(
      self._fixed_goal_quat_tag, device=self.device, dtype=palm_quat_w.dtype
    ).expand(n, 4)
    tag_quat_w = quat_mul(palm_quat_w, tag_in_palm)
    self.goal_quat_w[env_ids] = quat_mul(tag_quat_w, tag_rel_goal)


def _force_fixed_reorient_goal(
  env_cfg: Any,
  *,
  axis: str = _DEFAULT_FIXED_GOAL_AXIS,
  deg: float = _DEFAULT_FIXED_GOAL_DEG,
) -> None:
  command_cfg = env_cfg.commands.get("reorient_command")
  if command_cfg is None:
    raise ValueError("Expected reorient_command in this reorient play cfg.")

  fixed_goal_quat_tag = _axis_angle_quat_wxyz(axis, deg)

  def build_fixed_goal_command(self: Any, env: Any) -> InHandReorientCommand:
    return _FixedGoalInHandReorientCommand(
      self,
      env,
      fixed_goal_quat_tag=fixed_goal_quat_tag,
    )

  command_cfg.build = MethodType(build_fixed_goal_command, command_cfg)


def _tensor_row(tensor: torch.Tensor, env_idx: int = 0) -> list[float]:
  return [float(v) for v in tensor[env_idx].detach().cpu().tolist()]


def _format_vec(values: list[float]) -> str:
  return "(" + ", ".join(f"{v:+.6f}" for v in values) + ")"


def _configured_cube_offset_in_palm(env: Any) -> tuple[float, float, float] | None:
  event = env.unwrapped.cfg.events.get("reset_object_pose")
  if event is None:
    return None
  offset = event.params.get("cube_offset_in_palm")
  if offset is None:
    return None
  return tuple(float(v) for v in offset)


def _format_init_pose(env: Any, *, env_idx: int = 0) -> str:
  unwrapped = env.unwrapped
  robot = unwrapped.scene["robot"]
  obj = unwrapped.scene["object"]

  palm_ids, palm_names = robot.find_bodies(("right_palm",))
  if not palm_ids:
    raise ValueError("Could not find body 'right_palm' for init-pose inspection.")
  palm_id = int(palm_ids[0])
  palm_name = palm_names[0] if palm_names else "right_palm"

  object_pos_w = obj.data.root_link_pos_w
  object_quat_w = obj.data.root_link_quat_w
  palm_pose_w = robot.data.body_link_pose_w[:, palm_id, :]
  palm_pos_w = palm_pose_w[:, :3]
  palm_quat_w = palm_pose_w[:, 3:7]
  object_pos_in_palm = quat_apply_inverse(
    palm_quat_w,
    object_pos_w - palm_pos_w,
  )

  lines = [
    "[init-pose] env_idx = " + str(env_idx),
    "[init-pose] palm_body = " + palm_name,
    "[init-pose] object_pos_w = " + _format_vec(_tensor_row(object_pos_w, env_idx)),
    "[init-pose] object_quat_w = "
    + _format_vec(_tensor_row(object_quat_w, env_idx)),
    "[init-pose] palm_pos_w = " + _format_vec(_tensor_row(palm_pos_w, env_idx)),
    "[init-pose] palm_quat_w = " + _format_vec(_tensor_row(palm_quat_w, env_idx)),
    "[init-pose] object_pos_in_palm = "
    + _format_vec(_tensor_row(object_pos_in_palm, env_idx)),
  ]
  configured_offset = _configured_cube_offset_in_palm(env)
  if configured_offset is not None:
    lines.append(
      "[init-pose] configured_cube_offset_in_palm = "
      + _format_vec(list(configured_offset))
    )
  lines.append("[init-pose] viewer starts paused; press Play/Step to continue.")
  return "\n".join(lines)


@contextmanager
def _inspect_init_pose_and_pause_viewer(enabled: bool) -> Iterator[None]:
  if not enabled:
    yield
    return

  original_native_setup = NativeMujocoViewer.setup
  original_viser_setup = ViserPlayViewer.setup
  printed = False

  def setup_with_inspection(self: Any) -> None:
    nonlocal printed
    original_setup = (
      original_native_setup
      if isinstance(self, NativeMujocoViewer)
      else original_viser_setup
    )
    original_setup(self)
    if not printed:
      print(_format_init_pose(self.env), flush=True)
      printed = True
    self.pause()

  NativeMujocoViewer.setup = setup_with_inspection
  ViserPlayViewer.setup = setup_with_inspection
  try:
    yield
  finally:
    NativeMujocoViewer.setup = original_native_setup
    ViserPlayViewer.setup = original_viser_setup


def run_play_entry(
  task_id: str,
  cfg: PlayConfig,
  overrides: list[str],
  *,
  fixed_object_init: bool = False,
  single_goal: bool = False,
  fixed_goal: bool = False,
  fixed_goal_axis: str = _DEFAULT_FIXED_GOAL_AXIS,
  fixed_goal_deg: float = _DEFAULT_FIXED_GOAL_DEG,
  cube_offset_in_palm: tuple[float, float, float] | None = None,
  inspect_init_pose: bool = False,
) -> None:
  env_cfg, agent_cfg = prepare_task_cfgs(task_id, overrides, play=True)
  _force_zero_robot_pitch_reset(env_cfg)
  if fixed_object_init:
    fixed_object_offset = _resolve_fixed_object_offset(
      task_id,
      env_cfg,
      cube_offset_in_palm,
    )
    _force_fixed_object_init(env_cfg, cube_offset_in_palm=fixed_object_offset)
  if single_goal:
    _force_single_reorient_goal(env_cfg)
  if fixed_goal:
    _force_fixed_reorient_goal(
      env_cfg,
      axis=fixed_goal_axis,
      deg=fixed_goal_deg,
    )
  with _inspect_init_pose_and_pause_viewer(inspect_init_pose):
    run_play_with_cfg(task_id, cfg, env_cfg, agent_cfg)


def main() -> None:
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--task", type=str, default=None)
  parser.add_argument("--fixed-object-init", action="store_true")
  parser.add_argument("--single-goal", action="store_true")
  parser.add_argument("--fixed-goal", action="store_true")
  parser.add_argument(
    "--fixed-goal-axis",
    choices=("x", "y", "z"),
    default=_DEFAULT_FIXED_GOAL_AXIS,
  )
  parser.add_argument("--fixed-goal-deg", type=float, default=_DEFAULT_FIXED_GOAL_DEG)
  parser.add_argument(
    "--cube-offset-in-palm",
    nargs=3,
    type=float,
    metavar=("X", "Y", "Z"),
    default=None,
    help=(
      "Override fixed object init position in palm-local metres. "
      "Revo3 fixed init defaults to the calibrated stable offset "
      f"{_DEFAULT_REVO3_FIXED_OBJECT_OFFSET_IN_PALM}."
    ),
  )
  parser.add_argument("--inspect-init-pose", action="store_true")
  parser.add_argument(
    "--no-terminations",
    nargs="?",
    const=True,
    default=None,
    type=_parse_bool_arg,
  )
  known, remaining = parser.parse_known_args()

  task_from_equals, tyro_args, overrides = split_special_args(remaining)
  chosen_task = known.task or task_from_equals or "reorient"
  task_id = resolve_task(
    chosen_task,
    task_aliases=DEFAULT_TASK_ALIASES,
    all_tasks=list_tasks(),
  )

  cfg = tyro.cli(
    PlayConfig,
    args=tyro_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" --task {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  if known.no_terminations is not None:
    cfg = replace(cfg, no_terminations=known.no_terminations)

  run_play_entry(
    task_id,
    cfg,
    overrides,
    fixed_object_init=known.fixed_object_init,
    single_goal=known.single_goal,
    fixed_goal=known.fixed_goal,
    fixed_goal_axis=known.fixed_goal_axis,
    fixed_goal_deg=known.fixed_goal_deg,
    cube_offset_in_palm=_as_offset_tuple(known.cube_offset_in_palm),
    inspect_init_pose=known.inspect_init_pose,
  )


if __name__ == "__main__":
  main()
