# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Interactive MuJoCo viewer showing palm-frame axes across wrist orientations."""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import wuji_mjlab.tasks  # noqa: F401  (registers WujiHand_Reorient)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_from_angle_axis,
)
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from wuji_mjlab.tasks.reorient.mdp.observations import tag_pose_w
from wuji_mjlab.tasks.reorient.reorient_constants import (
  REORIENT_CUBE_INIT_POS,
  REORIENT_ROBOT_ROOT_POS,
  REORIENT_ROBOT_ROOT_ROT,
)
from wuji_mjlab.utils.math import random_quat_uniform

TASK_ID = "WujiHand_Reorient"


@dataclass(frozen=True)
class WristCase:
  name: str
  quat_wxyz: tuple[float, float, float, float]


def _quat_axis_angle_wxyz(
  axis: tuple[float, float, float], deg: float
) -> tuple[float, float, float, float]:
  axis_t = torch.tensor([axis], dtype=torch.float32)
  axis_t = axis_t / axis_t.norm(dim=-1, keepdim=True)
  angle_t = torch.tensor([math.radians(deg)], dtype=torch.float32)
  q = quat_from_angle_axis(angle_t, axis_t).squeeze(0).tolist()
  return (q[0], q[1], q[2], q[3])


def build_wrist_cases(num_envs: int) -> list[WristCase]:
  """Return ``num_envs`` distinct wrist orientations covering the SO(3) sweep."""
  torch.manual_seed(0)
  random_q = random_quat_uniform(1, device="cpu").squeeze(0).tolist()
  random_quat = (random_q[0], random_q[1], random_q[2], random_q[3])

  pool: list[WristCase] = [
    WristCase("identity (Zp = Zw)", (1.0, 0.0, 0.0, 0.0)),
    WristCase(
      "+90 pitch (palm-up, Xp = -Zw)",
      _quat_axis_angle_wxyz((0.0, 1.0, 0.0), 90.0),
    ),
    WristCase(
      "+90 roll (Yp = +Zw)",
      _quat_axis_angle_wxyz((1.0, 0.0, 0.0), 90.0),
    ),
    WristCase(
      "+90 yaw (Xp = +Yw)",
      _quat_axis_angle_wxyz((0.0, 0.0, 1.0), 90.0),
    ),
    WristCase(
      "180 pitch (palm-down, Zp = -Zw)",
      _quat_axis_angle_wxyz((0.0, 1.0, 0.0), 180.0),
    ),
    WristCase(
      "+30 pitch",
      _quat_axis_angle_wxyz((0.0, 1.0, 0.0), 30.0),
    ),
    WristCase(
      "default reorient root rot",
      tuple(REORIENT_ROBOT_ROOT_ROT),  # type: ignore[arg-type]
    ),
    WristCase("random SO(3)", random_quat),
  ]
  if num_envs <= len(pool):
    return pool[:num_envs]
  cases = list(pool)
  while len(cases) < num_envs:
    cases.append(pool[(len(cases)) % len(pool)])
  return cases


def make_palm_axes_visualizer(
  env: ManagerBasedRlEnv,
  case_names: list[str],
  base_update_visualizers: Callable[..., None] | None,
):
  """Return a function that draws per-env world frame, gravity, and palm frame."""
  scene = env.scene
  robot = scene["robot"]
  palm_ids, _ = robot.find_bodies(".*_palm_link")
  if not palm_ids:
    raise RuntimeError("could not find palm body via pattern '.*_palm_link'")
  palm_body_id = int(palm_ids[0])

  tag_ids, _ = robot.find_sites(".*_wrist_tag")
  if not tag_ids:
    raise RuntimeError("could not find tag site via pattern '.*_wrist_tag'")

  env_origins_np = scene.env_origins.detach().cpu().numpy().astype(np.float32)
  world_R = np.eye(3, dtype=np.float32)

  palm_axis_len = 0.10
  tag_axis_len = 0.10
  world_axis_len = 0.20
  gravity_len = 0.15

  tag_axis_colors = ((0.0, 0.85, 0.85), (0.85, 0.0, 0.85), (0.9, 0.85, 0.0))

  def update(visualizer) -> None:
    if base_update_visualizers is not None:
      base_update_visualizers(visualizer)

    palm_pose = robot.data.body_link_pose_w[:, palm_body_id, :]
    palm_pos_t = palm_pose[:, 0:3]
    palm_quat_t = palm_pose[:, 3:7]
    palm_pos = palm_pos_t.detach().cpu().numpy()
    palm_R = matrix_from_quat(palm_quat_t).detach().cpu().numpy()

    tag_pos_t, tag_quat_t = tag_pose_w(robot, tag_ids)
    tag_pos = tag_pos_t.detach().cpu().numpy()
    tag_R = matrix_from_quat(tag_quat_t).detach().cpu().numpy()

    for env_idx in range(palm_pos.shape[0]):
      origin = env_origins_np[env_idx]
      pos = palm_pos[env_idx].astype(np.float32)
      R = palm_R[env_idx].astype(np.float32)
      tag_p = tag_pos[env_idx].astype(np.float32)
      tag_M = tag_R[env_idx].astype(np.float32)

      visualizer.add_frame(
        position=origin,
        rotation_matrix=world_R,
        scale=world_axis_len,
        axis_radius=0.0035,
        alpha=0.5,
        label=f"world_env{env_idx}",
      )

      grav_end = origin + np.array([0.0, 0.0, -gravity_len], dtype=np.float32)
      visualizer.add_arrow(
        start=origin,
        end=grav_end,
        color=(0.05, 0.05, 0.05, 0.95),
        width=0.005,
        label=f"gravity_env{env_idx}",
      )

      visualizer.add_cylinder(
        start=origin,
        end=np.array([origin[0], origin[1], pos[2]], dtype=np.float32),
        radius=0.0015,
        color=(0.6, 0.6, 0.6, 0.6),
        label=f"palm_offset_env{env_idx}",
      )

      visualizer.add_frame(
        position=pos,
        rotation_matrix=R,
        scale=palm_axis_len,
        axis_radius=0.005,
        alpha=1.0,
        label=f"palm_env{env_idx}",
      )

      visualizer.add_frame(
        position=tag_p,
        rotation_matrix=tag_M,
        scale=tag_axis_len,
        axis_radius=0.005,
        alpha=1.0,
        axis_colors=tag_axis_colors,
        label=f"tag_env{env_idx}",
      )

      zp_end = pos + 0.06 * R[:, 2]
      visualizer.add_sphere(
        center=zp_end,
        radius=0.006,
        color=(0.0, 0.0, 0.95, 0.85),
        label=f"zp_marker_env{env_idx}",
      )

  return update


def override_wrist_orientations(
  env: ManagerBasedRlEnv,
  cases: list[WristCase],
) -> None:
  """Set each env's robot wrist pose to its target wrist quaternion."""
  scene = env.scene
  robot = scene["robot"]
  num_envs = scene.num_envs

  env_origins = scene.env_origins.to(env.device)
  base_pos = torch.tensor(REORIENT_ROBOT_ROOT_POS, device=env.device).unsqueeze(0)
  positions = env_origins + base_pos
  quats = torch.tensor(
    [c.quat_wxyz for c in cases], dtype=torch.float32, device=env.device
  )
  pose = torch.cat([positions, quats], dim=-1)
  env_ids = torch.arange(num_envs, device=env.device, dtype=torch.int)

  if robot.is_fixed_base and robot.is_mocap:
    robot.write_mocap_pose_to_sim(pose, env_ids=env_ids)
  else:
    robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    zero_vel = torch.zeros((num_envs, 6), device=env.device)
    robot.write_root_link_velocity_to_sim(zero_vel, env_ids=env_ids)

  # The cube has a freejoint and no policy holds it, so it stays caged only in
  # the first frame.
  if "object" in scene.entities:
    cube = scene["object"]
    cube_offset = torch.tensor(REORIENT_CUBE_INIT_POS, device=env.device).unsqueeze(0)
    cube_pos = env_origins + cube_offset
    cube_quat = torch.tensor(
      [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=env.device
    ).expand(num_envs, -1)
    cube_pose = torch.cat([cube_pos, cube_quat], dim=-1)
    cube.write_root_link_pose_to_sim(cube_pose, env_ids=env_ids)
    cube_zero_vel = torch.zeros((num_envs, 6), device=env.device)
    cube.write_root_link_velocity_to_sim(cube_zero_vel, env_ids=env_ids)


def disable_dynamic_resets(env_cfg) -> None:
  """Disable terminations and randomization that would clobber our overrides."""
  env_cfg.terminations = {}
  if hasattr(env_cfg, "events") and isinstance(env_cfg.events, dict):
    for key in (
      "object_disturbance_force",
      "reset_object_disturbance_force",
    ):
      env_cfg.events.pop(key, None)


def run_visualization(
  num_envs: int,
  viewer_choice: str,
  device: str | None,
) -> None:
  device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = num_envs
  disable_dynamic_resets(env_cfg)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  # Viser's per-reward debug-vis GUI reads ``_debug_vis_enabled``; class-based
  # terms (e.g. CageEscapePenalty) don't set it and crash viewer setup.
  for _, func in env.reward_manager.get_visualizable_terms():
    if not hasattr(func, "_debug_vis_enabled"):
      func._debug_vis_enabled = False

  cases = build_wrist_cases(num_envs)
  print("\n[viz] wrist orientations per env:")
  for i, c in enumerate(cases):
    print(
      f"  env {i}: {c.name}  wxyz=({c.quat_wxyz[0]:+.3f}, {c.quat_wxyz[1]:+.3f}, "
      f"{c.quat_wxyz[2]:+.3f}, {c.quat_wxyz[3]:+.3f})"
    )
  print()

  env.reset()
  override_wrist_orientations(env, cases)
  env.sim.forward()

  base_visualizer = getattr(env, "update_visualizers", None)
  env.update_visualizers = make_palm_axes_visualizer(  # type: ignore[method-assign]
    env, [c.name for c in cases], base_visualizer
  )

  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=None)

  action_shape: tuple[int, ...] = wrapped_env.unwrapped.action_space.shape

  class PolicyZero:
    def __call__(self, obs) -> torch.Tensor:
      del obs
      return torch.zeros(action_shape, device=wrapped_env.unwrapped.device)

  policy = PolicyZero()

  if viewer_choice == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = viewer_choice

  print(f"[viz] launching {resolved_viewer} viewer with num_envs={num_envs}")
  print("[viz] tip (viser): in the GUI panel turn on 'Debug Viz' and 'All envs'")
  print("[viz] tip (native): press 'V' to toggle debug vis, 'A' to show all envs")

  if resolved_viewer == "native":
    NativeMujocoViewer(wrapped_env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(wrapped_env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  wrapped_env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--num-envs",
    type=int,
    default=4,
    help="number of envs / wrist orientations (1-8)",
  )
  parser.add_argument(
    "--viewer",
    type=str,
    default="auto",
    choices=("auto", "native", "viser"),
    help="viewer backend",
  )
  parser.add_argument("--device", type=str, default=None)
  args = parser.parse_args()

  if not (1 <= args.num_envs <= 8):
    print(f"[err] --num-envs must be in [1, 8], got {args.num_envs}", file=sys.stderr)
    sys.exit(2)

  run_visualization(
    num_envs=args.num_envs,
    viewer_choice=args.viewer,
    device=args.device,
  )


if __name__ == "__main__":
  main()
