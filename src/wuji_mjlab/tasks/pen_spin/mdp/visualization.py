# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import copy

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import combine_frame_transforms, matrix_from_quat
from mjlab.viewer.model_sync import VIEWER_MODEL_FIELDS, sync_model_fields

from .commands import wrist_local_points_to_world


def _numpy_ids(value: torch.Tensor) -> np.ndarray:
  return value.detach().cpu().numpy().astype(np.int64)


def _build_ghost_model(
  command,
  entity_name,
  env_idx,
  *,
  alpha,
  color,
) -> mujoco.MjModel:
  model = copy.deepcopy(command._env.sim.mj_model)
  sim = command._env.sim
  fields = getattr(sim, "expanded_fields", set()) & VIEWER_MODEL_FIELDS
  if fields:
    sync_model_fields(model, sim.model, fields, env_idx)
  model.geom_rgba[:, 3] = 0.0
  geom_ids = _numpy_ids(command._env.scene[entity_name].indexing.geom_ids)
  visual = geom_ids[
    (model.geom_contype[geom_ids] == 0) & (model.geom_conaffinity[geom_ids] == 0)
  ]
  if color is not None:
    model.geom_rgba[visual, :3] = color
  model.geom_rgba[visual, 3] = alpha
  return model


def _hand_ghost_pose(command, model, reference):
  indexing = command._env.scene["robot"].indexing
  joint_q_adr = _numpy_ids(indexing.joint_q_adr)
  joint_pos = reference["joint_pos"][0].detach().cpu().numpy()
  if joint_q_adr.size != joint_pos.size:
    raise ValueError(
      f"robot has {joint_q_adr.size} indexed joints but reference joints has {joint_pos.size}"
    )
  qpos = model.qpos0.copy()
  qpos[joint_q_adr] = joint_pos
  mocap_pos = np.zeros((model.nmocap, 3), dtype=np.float64)
  mocap_quat = np.zeros((model.nmocap, 4), dtype=np.float64)
  mocap_quat[:, 0] = 1.0
  mocap_pos[indexing.mocap_id] = _numpy(reference["wrist_pos"][0])
  mocap_quat[indexing.mocap_id] = _numpy(reference["wrist_quat"][0])
  return qpos, mocap_pos, mocap_quat


def _object_ghost_qpos(command, model, reference):
  indexing = command._env.scene["object"].indexing
  free_q_adr = _numpy_ids(indexing.free_joint_q_adr)
  if free_q_adr.size != 7:
    raise ValueError("object free joint must have seven qpos addresses")
  qpos = model.qpos0.copy()
  qpos[free_q_adr] = reference["object_pose"][0].detach().cpu().numpy()
  return qpos


def _world_reference(command, env_idx):
  env_ids = torch.tensor([env_idx], device=command.device, dtype=torch.long)
  reference = {
    name: getattr(command, name)[env_ids] for name in ("object_pose", "link_pos")
  }
  reference["joint_pos"] = command.joint_pos[env_ids]
  wrist_pose = command.wrist_pose_w[env_ids]
  wrist_pos, wrist_quat = wrist_pose[:, :3], wrist_pose[:, 3:]
  object_pos, object_quat = combine_frame_transforms(
    wrist_pos,
    wrist_quat,
    reference["object_pose"][:, :3],
    reference["object_pose"][:, 3:],
  )
  link_pos = wrist_local_points_to_world(
    reference["link_pos"],
    wrist_pos=wrist_pos,
    wrist_quat=wrist_quat,
  )
  return {
    **reference,
    "wrist_pos": wrist_pos,
    "wrist_quat": wrist_quat,
    "object_pose": torch.cat((object_pos, object_quat), dim=-1),
    "link_pos": link_pos,
  }


def _numpy(value: torch.Tensor) -> np.ndarray:
  return value.detach().cpu().numpy()


def _ghost_models(command, env_idx):
  if not hasattr(command, "_visualization_robot_ghost"):
    command._visualization_robot_ghost = _build_ghost_model(
      command,
      "robot",
      env_idx,
      alpha=command.cfg.viz.hand_alpha,
      color=command.cfg.viz.hand_color,
    )
  if not hasattr(command, "_visualization_object_models"):
    command._visualization_object_models = _build_ghost_model(
      command,
      "object",
      env_idx,
      alpha=command.cfg.viz.object_alpha,
      color=None,
    )
  return command._visualization_robot_ghost, command._visualization_object_models


def _draw_frames(command, visualizer, env_idx, reference) -> None:
  obj = command._env.scene["object"]
  visualizer.add_frame(
    _numpy(obj.data.root_link_pos_w[env_idx]),
    _numpy(matrix_from_quat(obj.data.root_link_quat_w[env_idx])),
    scale=command.cfg.viz.actual_frame_scale,
    label="object actual",
  )
  visualizer.add_frame(
    _numpy(reference["wrist_pos"][0]),
    _numpy(matrix_from_quat(reference["wrist_quat"][0])),
    scale=command.cfg.viz.wrist_frame_scale,
    label="right wrist reference",
  )
  visualizer.add_frame(
    _numpy(reference["object_pose"][0, :3]),
    _numpy(matrix_from_quat(reference["object_pose"][0, 3:])),
    scale=command.cfg.viz.reference_frame_scale,
    label="object reference",
    axis_radius=command.cfg.viz.reference_frame_radius,
    alpha=command.cfg.viz.reference_frame_alpha,
  )


def _draw_reference_object_trace(command, visualizer, env_idx) -> None:
  """Fading world-frame beads for the reference object's future trajectory.

  Enabled when ``cfg.viz.reference_trace_seconds > 0``. The beads and ghost
  share the command reference after the fixed-wrist reset rather than using the
  wrist pose stored in the source data.
  """
  seconds = command.cfg.viz.reference_trace_seconds
  if seconds <= 0.0:
    return
  stride = max(1, int(command.cfg.viz.reference_trace_stride))
  num = max(1, int(seconds / command.group.frame_dt / stride))
  steps = torch.arange(1, num + 1, device=command.device) * stride
  last_frame = command.trajectory_ends[env_idx] - 1
  frames = torch.unique_consecutive(
    (command.frame_ids[env_idx] + steps).clamp_max(last_frame)
  )
  obj = command._frames["object_pose"][frames]
  wrist = command._frames["wrist_pose_w"][frames]
  pos, _ = combine_frame_transforms(wrist[:, :3], wrist[:, 3:], obj[:, :3], obj[:, 3:])
  color = command.cfg.viz.reference_trace_color
  for k, point in enumerate(_numpy(pos + command._env.scene.env_origins[env_idx])):
    visualizer.add_sphere(
      point,
      0.0035,
      (*color, 0.85 * (1.0 - k / num) + 0.1),
      label=f"obj trace +{k}",
    )


def draw_reference_visualization(command, visualizer) -> None:
  for env_idx in visualizer.get_env_indices(command.num_envs):
    reference = _world_reference(command, env_idx)
    robot_model, object_model = _ghost_models(command, env_idx)
    hand_qpos, mocap_pos, mocap_quat = _hand_ghost_pose(command, robot_model, reference)
    visualizer.add_ghost_mesh(
      hand_qpos,
      robot_model,
      mocap_pos=mocap_pos,
      mocap_quat=mocap_quat,
      alpha=command.cfg.viz.hand_alpha,
      label="right hand reference",
    )
    if command.cfg.viz.reference_object_color is not None:
      object_model.geom_rgba[:, :3] = command.cfg.viz.reference_object_color
    visualizer.add_ghost_mesh(
      _object_ghost_qpos(command, object_model, reference),
      object_model,
      alpha=command.cfg.viz.object_alpha,
      label="object reference",
    )
    _draw_reference_object_trace(command, visualizer, env_idx)
    _draw_frames(command, visualizer, env_idx, reference)
    for link_idx, position in enumerate(reference["link_pos"][0]):
      visualizer.add_sphere(
        _numpy(position),
        command.cfg.viz.link_radius,
        command.cfg.viz.link_color,
        label=f"link_{link_idx} reference",
      )
