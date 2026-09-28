# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.buffers import CircularBuffer
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_angle_axis,
  quat_inv,
  quat_mul,
  quat_unique,
  subtract_frame_transforms,
)

# The pen body frame: shaft along +X, A_top at +X.
PEN_SHAFT_AXIS = 0

ROBOT = SceneEntityCfg("robot")
OBJECT = SceneEntityCfg("object")


def _reference_frame_pose(entity, entity_cfg):
  if entity_cfg.body_names is None:
    return entity.data.root_link_pos_w, entity.data.root_link_quat_w
  pos = entity.data.body_link_pos_w[:, entity_cfg.body_ids]
  quat = entity.data.body_link_quat_w[:, entity_cfg.body_ids]
  if pos.shape[1] != 1:
    raise ValueError(
      f"robot_cfg must select exactly one tracking reference body, selected {pos.shape[1]}"
    )
  return pos[:, 0], quat[:, 0]


def object_pose_wrist_local(env, robot_cfg=ROBOT, object_cfg=OBJECT):
  robot, obj = env.scene[robot_cfg.name], env.scene[object_cfg.name]
  wrist_pos, wrist_quat = _reference_frame_pose(robot, robot_cfg)
  pos, quat = subtract_frame_transforms(
    wrist_pos,
    wrist_quat,
    obj.data.root_link_pos_w,
    obj.data.root_link_quat_w,
  )
  return torch.cat((pos, quat), -1)


def _axis_points(env, pose, command_name, *, bounded=False):
  motion = env.command_manager.get_term(command_name)
  points = motion.axis_points.to(device=pose.device, dtype=pose.dtype)
  if bounded:
    return bounded_object_pose_to_points_wrist_local(
      pose,
      points,
      position_tanh_scale_m=motion.cfg.object_position_tanh_scale_m,
    ).reshape(env.num_envs, -1)
  return object_pose_to_points_wrist_local(pose, points).reshape(env.num_envs, -1)


def object_axis_points_wrist_local(
  env, robot_cfg=ROBOT, object_cfg=OBJECT, command_name="motion"
):
  return _axis_points(
    env, object_pose_wrist_local(env, robot_cfg, object_cfg), command_name
  )


def object_axis_points_wrist_local_bounded(
  env, robot_cfg=ROBOT, object_cfg=OBJECT, command_name="motion"
):
  return _axis_points(
    env, object_pose_wrist_local(env, robot_cfg, object_cfg), command_name, bounded=True
  )


def link_pos_wrist_local(env, link_cfg, robot_cfg=ROBOT):
  robot, wrist = env.scene[link_cfg.name], env.scene[robot_cfg.name]
  wrist_pos, wrist_quat = _reference_frame_pose(wrist, robot_cfg)
  value = world_points_to_wrist_local(
    robot.data.body_link_pos_w[:, link_cfg.body_ids],
    wrist_pos=wrist_pos,
    wrist_quat=wrist_quat,
  )
  return value.reshape(env.num_envs, -1)


def world_points_to_wrist_local(points, *, wrist_pos, wrist_quat):
  """Transform world-frame points with the selected wrist pose."""
  inv = quat_inv(wrist_quat).unsqueeze(1).expand(*points.shape[:-1], 4)
  return quat_apply(inv, points - wrist_pos.unsqueeze(1))


def joint_position_error(env, joint_cfg, command_name="motion"):
  """``joint_cfg`` must select the command's ``joint_names`` in the command's order."""
  motion = env.command_manager.get_term(command_name)
  return (
    env.scene[joint_cfg.name].data.joint_pos[:, joint_cfg.joint_ids] - motion.joint_pos
  )


def object_position_error(
  env, robot_cfg=ROBOT, object_cfg=OBJECT, command_name="motion"
):
  reference = env.command_manager.get_term(command_name).object_pose
  return object_pose_wrist_local(env, robot_cfg, object_cfg)[:, :3] - reference[:, :3]


def object_shaft_direction_error(
  env, robot_cfg=ROBOT, object_cfg=OBJECT, command_name="motion"
):
  """Angle between the pen shaft and the reference shaft, in the wrist frame.

  The shaft direction, not the full quaternion: roll about the shaft is the one
  pen rotation the fingers barely control, and a geodesic angle spends the
  whole threshold on it.
  """
  reference = env.command_manager.get_term(command_name).object_pose
  current = object_pose_wrist_local(env, robot_cfg, object_cfg)
  shaft_axis = torch.zeros_like(current[:, :3])
  shaft_axis[:, PEN_SHAFT_AXIS] = 1.0
  cosine = torch.sum(
    quat_apply(current[:, 3:], shaft_axis) * quat_apply(reference[:, 3:], shaft_axis),
    dim=-1,
  )
  return torch.acos(cosine.clamp(-1.0, 1.0)).unsqueeze(-1)


def object_axis_point_error(
  env, robot_cfg=ROBOT, object_cfg=OBJECT, command_name="motion"
):
  current = object_axis_points_wrist_local(env, robot_cfg, object_cfg, command_name)
  motion = env.command_manager.get_term(command_name)
  return current - motion.object_axis_points


def link_position_error(env, link_cfg, robot_cfg=ROBOT, command_name="motion"):
  current = link_pos_wrist_local(env, link_cfg, robot_cfg)
  motion = env.command_manager.get_term(command_name)
  return current - motion.link_pos.reshape(env.num_envs, -1)


# Object-point geometry and camera observation model.


def make_axis_points(
  radius_m: float = 0.10, *, device=None, dtype=None
) -> torch.Tensor:
  """The two points on the pen shaft (+X/-X, see ``PEN_SHAFT_AXIS``).

  They carry the pen's position and shaft direction and nothing else: roll about
  the shaft is the one pen rotation the fingers barely control, so the policy
  neither observes it nor is rewarded for it.
  """
  return torch.tensor(
    [
      [radius_m, 0.0, 0.0],
      [-radius_m, 0.0, 0.0],
    ],
    device=device,
    dtype=dtype,
  )


def object_pose_to_points_wrist_local(object_pose, object_points):
  pos = object_pose[..., :3]
  quat = object_pose[..., 3:7]
  batch_shape = object_pose.shape[:-1]
  num_points = object_points.shape[0]

  expanded_quat = quat.unsqueeze(-2).expand(*batch_shape, num_points, 4)
  expanded_points = object_points.expand(*batch_shape, num_points, 3)
  rotated = quat_apply(expanded_quat.reshape(-1, 4), expanded_points.reshape(-1, 3))
  rotated = rotated.reshape(*batch_shape, num_points, 3)
  return pos.unsqueeze(-2) + rotated


def bounded_object_pose_to_points_wrist_local(
  object_pose: torch.Tensor,
  object_points: torch.Tensor,
  *,
  position_tanh_scale_m: float,
) -> torch.Tensor:
  points = object_pose_to_points_wrist_local(object_pose, object_points)
  return position_tanh_scale_m * torch.tanh(points / position_tanh_scale_m)


def _check_range(name, value):
  lo, hi = (float(v) for v in value)
  if hi < lo:
    raise ValueError(f"{name} must be (lo, hi) with hi >= lo, got {value}")
  return lo, hi


class ObjectPoseCameraModel(ManagerTermBase):
  """Bounded wrist-local axis points with calibration noise, latency and dropout.

  Bias is fixed per episode; jitter is sampled per tick. Keep lag_ticks constant
  to avoid introducing extra frame repetitions alongside the dropout model."""

  def __init__(self, cfg, env):
    super().__init__(env)
    p = cfg.params
    self.robot_cfg = p["robot_cfg"]
    self.object_cfg = p["object_cfg"]
    self.command_name = p["command_name"]

    self.bias_position = _check_range("bias_position_m", p["bias_position_m"])
    self.bias_orientation = _check_range(
      "bias_orientation_rad", p["bias_orientation_rad"]
    )
    self.noise_position = _check_range("noise_position_m", p["noise_position_m"])
    self.noise_orientation = _check_range(
      "noise_orientation_rad", p["noise_orientation_rad"]
    )

    lag_min, lag_max = (int(v) for v in p["lag_ticks"])
    if lag_min < 0 or lag_max < lag_min:
      raise ValueError(f"lag_ticks must be 0 <= min <= max, got {p['lag_ticks']}")
    self.lag_min, self.lag_max = lag_min, lag_max

    self.dropout_enter_prob = float(p["dropout_enter_prob"])
    self.dropout_continue_prob = float(p["dropout_continue_prob"])
    self.dropout_max_ticks = int(p["dropout_max_ticks"])
    for name, value in (
      ("dropout_enter_prob", self.dropout_enter_prob),
      ("dropout_continue_prob", self.dropout_continue_prob),
    ):
      if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    if self.dropout_max_ticks < 0:
      raise ValueError(f"dropout_max_ticks must be >= 0, got {self.dropout_max_ticks}")

    n, device = env.num_envs, env.device
    # The oldest frame anyone can ask for: base latency plus a full dropout burst.
    self._max_age = self.lag_max + self.dropout_max_ticks
    self._history = CircularBuffer(
      max_len=self._max_age + 1, batch_size=n, device=device
    )
    self._bias_pos = torch.zeros(n, 3, device=device)
    self._bias_quat = torch.zeros(n, 4, device=device)
    self._bias_quat[:, 0] = 1.0
    self._age = torch.zeros(n, dtype=torch.long, device=device)
    self._holding = torch.zeros(n, dtype=torch.bool, device=device)
    self._hold_ticks = torch.zeros(n, dtype=torch.long, device=device)
    self._reset_ids: torch.Tensor | None = None
    self._resample_bias(slice(None))

  # ----- state ------------------------------------------------------------

  def _resample_bias(self, ids) -> None:
    """One constant calibration error per episode."""
    pos = self._bias_pos[ids]
    if self.bias_position != (0.0, 0.0):
      pos.uniform_(*self.bias_position)
    else:
      pos.zero_()
    self._bias_pos[ids] = pos

    quat = self._bias_quat[ids]
    if self.bias_orientation != (0.0, 0.0):
      angle = torch.empty(len(quat), device=quat.device, dtype=quat.dtype)
      angle.uniform_(*self.bias_orientation)
      axis = torch.randn(len(quat), 3, device=quat.device, dtype=quat.dtype)
      axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
      quat = quat_unique(quat_from_angle_axis(angle, axis))
    else:
      quat = torch.zeros_like(quat)
      quat[:, 0] = 1.0
    self._bias_quat[ids] = quat

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    batch_ids = None if isinstance(ids, slice) else ids
    self._history.reset(batch_ids=batch_ids)
    self._age[ids] = 0
    self._holding[ids] = False
    self._hold_ticks[ids] = 0
    self._resample_bias(ids)
    # Explicit partial reset recomputes observations in the same control
    # tick. Auto-reset instead computes them once after the new step.
    self._reset_ids = (
      batch_ids
      if batch_ids is not None
      and len(batch_ids) < self.num_envs
      and not self._env.cfg.auto_reset
      else None
    )
    return {}

  # ----- one observation --------------------------------------------------

  def __call__(self, env, **kwargs) -> torch.Tensor:
    del kwargs  # every parameter was bound at construction
    pose = object_pose_wrist_local(env, self.robot_cfg, self.object_cfg).clone()

    # What the camera measures: the true pose through a biased frame, plus jitter.
    pose[:, :3] += self._bias_pos
    pose[:, 3:] = quat_mul(self._bias_quat, pose[:, 3:])
    if self.noise_position != (0.0, 0.0):
      pose[:, :3] += torch.empty_like(pose[:, :3]).uniform_(*self.noise_position)
    if self.noise_orientation != (0.0, 0.0):
      angle = torch.empty(len(pose), device=pose.device, dtype=pose.dtype)
      angle.uniform_(*self.noise_orientation)
      axis = torch.randn_like(pose[:, :3])
      axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
      pose[:, 3:] = quat_mul(quat_from_angle_axis(angle, axis), pose[:, 3:])
    pose[:, 3:] = quat_unique(pose[:, 3:])

    reset_ids = self._reset_ids
    self._reset_ids = None
    if reset_ids is not None and self._history.is_initialized:
      self._history.backfill(pose, reset_ids)
      return _axis_points(
        env, self._history[self._age], self.command_name, bounded=True
      )
    self._history.append(pose)

    # What the control loop reads. Latency and dropout are both the age of the
    # frame on hand. A stream is monotonic, so the age is advanced, not redrawn.
    if self.dropout_enter_prob > 0.0:
      roll = torch.rand(env.num_envs, device=pose.device)
      capped = self._hold_ticks >= self.dropout_max_ticks
      holding = torch.where(
        self._holding,
        (roll < self.dropout_continue_prob) & ~capped,
        roll < self.dropout_enter_prob,
      )
      self._hold_ticks = torch.where(
        holding, self._hold_ticks + 1, torch.zeros_like(self._hold_ticks)
      )
      self._holding = holding
    else:
      holding = torch.zeros(env.num_envs, dtype=torch.bool, device=pose.device)

    aged = self._age + 1
    if self.lag_max > 0:
      fresh = torch.randint(
        self.lag_min,
        self.lag_max + 1,
        (env.num_envs,),
        dtype=torch.long,
        device=pose.device,
      )
    else:
      fresh = torch.zeros(env.num_envs, dtype=torch.long, device=pose.device)
    # Holding: the same frame, one tick older. Otherwise a new frame at the base
    # latency, never older than keeping the last one.
    age = torch.where(holding, aged, torch.minimum(fresh, aged))
    # A just-reset row has no history to reach back into yet.
    self._age = torch.minimum(age, self._history.current_length - 1).clamp_min(0)

    return _axis_points(env, self._history[self._age], self.command_name, bounded=True)
