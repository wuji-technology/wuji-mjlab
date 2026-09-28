# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
import numpy as np
import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .observations import (
  joint_position_error,
  link_pos_wrist_local,
  object_axis_point_error,
  object_pose_wrist_local,
)

_ROBOT = SceneEntityCfg("robot")
_OBJECT = SceneEntityCfg("object")


def _gaussian(error: torch.Tensor, sigma: float) -> torch.Tensor:
  return torch.exp(-torch.square(error / sigma))


def object_position_tracking_reward(
  env,
  robot_cfg: SceneEntityCfg = _ROBOT,
  object_cfg: SceneEntityCfg = _OBJECT,
  command_name: str = "motion",
  sigma: float = 0.05,
) -> torch.Tensor:
  current = object_pose_wrist_local(env, robot_cfg, object_cfg)
  target = env.command_manager.get_term(command_name).object_pose
  return _gaussian(torch.linalg.norm(current[:, :3] - target[:, :3], dim=-1), sigma)


def object_axis_point_tracking_reward(
  env,
  robot_cfg: SceneEntityCfg = _ROBOT,
  object_cfg: SceneEntityCfg = _OBJECT,
  command_name: str = "motion",
  sigma: float = 0.04,
) -> torch.Tensor:
  error = object_axis_point_error(env, robot_cfg, object_cfg, command_name).reshape(
    env.num_envs, -1, 3
  )
  return _gaussian(torch.linalg.norm(error, dim=-1).mean(dim=-1), sigma)


def action_rate_combined(env) -> torch.Tensor:
  """Squared first/second differences of action-manager inputs, summed over joints.

  These are dimensionless action differences, before residual scaling and EMA.
  Residual clipping inside the action term does not bound these histories;
  a configured wrapper action clip does, because it runs before the manager.
  The reward manager applies this term's weight and control-step dt separately.
  """
  action = env.action_manager.action
  previous = env.action_manager.prev_action
  previous_previous = env.action_manager.prev_prev_action
  velocity = action - previous
  acceleration = action - 2.0 * previous + previous_previous
  return torch.sum(velocity.square() + acceleration.square(), dim=-1)


class normalized_joint_torques_l2:
  """Sum squared actual actuator torques / nominal XML torque limits.

  Wuji Hand 2 has one gear-1 joint actuator per finger joint, so actuator_force is
  torque in N.m. The limits are the compiled model's nominal ones, so the penalty
  stays relative to the nominal under robot_effort_limits. Samples the last
  physics substep.
  """

  def __init__(self, cfg, env):
    asset_cfg = cfg.params["asset_cfg"]
    self._asset = env.scene[asset_cfg.name]
    self._ids = asset_cfg.actuator_ids
    global_ids = self._asset.indexing.ctrl_ids[self._ids].cpu().numpy()
    model = env.sim.mj_model
    ranges = model.actuator_forcerange[global_ids]
    gears = model.actuator_gear[global_ids]
    import mujoco

    if (
      len(global_ids) == 0
      or not np.all(model.actuator_trntype[global_ids] == mujoco.mjtTrn.mjTRN_JOINT)
      or not np.allclose(gears[:, 0], 1.0)
      or not np.allclose(gears[:, 1:], 0.0)
    ):
      raise ValueError("torque normalization requires nonempty gear-1 joint actuators")
    limits = ranges[:, 1]
    if (
      not np.all(model.actuator_forcelimited[global_ids])
      or not np.isfinite(ranges).all()
      or np.any(limits <= 0)
      or not np.allclose(ranges[:, 0], -limits)
    ):
      raise ValueError(
        "torque normalization requires finite positive symmetric force limits"
      )
    self._limits = torch.tensor(limits.copy(), dtype=torch.float32, device=env.device)

  def __call__(self, env, asset_cfg):
    torque = self._asset.data.actuator_force[:, self._ids]
    return torch.sum(torch.square(torque / self._limits), dim=-1)


class sustained_overcurrent(normalized_joint_torques_l2):
  """Penalize a joint that stays near its current limit, not one that touches it.

  Each joint carries a leaky integrator of its load ratio
  rho = |torque| / (this world's torque limit):

      x <- clip(x + dt * (rho / charge_s - x / leak_s), 0, 1)

  A sustained rho settles at rho * leak_s / charge_s. The penalty is the summed
  excess over ``threshold``, scaled to 1 per fully charged joint.
  """

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._global_ids = self._asset.indexing.ctrl_ids[self._ids]
    self._charge = torch.zeros(env.num_envs, self._limits.numel(), device=env.device)

  def reset(self, env_ids=None):
    self._charge[env_ids if env_ids is not None else slice(None)] = 0.0

  def __call__(self, env, asset_cfg, charge_s: float, leak_s: float, threshold: float):
    # The per-world limit, because robot_effort_limits randomizes it.
    limits = env.sim.model.actuator_forcerange[:, self._global_ids, 1]
    load = (self._asset.data.actuator_force[:, self._ids].abs() / limits).clamp(max=1.0)
    self._charge += env.step_dt * (load / charge_s - self._charge / leak_s)
    self._charge.clamp_(0.0, 1.0)
    return torch.sum((self._charge - threshold).clamp(min=0.0), dim=-1) / (
      1.0 - threshold
    )


def joint_position_tracking_reward(
  env,
  joint_cfg: SceneEntityCfg,
  command_name: str = "motion",
  sigma: float = 0.1,
) -> torch.Tensor:
  error = joint_position_error(env, joint_cfg, command_name)
  robot = env.scene[joint_cfg.name]
  limits = robot.data.joint_pos_limits[:, joint_cfg.joint_ids]
  joint_range = (limits[..., 1] - limits[..., 0]).clamp_min(1e-6)
  return _gaussian(torch.abs(error) / joint_range, sigma).mean(dim=-1)


def link_position_tracking_reward(
  env,
  link_cfg: SceneEntityCfg,
  robot_cfg: SceneEntityCfg = _ROBOT,
  command_name: str = "motion",
  sigma: float = 0.025,
) -> torch.Tensor:
  """Gaussian reward on per-link wrist-local position error against the reference."""
  current = link_pos_wrist_local(env, link_cfg=link_cfg, robot_cfg=robot_cfg)
  motion = env.command_manager.get_term(command_name)
  ref = motion.link_pos.reshape(env.num_envs, -1)
  error = torch.linalg.norm(
    current.reshape(env.num_envs, -1, 3) - ref.reshape(env.num_envs, -1, 3),
    dim=-1,
  )
  return _gaussian(error, sigma).mean(dim=-1)


class finger_self_collision_penalty(ManagerTermBase):
  """Count legacy finger links, merging hand2's distal and fingertip meshes."""

  def __init__(self, cfg, env):
    super().__init__(env)
    self._sensor = env.scene["finger_collision"]
    names = self._sensor.primary_names
    self._middle = [
      names.index(f"right_finger{finger}_link{link}_col")
      for finger in range(1, 6)
      for link in (2, 3)
    ]
    self._distal = [
      names.index(f"right_finger{finger}_link4_col") for finger in range(1, 6)
    ]
    self._tips = [
      names.index(f"right_finger{finger}_tip_sensor_col") for finger in range(1, 6)
    ]

  def __call__(self, env) -> torch.Tensor:
    found = self._sensor.data.found > 0
    middle = found[:, self._middle].float().sum(dim=-1)
    distal = (found[:, self._distal] | found[:, self._tips]).float().sum(dim=-1)
    return middle + distal
