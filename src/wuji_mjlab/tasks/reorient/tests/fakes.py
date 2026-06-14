# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import re
from types import SimpleNamespace

import torch


class FakeAsset:
  def __init__(
    self,
    *,
    num_envs: int,
    geom_offset: int,
    num_geoms: int = 3,
    body_offset: int,
    num_bodies: int = 2,
    joint_offset: int,
    num_joints: int = 4,
    device: torch.device,
  ) -> None:
    self.device = device
    self.num_bodies = num_bodies
    self._joint_names = [f"finger_{joint_id}" for joint_id in range(min(num_joints, 2))]
    self._joint_names.extend(
      f"thumb_{joint_id}"
      for joint_id in range(max(0, num_joints - len(self._joint_names)))
    )
    self._geom_names = [f"geom_{geom_id}" for geom_id in range(num_geoms)]
    self._body_names = [f"body_{body_id}" for body_id in range(num_bodies)]
    self.indexing = SimpleNamespace(
      geom_ids=torch.arange(geom_offset, geom_offset + num_geoms, device=device),
      body_ids=torch.arange(body_offset, body_offset + num_bodies, device=device),
      joint_v_adr=torch.arange(joint_offset, joint_offset + num_joints, device=device),
    )
    root_link_quat_w = torch.zeros((num_envs, 4), device=device)
    root_link_quat_w[:, 0] = 1.0
    default_root_state = torch.zeros((num_envs, 13), device=device)
    default_root_state[:, 3] = 1.0
    root_link_pose_w = torch.zeros((num_envs, 7), device=device)
    root_link_pose_w[:, 3] = 1.0
    body_link_pose_w = torch.zeros((num_envs, num_bodies, 7), device=device)
    body_link_pose_w[:, :, 3] = 1.0
    self.data = SimpleNamespace(
      default_root_state=default_root_state,
      root_link_pos_w=torch.zeros((num_envs, 3), device=device),
      root_link_quat_w=root_link_quat_w,
      root_link_pose_w=root_link_pose_w,
      body_link_pose_w=body_link_pose_w,
      root_link_vel_w=torch.zeros((num_envs, 6), device=device),
      default_joint_pos=torch.zeros((num_envs, num_joints), device=device),
      default_joint_vel=torch.zeros((num_envs, num_joints), device=device),
      joint_pos=torch.zeros((num_envs, num_joints), device=device),
      joint_vel=torch.zeros((num_envs, num_joints), device=device),
      soft_joint_pos_limits=torch.stack(
        (
          torch.full((num_envs, num_joints), -1.0, device=device),
          torch.full((num_envs, num_joints), 1.0, device=device),
        ),
        dim=-1,
      ),
      soft_joint_vel_limits=torch.full((num_envs, num_joints), 2.0, device=device),
      joint_vel_limits=torch.full((num_envs, num_joints), 2.5, device=device),
    )
    self.last_written_root_pose = None
    self.last_written_root_velocity = None
    self.last_written_joint_state = None

  def find_joints(self, joint_name: str) -> tuple[list[int], list[str]]:
    if not joint_name:
      return [], []
    pattern = re.compile(joint_name)
    matched = [
      (idx, name)
      for idx, name in enumerate(self._joint_names)
      if pattern.fullmatch(name)
    ]
    if not matched:
      return [], []
    joint_ids, names = zip(*matched, strict=False)
    return list(joint_ids), list(names)

  def find_geoms(self, geom_name: str) -> tuple[list[int], list[str]]:
    if not geom_name:
      return [], []
    if isinstance(geom_name, tuple):
      geom_ids: list[int] = []
      names: list[str] = []
      for pattern in geom_name:
        matched_ids, matched_names = self.find_geoms(pattern)
        geom_ids.extend(matched_ids)
        names.extend(matched_names)
      return geom_ids, names
    if geom_name in (".*palm_.*", ".*finger.*_col"):
      geom_ids = list(range(len(self._geom_names)))
      return geom_ids, [self._geom_names[idx] for idx in geom_ids]
    if geom_name == "cube" and self._geom_names:
      return [0], [self._geom_names[0]]
    pattern = re.compile(geom_name)
    matched = [
      (idx, name)
      for idx, name in enumerate(self._geom_names)
      if pattern.fullmatch(name)
    ]
    if not matched:
      return [], []
    geom_ids, names = zip(*matched, strict=False)
    return list(geom_ids), list(names)

  def find_bodies(self, body_name: str) -> tuple[list[int], list[str]]:
    if not body_name:
      return [], []
    if isinstance(body_name, tuple):
      body_ids: list[int] = []
      names: list[str] = []
      for pattern in body_name:
        matched_ids, matched_names = self.find_bodies(pattern)
        body_ids.extend(matched_ids)
        names.extend(matched_names)
      return body_ids, names
    if body_name == "cube" and self._body_names:
      return [0], [self._body_names[0]]
    pattern = re.compile(body_name)
    matched = [
      (idx, name)
      for idx, name in enumerate(self._body_names)
      if pattern.fullmatch(name)
    ]
    if not matched:
      return [], []
    body_ids, names = zip(*matched, strict=False)
    return list(body_ids), list(names)

  def write_root_link_pose_to_sim(
    self, pose: torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    env_ids = _normalize_env_ids(
      self.data.root_link_pose_w.shape[0], env_ids, self.device
    )
    self.data.root_link_pose_w[env_ids] = pose
    self.data.root_link_pos_w[env_ids] = pose[:, :3]
    self.data.root_link_quat_w[env_ids] = pose[:, 3:7]
    self.last_written_root_pose = (pose.clone(), env_ids.clone())  # type: ignore[union-attr]

  def write_root_link_velocity_to_sim(
    self, velocity: torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    env_ids = _normalize_env_ids(
      self.data.root_link_vel_w.shape[0], env_ids, self.device
    )
    self.data.root_link_vel_w[env_ids] = velocity
    self.last_written_root_velocity = (velocity.clone(), env_ids.clone())  # type: ignore[union-attr]

  def write_joint_state_to_sim(
    self,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    env_ids: torch.Tensor | None = None,
  ) -> None:
    env_ids = _normalize_env_ids(
      self.data.default_joint_pos.shape[0], env_ids, self.device
    )
    self.data.joint_pos[env_ids] = joint_pos
    self.data.joint_vel[env_ids] = joint_vel
    self.last_written_joint_state = (
      joint_pos.clone(),
      joint_vel.clone(),
      env_ids.clone(),  # type: ignore[union-attr]
    )


class FakeScene(dict):
  def __init__(self, *, num_envs: int, device: torch.device) -> None:
    robot = FakeAsset(
      num_envs=num_envs,
      geom_offset=0,
      body_offset=0,
      joint_offset=0,
      device=device,
    )
    obj = FakeAsset(
      num_envs=num_envs,
      geom_offset=robot.indexing.geom_ids.numel(),
      body_offset=robot.indexing.body_ids.numel(),
      joint_offset=robot.indexing.joint_v_adr.numel(),
      num_geoms=1,
      num_bodies=1,
      num_joints=0,
      device=device,
    )
    super().__init__({"robot": robot, "object": obj})
    self.env_origins = torch.zeros((num_envs, 3), device=device)


class FakeSim:
  def __init__(self, *, num_envs: int, scene: FakeScene, device: torch.device) -> None:
    total_geoms = sum(asset.indexing.geom_ids.numel() for asset in scene.values())
    total_bodies = sum(asset.indexing.body_ids.numel() for asset in scene.values())
    total_joint_adrs = sum(
      asset.indexing.joint_v_adr.numel() for asset in scene.values()
    )
    self.per_world_default_fields: tuple[str, ...] = ("geom_friction",)
    self._default_fields = {
      "geom_friction": torch.ones((num_envs, total_geoms, 3), device=device),
      "geom_size": torch.full((total_geoms, 3), 0.05, device=device),
      "body_mass": torch.ones((total_bodies,), device=device),
      "body_inertia": torch.ones((total_bodies, 3), device=device),
      "geom_solref": torch.ones((total_geoms, 2), device=device),
      "geom_solimp": torch.ones((total_geoms, 3), device=device),
      "dof_damping": torch.ones((total_joint_adrs,), device=device),
      "dof_armature": torch.ones((total_joint_adrs,), device=device),
      "dof_frictionloss": torch.ones((total_joint_adrs,), device=device),
      "body_ipos": torch.zeros((total_bodies, 3), device=device),
    }
    self.model = SimpleNamespace(
      geom_friction=self._default_fields["geom_friction"].clone(),
      geom_size=self._default_fields["geom_size"]
      .unsqueeze(0)
      .repeat(num_envs, 1, 1)
      .clone(),
      geom_rbound=torch.ones((num_envs, total_geoms), device=device),
      geom_aabb=torch.zeros((num_envs, total_geoms, 6), device=device),
      body_mass=self._default_fields["body_mass"]
      .unsqueeze(0)
      .repeat(num_envs, 1)
      .clone(),
      body_inertia=self._default_fields["body_inertia"]
      .unsqueeze(0)
      .repeat(num_envs, 1, 1)
      .clone(),
      geom_solref=self._default_fields["geom_solref"]
      .unsqueeze(0)
      .repeat(num_envs, 1, 1)
      .clone(),
      geom_solimp=self._default_fields["geom_solimp"]
      .unsqueeze(0)
      .repeat(num_envs, 1, 1)
      .clone(),
      body_ipos=self._default_fields["body_ipos"]
      .unsqueeze(0)
      .repeat(num_envs, 1, 1)
      .clone(),
      dof_damping=self._default_fields["dof_damping"]
      .unsqueeze(0)
      .repeat(num_envs, 1)
      .clone(),
      dof_armature=self._default_fields["dof_armature"]
      .unsqueeze(0)
      .repeat(num_envs, 1)
      .clone(),
      dof_frictionloss=self._default_fields["dof_frictionloss"]
      .unsqueeze(0)
      .repeat(num_envs, 1)
      .clone(),
    )

  def get_default_field(self, field: str) -> torch.Tensor:
    return self._default_fields[field]


def make_fake_event_env(num_envs: int = 4, device: str = "cpu") -> SimpleNamespace:
  """Build a lightweight scaffold for reorient event-focused tests."""
  torch_device = torch.device(device)
  scene = FakeScene(num_envs=num_envs, device=torch_device)
  sim = FakeSim(num_envs=num_envs, scene=scene, device=torch_device)
  return SimpleNamespace(
    num_envs=num_envs,
    device=torch_device,
    scene=scene,
    sim=sim,
    episode_length_buf=torch.zeros(num_envs, dtype=torch.long, device=torch_device),
    step_dt=0.02,
    max_episode_length=500,
    common_step_counter=0,
    curriculum_manager=SimpleNamespace(_curriculum_state={}),
    termination_manager=SimpleNamespace(
      _term_dones={},
      terminated=torch.zeros(num_envs, dtype=torch.bool, device=torch_device),
    ),
  )


def make_command_manager(term) -> SimpleNamespace:
  """Wrap an object so it behaves like a CommandManager exposing one term."""
  return SimpleNamespace(get_term=lambda _name: term)


def _normalize_env_ids(
  num_envs: int,
  env_ids: torch.Tensor | None,
  device: torch.device,
) -> torch.Tensor:
  if env_ids is None:
    return torch.arange(num_envs, device=device)
  return env_ids.to(device=device, dtype=torch.long)
