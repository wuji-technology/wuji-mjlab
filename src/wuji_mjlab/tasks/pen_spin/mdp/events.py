# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
import math

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.dr.geom import _recompute_geom_bounds
from mjlab.envs.mdp.events import apply_body_impulse, resolve_env_ids
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  combine_frame_transforms,
  quat_apply,
  quat_apply_inverse,
  quat_from_angle_axis,
  quat_mul,
  quat_unique,
  sample_uniform,
)

from wuji_mjlab.utils.curriculum import get_curriculum_value

from .observations import PEN_SHAFT_AXIS

_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")


def _gravity_magnitude(env) -> float:
  return float(torch.as_tensor(env.sim.model.opt.gravity).reshape(-1)[:3].norm())


@requires_model_fields(
  "geom_size",
  "geom_pos",
  "geom_rbound",
  "geom_aabb",
  "body_inertia",
  "body_iquat",
  recompute=RecomputeLevel.set_const_0,
)
def randomize_pen_dimensions(
  env,
  env_ids,
  diameter_range_m=(0.012, 0.018),
  total_length_range_m=(0.200, 0.240),
  asset_cfg=SceneEntityCfg(
    "object",
    geom_names=("cube", "A_bottom", "A_top"),
    body_names=("cube",),
    preserve_order=True,
  ),
):
  """Independent uniform dimensions per world; keep the two marker boxes fixed.

  Run after mass DR. Recompute inertia for the current total mass using a uniform
  volume distribution across the shaft and two boxes. Keep the independent COM
  offset DR. No mass changes and no accumulation on repeated calls.
  """
  for name, bounds, minimum in (
    ("diameter", diameter_range_m, 0.0),
    ("total length", total_length_range_m, 0.072),
  ):
    if (
      len(bounds) != 2
      or not all(math.isfinite(v) for v in bounds)
      or not minimum < bounds[0] <= bounds[1]
    ):
      raise ValueError(f"invalid {name} range: {bounds}")
  env_ids = resolve_env_ids(env, env_ids)
  asset = env.scene[asset_cfg.name]
  gids = asset.indexing.geom_ids[asset_cfg.geom_ids].long()
  bids = asset.indexing.body_ids[asset_cfg.body_ids].long()
  if len(gids) != 3 or len(bids) != 1:
    raise ValueError("pen DR requires shaft, bottom box, top box and one body")
  model = env.sim.model
  radius = torch.empty(len(env_ids), device=env.device).uniform_(*diameter_range_m) / 2
  length = torch.empty_like(radius).uniform_(*total_length_range_m)
  shaft_length = length - 0.072
  half_shaft = shaft_length / 2
  end_center = length / 2 - 0.018
  shaft, bottom, top = gids.unbind()
  model.geom_size[env_ids, shaft, 0] = radius
  model.geom_size[env_ids, shaft, 1] = half_shaft
  # Boxes are rotated with the canonical asset; their local sizes stay 9/9/18 mm.
  model.geom_pos[env_ids, bottom, PEN_SHAFT_AXIS] = -end_center
  model.geom_pos[env_ids, top, PEN_SHAFT_AXIS] = end_center
  _recompute_geom_bounds(env, env_ids, asset_cfg)

  body = bids[0]
  mass = model.body_mass[env_ids % model.body_mass.shape[0], body]
  shaft_volume = torch.pi * radius.square() * shaft_length
  box_volume = 0.018 * 0.018 * 0.036
  density = mass / (shaft_volume + 2 * box_volume)
  shaft_mass, box_mass = density * shaft_volume, density * box_volume
  axial = shaft_mass * radius.square() / 2 + 2 * box_mass * (0.018**2 + 0.018**2) / 12
  transverse = shaft_mass * (3 * radius.square() + shaft_length.square()) / 12
  transverse += 2 * box_mass * ((0.018**2 + 0.036**2) / 12 + end_center.square())
  model.body_inertia[env_ids, body] = torch.stack(
    (axial, transverse, transverse), dim=-1
  )
  model.body_iquat[env_ids, body] = torch.tensor(
    [1.0, 0.0, 0.0, 0.0], device=env.device
  )


def reset_to_reference_state(
  env,
  env_ids,
  command_name="motion",
  object_reset_axial_slide_range=(0.0, 0.0),
  object_reset_roll_range=(0.0, 0.0),
  curriculum_term=None,
):
  """Reset hand and pen onto one sampled reference frame.

  Noise is confined to the two pen motions that leave every finger-shaft
  distance unchanged: a slide along the shaft and a roll about it. A world-frame
  offset or tilt starts the pen inside the fingers and the contact solver throws
  it out. Off-axis robustness comes from the step-mode wrench instead.
  """
  motion = env.command_manager.get_term(command_name)
  motion.prepare_reset(env_ids)
  if curriculum_term is not None:
    # Zero as the missing-state fallback keeps the very first reset inside the warmup.
    scale = get_curriculum_value(env, curriculum_term, 0.0)
    object_reset_axial_slide_range = tuple(
      float(value) * scale for value in object_reset_axial_slide_range
    )
    object_reset_roll_range = tuple(
      float(value) * scale for value in object_reset_roll_range
    )
  apply_reference_state(
    env,
    env_ids,
    motion,
    object_axial_slide_range=object_reset_axial_slide_range,
    object_roll_range=object_reset_roll_range,
  )


def apply_reference_state(
  env,
  env_ids,
  motion,
  object_axial_slide_range=(0.0, 0.0),
  object_roll_range=(0.0, 0.0),
):
  """Write the command's current reference frame into the simulation."""

  def ref(value):
    return value[env_ids]

  robot = env.scene["robot"]
  obj = env.scene["object"]
  wrist_pose = ref(motion.wrist_pose_w)
  robot.write_mocap_pose_to_sim(wrist_pose, env_ids=env_ids)
  robot.write_joint_state_to_sim(
    ref(motion.joint_pos),
    ref(motion.joint_vel),
    joint_ids=_reference_joint_ids(robot, motion, device=env.device),
    env_ids=env_ids,
  )

  pose = ref(motion.object_pose)
  pos, quat = combine_frame_transforms(
    wrist_pose[:, :3], wrist_pose[:, 3:], pose[:, :3], pose[:, 3:]
  )
  shaft_axis = torch.zeros_like(pos)
  shaft_axis[:, PEN_SHAFT_AXIS] = 1.0
  slide = torch.empty((len(env_ids), 1), device=env.device, dtype=pos.dtype).uniform_(
    *object_axial_slide_range
  )
  roll = torch.empty(len(env_ids), device=env.device, dtype=pos.dtype).uniform_(
    *object_roll_range
  )
  pos = pos + quat_apply(quat, shaft_axis * slide)
  quat = quat_unique(quat_mul(quat, quat_from_angle_axis(roll, shaft_axis)))

  # Object pose and velocity must use one consistent world-frame state.
  root_state = torch.cat(
    (pos, quat, ref(motion.object_linvel_w), ref(motion.object_angvel_w)),
    -1,
  )
  obj.write_root_state_to_sim(root_state, env_ids=env_ids)


def _reference_joint_ids(robot, motion, *, device):
  joint_names = tuple(motion.joint_names)
  joint_ids, found_names = robot.find_joints(
    joint_names,
    preserve_order=True,
  )
  if tuple(found_names) != joint_names:
    raise ValueError(
      f"robot joints do not match command joint_names: expected {joint_names}, found {tuple(found_names)}"
    )
  return torch.tensor(joint_ids, device=device, dtype=torch.long)


class apply_object_disturbance_wrench:
  """Apply a curriculum-scaled intermittent 6D wrench to the object."""

  def __init__(self, cfg, env) -> None:
    asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_OBJECT_CFG)
    asset: Entity = env.scene[asset_cfg.name]
    self._body_ids = asset_cfg.body_ids
    self._global_body_ids = asset.indexing.body_ids[self._body_ids].long()
    self._num_bodies = (
      len(self._body_ids)
      if isinstance(self._body_ids, (list, tuple))
      else asset.num_bodies
    )
    self._num_envs = env.num_envs
    self._device = env.device
    shape = (self._num_envs, self._num_bodies, 3)
    self._force = torch.zeros(shape, device=self._device)
    self._torque = torch.zeros(shape, device=self._device)
    self._phase_active = torch.zeros(
      self._num_envs, device=self._device, dtype=torch.bool
    )
    self._phase_time_left = torch.zeros(self._num_envs, device=self._device)
    self._decay_rate = torch.ones(self._num_envs, device=self._device)
    self._gravity_mag = _gravity_magnitude(env)

  def __call__(
    self,
    env,
    env_ids,
    *,
    grav_mult_range: tuple[float, float],
    angular_accel: float,
    active_s_range: tuple[float, float],
    silence_s_range: tuple[float, float],
    decay_to_frac: float,
    warmup_time_s: float,
    curriculum_term: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  ) -> None:
    """``grav_mult_range`` draws the linear acceleration per pulse in multiples
    of g; the angular channel follows the same draw with ``angular_accel`` as
    the peak, keeping the force/torque ratio constant.

    The pulse state machine runs continuously after the episode warmup."""
    del env_ids
    device = self._device
    asset: Entity = env.scene[asset_cfg.name]
    step_dt = float(env.step_dt)
    scale = get_curriculum_value(env, curriculum_term, 0.0)
    angular_accel = float(angular_accel) * scale

    episode_time = env.episode_length_buf.float() * step_dt
    warmup_mask = episode_time < float(warmup_time_s)
    disturb_mask = ~warmup_mask
    self._phase_time_left[disturb_mask] -= step_dt
    expired = disturb_mask & (self._phase_time_left <= 0.0)
    was_active = self._phase_active.clone()

    continuing = was_active & disturb_mask
    if continuing.any():
      rate = self._decay_rate[continuing].view(-1, 1, 1)
      self._force[continuing] *= rate
      self._torque[continuing] *= rate

    to_active = expired & (~was_active)
    if to_active.any():
      ids = to_active.nonzero(as_tuple=False).squeeze(-1)
      body_mass = self._selected_body_field(env, ids, "body_mass")
      force_dir = self._random_unit_dirs(ids.numel(), device)
      lo, hi = float(grav_mult_range[0]), float(grav_mult_range[1])
      mult = torch.rand((ids.numel(), 1, 1), device=device) * (hi - lo) + lo
      pulse_linear_accel = mult * (self._gravity_mag * scale)
      pulse_angular_accel = (mult / max(hi, 1e-6)) * angular_accel
      self._force[ids] = force_dir * (pulse_linear_accel * body_mass.unsqueeze(-1))

      body_inertia = self._selected_body_field(env, ids, "body_inertia")
      body_iquat = self._selected_body_field(env, ids, "body_iquat")
      world_quat = asset.data.body_link_quat_w[ids][:, self._body_ids]
      principal_to_world = quat_mul(world_quat, body_iquat)
      target_world = self._random_unit_dirs(ids.numel(), device) * pulse_angular_accel
      torque_principal = body_inertia * quat_apply_inverse(
        principal_to_world, target_world
      )
      self._torque[ids] = quat_apply(principal_to_world, torque_principal)
      self._phase_active[ids] = True
      active_duration = sample_uniform(*map(float, active_s_range), ids.numel(), device)
      self._phase_time_left[ids] = active_duration
      self._decay_rate[ids] = float(decay_to_frac) ** (
        step_dt / active_duration.clamp_min(step_dt)
      )
    to_silence = expired & was_active
    if to_silence.any():
      ids = to_silence.nonzero(as_tuple=False).squeeze(-1)
      self._force[ids] = 0.0
      self._torque[ids] = 0.0
      self._phase_active[ids] = False
      self._phase_time_left[ids] = sample_uniform(
        *map(float, silence_s_range), ids.numel(), device
      )

    force_out = self._force.clone()
    torque_out = self._torque.clone()
    if warmup_mask.any():
      ids = warmup_mask.nonzero(as_tuple=False).squeeze(-1)
      force_out[ids] = 0.0
      torque_out[ids] = 0.0

    asset.write_external_wrench_to_sim(
      force_out,
      torque_out,
      env_ids=None,
      body_ids=asset_cfg.body_ids,
    )

  def reset(self, env_ids=None) -> None:
    idx = slice(None) if env_ids is None else env_ids
    self._force[idx] = 0.0
    self._torque[idx] = 0.0
    self._phase_active[idx] = False
    self._phase_time_left[idx] = 0.0
    self._decay_rate[idx] = 1.0

  def _random_unit_dirs(self, count: int, device) -> torch.Tensor:
    direction = torch.randn((count, self._num_bodies, 3), device=device)
    return direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)

  def _selected_body_field(self, env, ids, field):
    env_grid, body_grid = torch.meshgrid(ids, self._global_body_ids, indexing="ij")
    return getattr(env.sim.model, field)[env_grid, body_grid]


class apply_link_random_force(apply_body_impulse):
  """Mass-normalized random-direction bumps at random points on selected links.

  Per (env, body) on each trigger: an isotropic direction, a magnitude
  ``mult * |gravity| * body_mass * force_scale`` and an application point inside
  the link's geom bounding sphere scaled by ``offset_radius_scale``. Optional
  scale maps use body names, default to one, and never modify model fields.
  Held for ``duration_s``, then zero for ``cooldown_s``.

  Each (env, body) cell runs its own duty-cycle clock; the inherited
  ``apply_body_impulse`` timers are per-env and would fire all five fingertips
  together. ``curriculum_term`` scales the magnitude.
  """

  def __init__(self, cfg, env) -> None:
    super().__init__(cfg, env)
    self._gravity_mag = _gravity_magnitude(env)
    self._global_body_ids = self._asset.indexing.body_ids[self._body_ids].to(
      device=self._device, dtype=torch.long
    )
    self._force_scale = self._body_scale(cfg.params.get("force_scale"), "force_scale")
    self._link_radius = self._compute_link_radius(env) * self._body_scale(
      cfg.params.get("offset_radius_scale"), "offset_radius_scale"
    )  # (num_bodies,)
    # Per-(env, body) timers and wrench buffers replace the inherited per-env ones.
    shape = (self._num_envs, self._num_bodies)
    self._active = torch.zeros(shape, device=self._device, dtype=torch.bool)
    self._time_remaining = torch.zeros(shape, device=self._device)
    self._interval_time_left = torch.zeros(shape, device=self._device)
    self._force = torch.zeros((*shape, 3), device=self._device)
    self._torque = torch.zeros((*shape, 3), device=self._device)

  def _body_scale(self, scales, name) -> torch.Tensor:
    """Resolve coefficients by selected body name, independent of selector order."""
    ids = (
      range(self._asset.num_bodies)[self._body_ids]
      if isinstance(self._body_ids, slice)
      else self._body_ids
    )
    names = [self._asset.body_names[i] for i in ids]
    scales = {} if scales is None else scales
    unknown = scales.keys() - set(names)
    if unknown:
      raise ValueError(f"{name} contains unselected bodies: {sorted(unknown)}")
    values = [float(scales.get(body, 1.0)) for body in names]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
      raise ValueError(f"{name} must contain finite nonnegative coefficients")
    return torch.tensor(values, device=self._device, dtype=torch.float32)

  def _compute_link_radius(self, env) -> torch.Tensor:
    """Per-link offset radius from each link's geom bounding sphere (rbound)."""
    mjm = env.sim.mj_model
    radii = [
      float(mjm.geom_rbound[mjm.geom_bodyid == bid].max())
      for bid in self._global_body_ids.tolist()
    ]
    return torch.tensor(radii, device=self._device, dtype=torch.float32)

  def __call__(
    self,
    env,
    env_ids,
    *,
    grav_mult_range: tuple[float, float],
    duration_s: tuple[float, float],
    cooldown_s: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    force_scale: dict[str, float] | None = None,
    offset_radius_scale: dict[str, float] | None = None,
    curriculum_term: str | None = None,
  ) -> None:
    # Scale maps are bound once at construction; step mode operates on all envs.
    del env_ids, asset_cfg, force_scale, offset_radius_scale
    dt = self._step_dt
    device = self._device

    active, remaining, interval = (
      self._active,
      self._time_remaining,
      self._interval_time_left,
    )

    # Expire active bumps -> zero that cell's wrench and start its cooldown.
    remaining[active] -= dt
    expired = active & (remaining <= 0)
    if expired.any():
      self._force[expired] = 0.0
      self._torque[expired] = 0.0
      active[expired] = False
      remaining[expired] = 0.0
      lo, hi = cooldown_s
      interval[expired] = torch.rand(int(expired.sum()), device=device) * (hi - lo) + lo

    # Advance the cooldown clock on idle cells only.
    interval[~active] -= dt
    fire = (~active) & (interval <= 0)
    if fire.any():
      k = int(fire.sum())

      # Random isotropic direction, one per firing (env, body) cell.
      f_dir = torch.randn((k, 3), device=device)
      f_dir = f_dir / f_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)

      # Magnitude = mult * g * mass, scaled by the curriculum when configured.
      lo, hi = grav_mult_range
      mult = torch.rand((k, 1), device=device) * (hi - lo) + lo
      if curriculum_term is not None:
        mult = mult * get_curriculum_value(env, curriculum_term, 0.0)
      mass = env.sim.model.body_mass[:, self._global_body_ids][fire].unsqueeze(
        -1
      )  # (k,1)
      scale = self._force_scale.expand(self._num_envs, -1)[fire].unsqueeze(-1)
      forces = f_dir * (mult * self._gravity_mag * mass * scale)

      # Random application point inside the link bounding sphere (uniform in ball).
      off = torch.randn((k, 3), device=device)
      off = off / off.norm(dim=-1, keepdim=True).clamp_min(1e-6)
      u = torch.rand((k, 1), device=device) ** (1.0 / 3.0)
      radius = (
        self._link_radius.view(1, -1).expand(self._num_envs, -1)[fire].unsqueeze(-1)
      )
      offset_local = off * u * radius
      body_quat = self._asset.data.body_com_quat_w[:, self._body_ids][fire]  # (k,4)
      offset_w = quat_apply(body_quat, offset_local)

      self._force[fire] = forces
      self._torque[fire] = torch.cross(offset_w, forces, dim=-1)

      dl, dh = duration_s
      remaining[fire] = torch.rand(k, device=device) * (dh - dl) + dl
      active[fire] = True
      lo, hi = cooldown_s
      interval[fire] = torch.rand(k, device=device) * (hi - lo) + lo

    # Independent cells cannot be selected through env_ids/body_ids; idle cells hold zero.
    self._asset.write_external_wrench_to_sim(
      self._force, self._torque, env_ids=None, body_ids=self._body_ids
    )

  def reset(self, env_ids=None) -> None:
    """Clear the per-cell wrench and seed each cell's clock with a random phase.

    Zeroing ``_interval_time_left`` would fire all five tips together on the
    step after every reset.
    """
    ids = slice(None) if env_ids is None else env_ids
    for buf in (self._active, self._time_remaining, self._force, self._torque):
      buf[ids] = 0
    hi = float(self._cooldown_s[1])
    phase = self._interval_time_left[ids]
    self._interval_time_left[ids] = torch.rand_like(phase) * hi

  def debug_vis(self, visualizer) -> None:
    """Draw arrows for the fingertips that are currently bumping.

    The inherited version reads ``self._active[env_idx]`` as a scalar, which the
    per-(env, body) mask is not.
    """
    if not self._active.any():
      return
    viz = self._viz_cfg
    min_sq = viz.min_force * viz.min_force
    wrench = self._asset.data.body_external_wrench  # (nworld, nbody, 6)
    com_pos = self._asset.data.body_com_pos_w
    body_ids = (
      self._body_ids
      if isinstance(self._body_ids, (list, tuple))
      else range(wrench.shape[1])
    )
    for env_idx in visualizer.get_env_indices(self._num_envs):
      for slot, body in enumerate(body_ids):
        if not bool(self._active[env_idx, slot]):
          continue
        force = wrench[env_idx, body, :3]
        if (force * force).sum().item() < min_sq:
          continue
        start = com_pos[env_idx, body].cpu().numpy()
        visualizer.add_arrow(
          start=start,
          end=start + force.cpu().numpy() * viz.scale,
          color=viz.rgba,
          width=viz.width,
        )
