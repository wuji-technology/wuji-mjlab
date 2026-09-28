# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Reorient environment configuration for Wuji Hand."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from wuji_mjlab.tasks.reorient.reorient_constants import REORIENT_PALM_NORMAL_AXIS
from wuji_mjlab.tasks.reorient.reorient_terms import (
  HAND1_DISTAL_FINGER_OBJECT_BODY_NAMES,
  HAND1_FINGER_CONTACT_PARAM_GEOMS,
  HAND1_GEOM_SIZE_DR_GEOMS,
  HAND1_SOFT_PAD_GEOMS,
  HAND1_TIP_COLLISION_GEOMS,
  HAND1_TIP_CONTACT_BODY_NAMES,
  build_reorient_actions,
  build_reorient_commands,
  build_reorient_curriculum,
  build_reorient_events,
  build_reorient_metrics,
  build_reorient_observations,
  build_reorient_rewards,
  build_reorient_sensors,
  build_reorient_terminations,
)

_HISTORY_LENGTH = 3

TIP_SITE_NAMES = (".*_finger[1-5]_tip",)

TIP_BODY_NAMES = (".*_finger[1-5]_link4",)


UNDESIRED_OBJECT_CONTACT_BODIES = (
  ".*_palm_link",
  ".*_finger1_link1",
  ".*_finger2_link1",
  ".*_finger2_link2",
  ".*_finger2_link3",
  ".*_finger3_link1",
  ".*_finger3_link2",
  ".*_finger3_link3",
  ".*_finger4_link1",
  ".*_finger4_link2",
  ".*_finger4_link3",
  ".*_finger5_link1",
  ".*_finger5_link2",
)

_PLAY_DISABLED_EVENTS = frozenset(
  {
    "object_velocity_disturbance",
    "object_disturbance_force",
    "reset_object_disturbance_force",
    "object_com",
    "robot_friction",
    "robot_geom_size",
    "contact_params_palm_thumb",
    "contact_params_fingers",
    "object_size",
    "object_mass",
    "pd_gains",
    "robot_dof_armature",
    "encoder_bias",
    "robot_link_inertia",
    "robot_link_mass",
  }
)


def make_reorient_env_cfg(
  play: bool = False,
  num_envs: int = 8192,
  palm_subtree_body: str = "right_palm_link",
  soft_pad_geoms: tuple[str, ...] = HAND1_SOFT_PAD_GEOMS,
  geom_size_dr_geoms: tuple[str, ...] | None = HAND1_GEOM_SIZE_DR_GEOMS,
  tip_collision_geoms: tuple[str, ...] = HAND1_TIP_COLLISION_GEOMS,
  tip_contact_body_names: tuple[str, ...] = HAND1_TIP_CONTACT_BODY_NAMES,
  distal_finger_object_body_names: tuple[
    str, ...
  ] = HAND1_DISTAL_FINGER_OBJECT_BODY_NAMES,
  finger_contact_param_geoms: tuple[str, ...] = HAND1_FINGER_CONTACT_PARAM_GEOMS,
  cage_up_axis: int = REORIENT_PALM_NORMAL_AXIS,
  robot_tilt_pivot_in_root: tuple[float, float, float] | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create Reorient task configuration.

  Args:
    play: If True, switch to evaluation defaults (small num_envs, longer episode, debug viz, no policy obs noise, all startup DR / interval disturbance events stripped, curriculum cleared).
    geom_size_dr_geoms: Pass ``None`` for mesh collision geoms; ``randomize_geom_size_uniform`` only supports primitives.
    cage_up_axis: Palm-frame axis aligned with the hand's surface normal.
    robot_tilt_pivot_in_root: Wuji Hand 2's mount point in its palm-root frame; reset pitch rotates about world Y through that point when set."""
  observations = build_reorient_observations(
    history_length=_HISTORY_LENGTH,
    tip_body_names=TIP_BODY_NAMES,
  )
  actions = build_reorient_actions()
  commands = build_reorient_commands()
  events = build_reorient_events(
    soft_pad_geoms=soft_pad_geoms,
    geom_size_dr_geoms=geom_size_dr_geoms,
    finger_contact_param_geoms=finger_contact_param_geoms,
    robot_tilt_pivot_in_root=robot_tilt_pivot_in_root,
  )
  rewards = build_reorient_rewards(
    tip_site_names=TIP_SITE_NAMES,
    cage_up_axis=cage_up_axis,
  )
  terminations = build_reorient_terminations()
  curriculum = build_reorient_curriculum()
  metrics = build_reorient_metrics(
    tip_site_names=TIP_SITE_NAMES, palm_normal_axis=cage_up_axis
  )
  sensors = build_reorient_sensors(
    tip_collision_geoms=tip_collision_geoms,
    tip_body_names=TIP_BODY_NAMES,
    distal_finger_object_body_names=distal_finger_object_body_names,
    tip_contact_body_names=tip_contact_body_names,
    undesired_object_contact_bodies=UNDESIRED_OBJECT_CONTACT_BODIES,
    palm_subtree_body=palm_subtree_body,
  )

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=num_envs,
      env_spacing=0.75,
      extent=0.8,
      sensors=sensors,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",
      distance=0.5,
      elevation=-18.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=180,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.01,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=5,
    episode_length_s=50.0,
  )

  if play:
    cfg.scene.num_envs = 4
    cfg.scene.env_spacing = 0.5
    cfg.episode_length_s = 60.0
    cfg.commands["reorient_command"].debug_vis = True
    cfg.observations["policy"].enable_corruption = False
    for name in ("cube_pos_in_tag", "cube_ori_error"):
      cfg.observations["policy"].terms[name].params["injection_prob"] = 0.0
    for key in _PLAY_DISABLED_EVENTS:
      cfg.events.pop(key, None)
    cfg.curriculum.clear()

  return cfg
