# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Term-group builders for the reorient task.

``reorient_env_cfg.make_reorient_env_cfg`` calls these ``build_*`` functions
to assemble a ``ManagerBasedRlEnvCfg``. Splitting the per-group construction
out of the env-cfg keeps the top-level assembly file small and makes
individual term groups easy to inspect, override, or unit-test.

Robot-binding note: this module owns the full reorient task design,
including decisions that conceptually belong at the robot-binding layer
(``config/wuji_hand/env_cfgs.py``):

- 7 contact sensors (``build_reorient_sensors``) — patterns include
  the hardcoded ``right_palm_link`` subtree filter for ``finger_collision``.
- The ``palm_detach`` reward and the boosted ``finger_collision`` weight.
- The 2-group startup contact_params DR split, grouped by hand anatomy:
  ``contact_params_palm_thumb`` for the 5 geoms of the continuous
  silicone-pad compliance zone (palm + entire thumb: link2_col,
  link2_softbody, link3, link4), and ``contact_params_fingers`` for the
  12 geoms of the rigid grasping fingers (fingers 2-5, link2-4). Width
  × (2.0, 5.0) on the soft-pad group, width × (1.0, 2.0) on the fingers
  group; both events carry solimp_dmin_range=(0.5, 1.0). The thumb
  fingertip's link4 mesh belongs in the soft-pad zone with the rest of
  the thumb, not in a separate fingertip bucket.

The Wuji Hand right-hand asset is therefore baked in here. A second
robot binding (e.g. left hand) would need either a parallel terms module
or an explicit override at the binding site.
"""

from __future__ import annotations

import math

from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, SensorCfg
from mjlab.utils.noise import GaussianNoiseCfg as Gnoise
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from wuji_mjlab.tasks.reorient import mdp
from wuji_mjlab.tasks.reorient.mdp.commands import InHandReorientCommandCfg


def build_reorient_observations(
  history_length: int,
  tip_body_names: tuple[str, ...],
) -> dict[str, ObservationGroupCfg]:
  """Build policy and critic observation groups."""
  policy_terms = {
    "noisy_joint_angles": ObservationTermCfg(
      func=mdp.joint_pos_limit_normalized,
      noise=Unoise(n_min=-0.06, n_max=0.06),
      history_length=history_length,
    ),
    "qpos_error": ObservationTermCfg(
      func=mdp.joint_pos_target_error,
      history_length=history_length,
    ),
    # Absolute cube position in tag frame (not a delta from a reference).
    "cube_pos_in_tag": ObservationTermCfg(
      func=mdp.cube_pos_in_tag,
      params={"injection_prob": 0.02},
      noise=Unoise(n_min=-0.008, n_max=0.008),
      history_length=history_length,
    ),
    "cube_ori_error": ObservationTermCfg(
      func=mdp.goal_rot_err_6d,
      params={
        "command_name": "reorient_command",
        "object_cfg": SceneEntityCfg("object"),
        "injection_prob": 0.02,
      },
      noise=Gnoise(std=0.05),
      history_length=history_length,
    ),
    "action_history": ObservationTermCfg(
      func=mdp.previous_raw_action,
      history_length=history_length,
    ),
  }

  critic_terms = {
    "joint_angles": ObservationTermCfg(
      func=mdp.joint_pos_limit_normalized,
      history_length=history_length,
    ),
    "qpos_error": ObservationTermCfg(
      func=mdp.joint_pos_target_error,
      history_length=history_length,
    ),
    "cube_pos_in_tag": ObservationTermCfg(
      func=mdp.cube_pos_in_tag,
      history_length=history_length,
    ),
    "cube_ori_error": ObservationTermCfg(
      func=mdp.goal_rot_err_6d,
      params={
        "command_name": "reorient_command",
        "object_cfg": SceneEntityCfg("object"),
      },
      history_length=history_length,
    ),
    "action_history": ObservationTermCfg(
      func=mdp.previous_raw_action,
      history_length=history_length,
    ),
    # Privileged observations
    "true_joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      history_length=history_length,
    ),
    "true_joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      history_length=history_length,
    ),
    "fingertip_positions": ObservationTermCfg(
      func=mdp.body_pos_rel,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=tip_body_names),
        "offset": (0.0, 0.0, 0.5),
      },
      scale=5.0,
      history_length=history_length,
    ),
    "cube_pos_in_tag_clean": ObservationTermCfg(
      func=mdp.cube_pos_in_tag,
    ),
    "cube_ori_error_clean": ObservationTermCfg(
      func=mdp.goal_rot_err_6d,
      params={
        "command_name": "reorient_command",
        "object_cfg": SceneEntityCfg("object"),
      },
    ),
    "cube_linvel": ObservationTermCfg(
      func=mdp.root_lin_vel_w,
      params={"asset_cfg": SceneEntityCfg("object")},
    ),
    "cube_angvel": ObservationTermCfg(
      func=mdp.root_ang_vel_w,
      params={"asset_cfg": SceneEntityCfg("object")},
    ),
    "pert_dir": ObservationTermCfg(
      func=mdp.perturbation_direction,
    ),
    "pert_velocity": ObservationTermCfg(
      func=mdp.perturbation_velocity,
    ),
    "state_progress": ObservationTermCfg(
      func=mdp.command_state_progress,
      params={"command_name": "reorient_command"},
    ),
    "cage_counter": ObservationTermCfg(
      func=mdp.cage_counter_progress,
      params={"max_outside_steps": 10},
    ),
    "dr_params": ObservationTermCfg(
      func=mdp.dr_params_privileged,
      params={
        "robot_cfg": SceneEntityCfg(
          "robot",
          geom_names=(".*palm_.*", ".*finger.*_col"),
          actuator_names=(".*",),
          joint_names=(".*",),
        ),
        "object_cfg": SceneEntityCfg(
          "object",
          body_names=("cube",),
          geom_names=("cube",),
        ),
      },
    ),
    "palm_rot_6d": ObservationTermCfg(
      func=mdp.palm_rot_6d_w,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
  }

  return {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }


def build_reorient_sensors(
  tip_collision_geoms: tuple[str, ...],
  tip_body_names: tuple[str, ...],
  undesired_object_contact_bodies: tuple[str, ...],
) -> tuple[SensorCfg, ...]:
  """Build the 7 contact sensors required by the reorient task.

  Hardcoded Wuji Hand right-hand: ``finger_collision`` filters against
  the ``right_palm_link`` subtree.
  """
  tip_object_contact = ContactSensorCfg(
    name="tip_object_contact",
    primary=ContactMatch(mode="geom", pattern=tip_collision_geoms, entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="cube", entity="object"),
    fields=("found", "force"),
    reduce="mindist",
    num_slots=1,
  )
  palm_object_contact = ContactSensorCfg(
    name="palm_object_contact",
    primary=ContactMatch(mode="body", pattern=".*_palm_link", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="cube", entity="object"),
    fields=("force",),
    reduce="netforce",
    num_slots=1,
  )
  # Used by palm_detach: cube clear of palm/proximal AND touching distal fingers.
  palm_object_found = ContactSensorCfg(
    name="palm_object_found",
    primary=ContactMatch(
      mode="body",
      pattern=(".*_palm_link", ".*_finger.*_link[12]"),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="cube", entity="object"),
    fields=("found",),
    reduce="mindist",
    num_slots=1,
  )
  distal_finger_object_found = ContactSensorCfg(
    name="distal_finger_object_found",
    primary=ContactMatch(
      mode="body",
      pattern=(".*_finger.*_link[34]",),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="cube", entity="object"),
    fields=("found",),
    reduce="mindist",
    num_slots=1,
  )
  undesired_object_contact = ContactSensorCfg(
    name="undesired_object_contact",
    primary=ContactMatch(
      mode="body", pattern=undesired_object_contact_bodies, entity="robot"
    ),
    secondary=ContactMatch(mode="body", pattern="cube", entity="object"),
    fields=("force",),
    reduce="netforce",
    num_slots=1,
  )
  robot_contact = ContactSensorCfg(
    name="robot_contact",
    primary=ContactMatch(mode="body", pattern=tip_body_names, entity="robot"),
    fields=("force",),
    reduce="netforce",
    num_slots=1,
  )
  finger_collision = ContactSensorCfg(
    name="finger_collision",
    primary=ContactMatch(mode="geom", pattern=(".*finger.*_col",), entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="right_palm_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=5,
  )
  return (
    tip_object_contact,
    palm_object_contact,
    palm_object_found,
    distal_finger_object_found,
    undesired_object_contact,
    robot_contact,
    finger_collision,
  )


def build_reorient_actions() -> dict[str, ActionTermCfg]:
  """Build the action term group."""
  return {
    "joint_pos": mdp.JointPositionOffsetEMAActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      action_scale=0.5,
      ema_alpha=0.5,
      warmup_time_s=0.4,
    )
  }


def build_reorient_commands() -> dict[str, CommandTermCfg]:
  """Build the command term group."""
  return {
    "reorient_command": InHandReorientCommandCfg(
      entity_name="object",
      resampling_time_range=(1.0e6, 1.0e6),
      success_threshold=0.2,
      success_hold_steps=5,
      goal_switch_delay=20,
      min_goal_interval=0.0,
      debug_vis=False,
    )
  }


def build_reorient_events() -> dict[str, EventTermCfg]:
  """Build all event terms (resets, intervals, startup DR)."""
  return {
    "reset_robot_pose": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "pitch": (-0.4, 0.1),
        },
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_within_limits_range,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "position_range": {
          ".*_joint1": (-0.3, -0.1),
          ".*_joint2": (0.0, 0.0),
          ".*_joint3": (-0.3, -0.1),
          ".*_joint4": (-0.3, -0.1),
        },
        "velocity_range": {},
        "use_default_offset": True,
        "operation": "abs",
      },
    ),
    "reset_object_pose": EventTermCfg(
      func=mdp.reset_object_orientation,
      mode="reset",
      params={
        "pos_noise": 0.01,
        "asset_cfg": SceneEntityCfg("object"),
      },
    ),
    "object_disturbance_force": EventTermCfg(
      func=mdp.apply_velocity_disturbance,
      mode="interval",
      interval_range_s=(0.6, 1.8),
      params={
        "min_speed": 0.05,
        "max_speed": 0.15,
        "warmup_time_s": 3.0,
        "warmup_frac": 0.05,
        "rampup_frac": 0.80,
        "adaptive_curriculum_term": "adaptive_episode",
      },
    ),
    "reset_object_disturbance_force": EventTermCfg(
      func=mdp.reset_disturbance_caches,
      mode="reset",
    ),
    "reset_joint_acc_cache": EventTermCfg(
      func=mdp.reset_joint_acc_cache,
      mode="reset",
    ),
    # Domain randomization (startup)
    "object_com": EventTermCfg(
      mode="startup",
      func=mdp.dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("object", body_names=("cube",)),
        "operation": "add",
        "ranges": (-0.003, 0.003),
      },
    ),
    "robot_friction": EventTermCfg(
      mode="startup",
      func=mdp.dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          geom_names=(".*palm_.*", ".*finger.*_col"),
        ),
        "operation": "scale",
        "ranges": (0.7, 1.3),
      },
    ),
    "robot_geom_size": EventTermCfg(
      mode="startup",
      func=mdp.randomize_geom_size_uniform,
      params={
        # randomize_geom_size_uniform rejects mesh geoms; restrict to the 10 primitive capsules.
        "asset_cfg": SceneEntityCfg(
          "robot",
          geom_names=(r".*finger[1-5]_link[2-3]_col",),
        ),
        "scale_range": (0.97, 1.03),
      },
    ),
    # 2-group contact DR: thumb+palm (soft 2-5mm) vs fingers 2-5 (rigid 1-2mm).
    "contact_params_palm_thumb": EventTermCfg(
      mode="startup",
      func=mdp.randomize_contact_params,
      params={
        "robot_cfg": SceneEntityCfg(
          "robot",
          geom_names=(
            "right_palm_collision",
            "right_finger1_link2_col",
            "right_finger1_link2_softbody_col",
            "right_finger1_link3_col",
            "right_finger1_link4_col",
          ),
        ),
        "solref_timeconst_range": (1.0, 2.0),
        "solref_dampratio_range": (0.8, 1.2),
        # 2-5 mm soft-zone: the lower end overlaps the finger event's
        # max so a worn pad behaves like a stiff finger.
        "solimp_width_range": (2.0, 5.0),
        "solimp_dmin_range": (0.5, 1.0),
      },
    ),
    "contact_params_fingers": EventTermCfg(
      mode="startup",
      func=mdp.randomize_contact_params,
      params={
        "robot_cfg": SceneEntityCfg(
          "robot",
          geom_names=(r".*finger[2-5]_link[2-4]_col",),
        ),
        "solref_timeconst_range": (1.0, 2.0),
        "solref_dampratio_range": (0.8, 1.2),
        "solimp_width_range": (1.0, 2.0),
        "solimp_dmin_range": (0.5, 1.0),
      },
    ),
    "object_size": EventTermCfg(
      mode="startup",
      func=mdp.randomize_geom_size_uniform,
      params={
        "asset_cfg": SceneEntityCfg("object", geom_names=("cube",)),
        "scale_range": (0.85, 1.15),
      },
    ),
    "object_mass": EventTermCfg(
      mode="startup",
      func=mdp.randomize_body_mass_and_inertia,
      params={
        "asset_cfg": SceneEntityCfg("object", body_names=("cube",)),
        "scale_range": (0.4, 1.6),
      },
    ),
    "pd_gains": EventTermCfg(
      mode="startup",
      func=mdp.dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (0.75, 1.5),
        "kd_range": (0.5, 2.0),
        "distribution": "log_uniform",
        "operation": "scale",
      },
    ),
    "robot_dof_damping": EventTermCfg(
      mode="startup",
      func=mdp.dr.dof_damping,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "operation": "scale",
        "distribution": "log_uniform",
        "ranges": (0.3, 3.0),
      },
    ),
    "robot_dof_armature": EventTermCfg(
      mode="startup",
      func=mdp.dr.dof_armature,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "operation": "scale",
        "ranges": (0.75, 1.3),
      },
    ),
    "robot_dof_frictionloss": EventTermCfg(
      mode="startup",
      func=mdp.dr.dof_frictionloss,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "operation": "scale",
        "ranges": (0.5, 2.0),
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=mdp.randomize_encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "bias_range": (-0.01, 0.01),
      },
    ),
    "robot_link_inertia": EventTermCfg(
      mode="startup",
      func=mdp.randomize_body_inertia,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "scale_range": (0.4, 1.5),
      },
    ),
    "robot_link_mass": EventTermCfg(
      mode="startup",
      func=mdp.dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "operation": "scale",
        "ranges": (0.4, 1.5),
      },
    ),
  }


def build_reorient_rewards(
  tip_site_names: tuple[str, ...],
) -> dict[str, RewardTermCfg]:
  """Build the reward term group."""
  return {
    # Dense rewards (× step_dt)
    "orientation_alignment": RewardTermCfg(
      func=mdp.orientation_alignment,
      weight=15.0,
      params={
        "command_name": "reorient_command",
        "margin": math.pi,
        "bound": 0.2,
      },
    ),
    "hand_pose": RewardTermCfg(
      func=mdp.hand_pose_penalty,
      weight=-0.2,
    ),
    "action_rate": RewardTermCfg(
      func=mdp.action_rate_combined,
      weight=-1.0,
    ),
    "torque": RewardTermCfg(
      func=mdp.torque_penalty,
      weight=-24.0,
    ),
    "tip_slide": RewardTermCfg(
      func=mdp.tip_slide_penalty,
      weight=-0.3,
      params={
        "robot_cfg": SceneEntityCfg("robot", site_names=tip_site_names),
        "sensor_cfg": SceneEntityCfg("tip_object_contact"),
        "contact_threshold": 0.0,
      },
    ),
    "cage_escape": RewardTermCfg(
      func=mdp.CageEscapePenalty,
      weight=-500.0,
      params={
        "robot_cfg": SceneEntityCfg(
          "robot", body_names=(".*_palm_link", ".*_finger.*_link[1-4]")
        ),
        "margin": 0.01,
      },
    ),
    "finger_collision": RewardTermCfg(
      func=mdp.finger_self_collision_penalty,
      weight=-1.0,
      params={
        "sensor_cfg": SceneEntityCfg("finger_collision"),
      },
    ),
    "hold_escalation": RewardTermCfg(
      func=mdp.hold_escalation,
      weight=11.4,
      params={
        "command_name": "reorient_command",
      },
    ),
    # palm_detach: cube held only by distal fingers (link3/link4).
    "palm_detach": RewardTermCfg(
      func=mdp.palm_detach_reward,
      weight=0.5,
    ),
  }


def build_reorient_terminations() -> dict[str, TerminationTermCfg]:
  """Build the termination term group."""
  return {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "cage_drop": TerminationTermCfg(
      func=mdp.cage_drop,
      params={"max_outside_steps": 10},
    ),
  }


def build_reorient_curriculum() -> dict[str, CurriculumTermCfg]:
  """Build the curriculum term group."""
  return {
    "success_curriculum": CurriculumTermCfg(
      func=mdp.reorient_success_curriculum,
      params={
        "command_name": "reorient_command",
        "count_threshold": 3,
        "delta_per_loop": 0.08,
      },
    ),
    "adaptive_episode": CurriculumTermCfg(
      func=mdp.adaptive_episode_curriculum,
      params={
        "target_steps": 800,
        "inc_rate": 0.1,
        "dec_rate": 0.2,
        "min_scale": 0.05,
        "drop_gamma": 1.0,
      },
    ),
  }


def build_reorient_metrics() -> dict[str, MetricsTermCfg]:
  """Build the metrics term group.

  Note (per-world-mesh mjlab): ``MetricsTermCfg.reduce`` is not exposed; all
  metrics are averaged over ``step_count``. Metrics that used to be
  ``reduce="last"`` (``cube_survival_steps``, ``goal_reach_count``) therefore
  report a per-episode mean of the underlying counter instead of its final
  value — still a monotonic training signal, just scaled by ~(N+1)/(2N).
  """
  return {
    "action_delta_rms": MetricsTermCfg(func=mdp.action_delta_rms),
    "action_jerk_rms": MetricsTermCfg(func=mdp.action_jerk_rms),
    "cage_escape_frequency": MetricsTermCfg(func=mdp.cage_escape_frequency),
    "fingertip_contact_count": MetricsTermCfg(
      func=mdp.fingertip_contact_count,
      params={"sensor_cfg": SceneEntityCfg("tip_object_contact")},
    ),
    "torque_saturation_ratio": MetricsTermCfg(
      func=mdp.torque_saturation_ratio,
      params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",))},
    ),
    "joint_acceleration_rms": MetricsTermCfg(
      func=mdp.joint_acceleration_rms,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_vel_rms": MetricsTermCfg(
      func=mdp.joint_vel_rms,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "cube_height_above_palm": MetricsTermCfg(
      func=mdp.cube_height_above_palm,
      params={"robot_cfg": SceneEntityCfg("robot", body_names=(".*_palm_link",))},
    ),
    "finger_collision_frequency": MetricsTermCfg(
      func=mdp.finger_collision_frequency,
      params={"sensor_cfg": SceneEntityCfg("finger_collision")},
    ),
    "cube_survival_steps": MetricsTermCfg(func=mdp.cube_survival_steps),
    "success_interval": MetricsTermCfg(
      func=mdp.success_interval,
      params={"command_name": "reorient_command"},
    ),
    "goal_reach_count": MetricsTermCfg(
      func=mdp.curriculum_progress,
      params={"command_name": "reorient_command"},
    ),
  }
