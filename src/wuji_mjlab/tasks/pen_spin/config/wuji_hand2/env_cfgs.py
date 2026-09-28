# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Complete Wuji Hand 2 pen-spin environment configuration."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr, time_out
from mjlab.managers import (
  CurriculumTermCfg,
  EventTermCfg,
  MetricsTermCfg,
  RewardTermCfg,
  SceneEntityCfg,
  TerminationTermCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg

from wuji_mjlab.tasks.pen_spin import mdp
from wuji_mjlab.tasks.pen_spin.pen_spin_env_cfg import (
  configure_pen_dimensions_dr,
  make_pen_spin_env_cfg,
  pin_pen_dimensions_to_hardware,
)

from .hand2_constants import (
  HAND2_MUJOCO_IMPRATIO,
  HAND2_NCONMAX,
  HAND2_NJMAX,
  PEN_SPIN_FINGER_CONTACT_GEOMS,
  PEN_SPIN_HAND_CONTACT_GEOMS,
  RIGHT_FINGER_JOINT_NAMES,
  RIGHT_FIXED_WRIST_POSE_W,
  RIGHT_HAND_LINK_NAMES,
  get_pen_spin_hand2_cfg,
)

_RIGHT_PALM_LINK_NAME = "right_palm_link"


def _finger_joint_cfg() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot",
    joint_names=RIGHT_FINGER_JOINT_NAMES,
    preserve_order=True,
  )


def _palm_body_cfg() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot",
    body_names=(_RIGHT_PALM_LINK_NAME,),
  )


def _object_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("object")


def _configure_hand2_domain_randomization(
  cfg: ManagerBasedRlEnvCfg,
  *,
  play: bool,
) -> None:
  """Attach the production Wuji Hand 2 DR surface without polluting the base task."""
  reset = cfg.events["reset_to_reference_state"]
  if play:
    cfg.events = {"reset_to_reference_state": reset}
    # Everything else is training-only; the pen still has to match the hardware pen.
    pin_pen_dimensions_to_hardware(cfg)
    return

  contact_geoms = SceneEntityCfg("robot", geom_names=PEN_SPIN_HAND_CONTACT_GEOMS)
  robot_joints = SceneEntityCfg("robot", joint_names=".*")
  cfg.events = {
    "reset_to_reference_state": reset,
    # Axial and radial are split because their limits differ by 80x: I_xx is
    # 140x below I_yy, and a radial COM offset beyond +-0.5 mm destabilizes
    # contacts at the 5 ms step.
    "object_com_axial": EventTermCfg(
      func=dr.body_com_offset,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("object", body_names=("cube",)),
        "operation": "add",
        "ranges": (-0.04, 0.04),
        "axes": (0,),
      },
    ),
    "object_com_radial": EventTermCfg(
      func=dr.body_com_offset,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("object", body_names=("cube",)),
        "operation": "add",
        "ranges": (-0.0005, 0.0005),
        "axes": (1, 2),
      },
    ),
    # Robot friction takes precedence over the pen because its geom priority is higher.
    "robot_friction": EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": contact_geoms,
        "operation": "abs",
        "distribution": "log_uniform",
        "ranges": (0.35, 1.4),
        "axes": [0],
      },
    ),
    "robot_torsional_friction": EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": contact_geoms,
        "operation": "scale",
        "distribution": "log_uniform",
        "ranges": (0.5, 2.0),
        "axes": [1],
      },
    ),
    # The XML's time-constant solref/solimp, randomized as the reorient task
    # does for the same hand's fingers.
    "contact_params": EventTermCfg(
      func=mdp.randomize_contact_params,
      mode="startup",
      params={
        "robot_cfg": contact_geoms,
        "solref_timeconst_range": (1.0, 2.0),
        "solref_dampratio_range": (0.8, 1.2),
        "solimp_width_range": (1.0, 2.0),
        "solimp_dmin_range": (0.5, 1.0),
      },
    ),
    # A 0.4-1.6x mass span; mass and inertia both scale by e^(2 alpha).
    # pen_dimensions then rebuilds the inertia for the drawn geometry.
    "object_mass": EventTermCfg(
      func=dr.pseudo_inertia,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("object", body_names=("cube",)),
        "alpha_range": (0.5 * math.log(0.4), 0.5 * math.log(1.6)),
      },
    ),
    # The nominal gains are identified (hand2_constants.py); the band covers the
    # motor types whose Kt is scaled rather than measured. kd reaches lower
    # because the firmware's filtered velocity term fits at 0.85-1.0 of kd.
    "pd_gains": EventTermCfg(
      func=dr.pd_gains,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (0.9, 1.1),
        "kd_range": (0.8, 1.1),
        "distribution": "log_uniform",
        "operation": "scale",
      },
    ),
    "encoder_bias": EventTermCfg(
      func=dr.encoder_bias,
      mode="startup",
      params={
        "asset_cfg": robot_joints,
        "bias_range": (-0.01, 0.01),
      },
    ),
    # Mass and inertia move together so I1 + I2 >= I3 holds. alpha is the
    # density scale (mass and inertia both e^(2a), a 0.4-1.5x mass span), d an
    # axis stretch. Not ".*": mocap_base has mass 0 and pseudo_inertia divides by it.
    "robot_link_inertial": EventTermCfg(
      func=dr.pseudo_inertia,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_HAND_LINK_NAMES),
        "alpha_range": (-0.458, 0.203),
        "d_range": (-0.2, 0.2),
      },
    ),
    # No dof_damping term: the compiled dof_damping is all zero, so "scale" is the identity.
    "robot_dof_armature": EventTermCfg(
      func=dr.dof_armature,
      mode="startup",
      params={
        "asset_cfg": robot_joints,
        "operation": "scale",
        "ranges": (0.75, 1.3),
      },
    ),
    # The nominal is identified per joint on one hand (hand2_constants.py); the band
    # covers the estimator's ~10% low bias, wear, temperature and unit spread.
    "robot_joint_friction": EventTermCfg(
      func=dr.joint_friction,
      mode="startup",
      params={
        "asset_cfg": robot_joints,
        "operation": "scale",
        "ranges": (0.5, 1.5),
      },
    ),
    # The current cap is exact, the torque constant is not. The band reaches
    # further down than up: on hardware the side-swing joints hit the cap doing
    # what took 55% of it here.
    "robot_effort_limits": EventTermCfg(
      func=dr.effort_limits,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "effort_limit_range": (0.7, 1.05),
        "operation": "scale",
      },
    ),
    "object_disturbance": EventTermCfg(
      func=mdp.apply_object_disturbance_wrench,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("object", body_names=("cube",)),
        "grav_mult_range": (1.0, 2.0),
        "angular_accel": 375.0,
        "active_s_range": (0.2, 0.4),
        "silence_s_range": (0.5, 1.2),
        "decay_to_frac": 0.02,
        "warmup_time_s": 1.0,
        "curriculum_term": "disturbance_success_rate",
      },
    ),
  }

  configure_pen_dimensions_dr(cfg)


def _configure_hand2_fingertip_bumps(
  cfg: ManagerBasedRlEnvCfg,
  *,
  play: bool,
) -> None:
  """Random bumps on the five fingertip links, ramped like
  ``object_disturbance``. Play builds stay bump-free."""
  if play:
    return
  cfg.events["fingertip_random_force"] = EventTermCfg(
    mode="step",
    func=mdp.apply_link_random_force,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names="right_finger._link4"),
      "grav_mult_range": (0.0, 4.0),
      # Rounded compensation keeps wrench amplitudes within 6% of e8f142b
      # without changing hand2 masses, their DR, or the random draws.
      "force_scale": {f"right_finger{finger}_link4": 1.5 for finger in range(1, 6)},
      "offset_radius_scale": {
        f"right_finger{finger}_link4": 0.9 for finger in range(1, 6)
      },
      "duration_s": (0.1, 0.3),
      "cooldown_s": (0.5, 1.5),
      "curriculum_term": "disturbance_success_rate",
    },
  )


def _configure_hand2_reward_and_curriculum_surface(
  cfg: ManagerBasedRlEnvCfg,
  *,
  play: bool,
) -> None:
  """Rewards, sensors, metrics, action mask and the disturbance curriculum."""
  palm_cfg = _palm_body_cfg()
  finger_joint_cfg = _finger_joint_cfg()
  object_cfg = _object_cfg()

  cfg.rewards = {
    "object_pos": RewardTermCfg(
      func=mdp.object_position_tracking_reward,
      weight=2.0,
      params={
        "robot_cfg": palm_cfg,
        "object_cfg": object_cfg,
        "sigma": 0.05,
      },
    ),
    "object_rot": RewardTermCfg(
      func=mdp.object_axis_point_tracking_reward,
      weight=8.0,
      params={
        "robot_cfg": palm_cfg,
        "object_cfg": object_cfg,
        "sigma": 0.04,
      },
    ),
    "link_pos": RewardTermCfg(
      func=mdp.link_position_tracking_reward,
      weight=1.0,
      params={
        "link_cfg": SceneEntityCfg(
          "robot", body_names=RIGHT_HAND_LINK_NAMES, preserve_order=True
        ),
        "robot_cfg": palm_cfg,
      },
    ),
    "joint_pos": RewardTermCfg(
      func=mdp.joint_position_tracking_reward,
      weight=1.0,
      params={"joint_cfg": finger_joint_cfg},
    ),
    "action_rate": RewardTermCfg(
      func=mdp.action_rate_combined,
      weight=-0.1,
    ),
    # Dimensionless torque utilization: -dt * sum((torque / torque_limit)**2), ungated.
    "torque": RewardTermCfg(
      func=mdp.normalized_joint_torques_l2,
      weight=-1.0,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", actuator_names=("right_finger.*_joint.*",), preserve_order=True
        )
      },
    ),
    # Short peaks at the current cap are free; a joint held near it is not.
    # -4 makes one stalled joint cost half of what object_rot (8) can earn.
    "overcurrent": RewardTermCfg(
      func=mdp.sustained_overcurrent,
      weight=-4.0,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", actuator_names=("right_finger.*_joint.*",), preserve_order=True
        ),
        "charge_s": 1.0,
        "leak_s": 2.0,
        "threshold": 0.8,
      },
    ),
  }

  # Sense both hand2 distal pieces; the reward merges link4 and tip so a
  # single logical distal link is not counted twice.
  cfg.scene.sensors = tuple(cfg.scene.sensors) + (
    ContactSensorCfg(
      name="finger_collision",
      primary=ContactMatch(
        mode="geom", pattern=PEN_SPIN_FINGER_CONTACT_GEOMS, entity="robot"
      ),
      secondary=ContactMatch(
        mode="subtree", pattern=_RIGHT_PALM_LINK_NAME, entity="robot"
      ),
      fields=("found",),
      reduce="none",
      num_slots=1,
    ),
  )
  cfg.rewards["finger_collision"] = RewardTermCfg(
    func=mdp.finger_self_collision_penalty,
    weight=-1.0,
  )

  if play:
    return

  cfg.metrics = {
    "object_pos_tracking_error_cm": MetricsTermCfg(
      mdp.object_pos_tracking_error_cm, params={"robot_cfg": palm_cfg}
    ),
    "object_ori_tracking_error_deg": MetricsTermCfg(
      mdp.object_ori_tracking_error_deg, params={"robot_cfg": palm_cfg}
    ),
    # Smoothness diagnostics: action_rate_sq + action_acc_sq is the unweighted
    # action_rate reward; the joint terms show what reaches the actual motion.
    "action_rate_sq": MetricsTermCfg(mdp.action_rate_l2),
    "action_acc_sq": MetricsTermCfg(mdp.action_acc_l2),
    "joint_vel_rms_rad_s": MetricsTermCfg(
      mdp.joint_vel_rms, params={"joint_cfg": finger_joint_cfg}
    ),
    "joint_acc_rms_rad_s2": MetricsTermCfg(
      mdp.joint_acc_rms, params={"joint_cfg": finger_joint_cfg}
    ),
    "actuator_saturation_share": MetricsTermCfg(
      mdp.actuator_saturation_share,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", actuator_names=("right_finger.*_joint.*",), preserve_order=True
        )
      },
    ),
  }

  # Adaptive sampling biases any time_out-based success rate (episodes that start
  # mid-clip reach reference_end more easily), so the disturbance ramps on steps
  # alone: zero through 40k control steps (iteration 1000), full at 240k (6000).
  cfg.curriculum = {
    "disturbance_success_rate": CurriculumTermCfg(
      func=mdp.linear_step_curriculum,
      params={"total_env_steps": 240_000, "warmup_env_steps": 40_000},
    )
  }

  cfg.events["reset_to_reference_state"].params.update(
    object_reset_axial_slide_range=(-0.02, 0.02),
    object_reset_roll_range=(-math.radians(20.0), math.radians(20.0)),
    curriculum_term="disturbance_success_rate",
  )
  act = cfg.actions["joint_pos"]
  act.mask_prob = 0.06
  act.mask_duration_max = 5
  act.mask_curriculum_term = "disturbance_success_rate"


def _configure_hand2_policy_interface(
  cfg: ManagerBasedRlEnvCfg,
) -> None:
  # The action, the joint observations and the command reference all use the
  # 20 finger joints in RIGHT_FINGER_JOINT_NAMES order; the action term checks it.
  cfg.actions["joint_pos"].actuator_names = ("right_finger.*_joint.*",)

  for group_name in ("policy", "critic"):
    for term_name in ("joint_pos", "joint_vel"):
      cfg.observations[group_name].terms[term_name].params["asset_cfg"] = (
        _finger_joint_cfg()
      )
  cfg.observations["critic"].terms["joint_position_error"].params["joint_cfg"] = (
    _finger_joint_cfg()
  )


def wuji_hand2_pen_spin_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Wuji Hand 2 pen-spin tracking on the fixed wrist mount.

  The motion command loads its recipe's clips when the environment is built.
  """
  cfg = make_pen_spin_env_cfg()
  cfg.scene.entities["robot"] = get_pen_spin_hand2_cfg()
  cfg.sim.mujoco.impratio = HAND2_MUJOCO_IMPRATIO
  cfg.sim.nconmax = HAND2_NCONMAX
  cfg.sim.njmax = HAND2_NJMAX

  motion = cfg.commands["motion"]
  motion.joint_names = RIGHT_FINGER_JOINT_NAMES
  motion.link_names = RIGHT_HAND_LINK_NAMES
  # The step-based disturbance curriculum exists because of this setting; keep the two together.
  motion.trajectory_sampling_mode = "adaptive"
  motion.adaptive_sampling_uniform_mix = 0.2
  motion.adaptive_sampling_bin_seconds = 0.5
  motion.first_frame_hold_prob = 0.15
  motion.first_frame_hold_s = (0.5, 3.0)
  for term_name in ("link_pos", "link_position_error"):
    cfg.observations["critic"].terms[term_name].params[
      "link_cfg"
    ].body_names = RIGHT_HAND_LINK_NAMES
  if play:
    # Play follows one clip from its first frame, the way deploy runs it:
    # the policy holds the first frame, then the clip starts.
    motion.trajectory_sampling_mode = "uniform"
    motion.first_frame_hold_prob = 1.0
    motion.first_frame_hold_s = (1.0, 1.0)
    motion.debug_vis = True
    cfg.scene.num_envs = 1
    policy = cfg.observations["policy"]
    policy.terms["joint_pos"].noise = None
    policy.terms["joint_vel"].noise = None
    # Play sees the pen exactly where it is: the camera model is swapped out, not zeroed.
    object_pose = policy.terms["object_pose"]
    object_pose.func = mdp.object_axis_points_wrist_local_bounded
    object_pose.params = {
      "command_name": object_pose.params["command_name"],
      "robot_cfg": object_pose.params["robot_cfg"],
      "object_cfg": object_pose.params["object_cfg"],
    }
    for group in cfg.observations.values():
      group.enable_corruption = False

  motion.fixed_wrist_pose_w = RIGHT_FIXED_WRIST_POSE_W
  _configure_hand2_policy_interface(cfg)

  palm_cfg = _palm_body_cfg()
  object_cfg = _object_cfg()
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "unstable": TerminationTermCfg(
      func=mdp.numerical_instability,
      params={"object_cfg": object_cfg},
    ),
    "object_position_error": TerminationTermCfg(
      func=mdp.object_position_error_exceeded,
      params={
        "robot_cfg": palm_cfg,
        "object_cfg": object_cfg,
        "threshold": 0.05,
      },
      time_out=False,
    ),
    "object_orientation_error": TerminationTermCfg(
      func=mdp.object_shaft_direction_error_exceeded,
      params={
        "robot_cfg": palm_cfg,
        "object_cfg": object_cfg,
        "threshold_deg": 60.0,
      },
      time_out=False,
    ),
    "reference_end": TerminationTermCfg(
      func=mdp.reference_end,
      time_out=True,
    ),
  }
  if play:
    cfg.terminations.pop("object_position_error")
    cfg.terminations.pop("object_orientation_error")

  _configure_hand2_domain_randomization(cfg, play=play)
  _configure_hand2_fingertip_bumps(cfg, play=play)
  _configure_hand2_reward_and_curriculum_surface(cfg, play=play)
  return cfg
