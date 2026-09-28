# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Robot-agnostic manager-based pen-spin (single-hand tracking) config."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  SceneEntityCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig

from wuji_mjlab.assets.objects.aruco_pen250 import get_pen_cfg
from wuji_mjlab.tasks.pen_spin import mdp
from wuji_mjlab.tasks.pen_spin.mdp.commands import SingleHandTrackingCommandCfg

HARDWARE_PEN_LENGTH_M = 0.250
HARDWARE_PEN_DIAMETER_M = 0.0149
PEN_MASS_KG = 0.030


def _pen_asset_cfg() -> SceneEntityCfg:
  return SceneEntityCfg(
    "object",
    geom_names=("cube", "A_bottom", "A_top"),
    body_names=("cube",),
    preserve_order=True,
  )


def pin_pen_dimensions_to_hardware(cfg) -> None:
  """Play holds the pen at the deployed size instead of randomizing it.

  The robot config owns the physical pen dimensions; recipes select only motion
  clips and never select or carry an object asset.
  """
  cfg.events["pen_dimensions"] = EventTermCfg(
    func=mdp.randomize_pen_dimensions,
    mode="startup",
    params={
      "diameter_range_m": (HARDWARE_PEN_DIAMETER_M, HARDWARE_PEN_DIAMETER_M),
      "total_length_range_m": (HARDWARE_PEN_LENGTH_M, HARDWARE_PEN_LENGTH_M),
      "asset_cfg": _pen_asset_cfg(),
    },
  )


def configure_pen_dimensions_dr(cfg) -> None:
  """Per-world pen geometry, spanning both the demonstrated pen and the real one.

  The range contains the 220 mm pen the reference motions were performed with and
  the 250 mm one on the bench: 235 +- 30 mm. randomize_pen_dimensions writes
  absolute sizes and must follow ``object_mass`` in the event order, because it
  spreads the randomized mass over the new geometry.
  """
  cfg.events["pen_dimensions"] = EventTermCfg(
    func=mdp.randomize_pen_dimensions,
    mode="startup",
    params={
      # Both pens are 15 mm across. The reset places fingers at the reference
      # clearances, so extra radius is penetration at the first step.
      "diameter_range_m": (0.014, 0.016),
      "total_length_range_m": (0.205, 0.265),
      "asset_cfg": _pen_asset_cfg(),
    },
  )


def make_pen_spin_env_cfg() -> ManagerBasedRlEnvCfg:
  def wrist_frame():
    return SceneEntityCfg(
      "robot",
      body_names=("right_palm_link",),
      preserve_order=True,
    )

  finger_joints = SceneEntityCfg("robot", joint_names=(), preserve_order=True)
  hand_links = SceneEntityCfg("robot", body_names=(), preserve_order=True)

  policy_terms = {
    "motion": ObservationTermCfg(
      mdp.generated_commands, params={"command_name": "motion"}
    ),
    # biased=True: hardware gives the policy q + b. The critic keeps the unbiased q.
    "joint_pos": ObservationTermCfg(
      mdp.joint_pos_rel,
      params={"asset_cfg": finger_joints, "biased": True},
      noise=UniformNoiseCfg(n_min=-0.015, n_max=0.015),
    ),
    "joint_vel": ObservationTermCfg(
      mdp.joint_vel_rel,
      params={"asset_cfg": finger_joints},
      scale=0.25,
      noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
    ),
    "object_pose": ObservationTermCfg(
      mdp.ObjectPoseCameraModel,
      params={
        "command_name": "motion",
        "robot_cfg": wrist_frame(),
        "object_cfg": SceneEntityCfg("object"),
        # Per episode: calibration errors are constants of a run.
        "bias_position_m": (-0.008, 0.008),
        "bias_orientation_rad": (-0.05, 0.05),
        # Per tick: PnP jitter. Measured sigma 0.2-0.3 mm and 0.154 deg;
        # these bounds sit ~4x above that.
        "noise_position_m": (-0.002, 0.002),
        "noise_orientation_rad": (-0.015, 0.015),
        # Pipeline latency is ~21 ms (exposure + detect + PnP + publish + the
        # 50 Hz loop's sampling wait), one control tick. Keep it constant.
        "lag_ticks": (1, 1),
        # Dropout bursts: 12% of ticks, mean 4.6 ticks with a tail to 38.
        "dropout_enter_prob": 0.03,
        "dropout_continue_prob": 0.78,
        "dropout_max_ticks": 40,
      },
    ),
    "last_action": ObservationTermCfg(
      mdp.last_action, params={"action_name": "joint_pos"}
    ),
  }
  critic_terms = {
    "motion": ObservationTermCfg(
      mdp.generated_commands, params={"command_name": "motion"}
    ),
    "joint_pos": ObservationTermCfg(
      mdp.joint_pos_rel, params={"asset_cfg": finger_joints}
    ),
    "joint_vel": ObservationTermCfg(
      mdp.joint_vel_rel, params={"asset_cfg": finger_joints}, scale=0.25
    ),
    "object_pose": ObservationTermCfg(
      mdp.object_axis_points_wrist_local_bounded,
      params={
        "command_name": "motion",
        "robot_cfg": wrist_frame(),
      },
    ),
    "last_action": ObservationTermCfg(
      mdp.last_action, params={"action_name": "joint_pos"}
    ),
    "link_pos": ObservationTermCfg(
      mdp.link_pos_wrist_local,
      params={
        "link_cfg": hand_links,
        "robot_cfg": wrist_frame(),
      },
    ),
    "joint_position_error": ObservationTermCfg(
      mdp.joint_position_error, params={"joint_cfg": finger_joints}
    ),
    "link_position_error": ObservationTermCfg(
      mdp.link_position_error,
      params={
        "link_cfg": hand_links,
        "robot_cfg": wrist_frame(),
      },
    ),
  }

  events = {
    "reset_to_reference_state": EventTermCfg(
      func=mdp.reset_to_reference_state,
      mode="reset",
      params={
        "command_name": "motion",
      },
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"object": get_pen_cfg(PEN_MASS_KG)},
      num_envs=4096,
      env_spacing=0.0,
    ),
    observations={
      "policy": ObservationGroupCfg(
        policy_terms, concatenate_terms=True, enable_corruption=True
      ),
      "critic": ObservationGroupCfg(
        critic_terms, concatenate_terms=True, enable_corruption=False
      ),
    },
    actions={
      "joint_pos": mdp.SingleHandResidualEMAActionCfg(
        entity_name="robot", actuator_names=()
      )
    },
    commands={
      "motion": SingleHandTrackingCommandCfg(
        recipe_path="",
        axis_point_radius_m=0.10,
        debug_vis=False,
      )
    },
    events=events,
    curriculum={},
    rewards={},
    metrics={},
    terminations={},
    viewer=ViewerConfig(
      # The Wuji Hand 2 wrist is fixed at z=0.5 m.  Looking near the ground
      # frames only its shadow instead of the hand and pen.
      lookat=(0.0, 0.0, 0.5),
      distance=0.45,
      elevation=-25.0,
      azimuth=-40.0,
      origin_type=ViewerConfig.OriginType.WORLD,
    ),
    # 50 Hz control to match the 50 fps reference clips (step_dt = 0.005 * 4).
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=0.005, integrator="implicitfast", iterations=10, ls_iterations=20
      ),
    ),
    decimation=4,
    # Longer than any clip (median ~25 s); episodes end on reference_end.
    episode_length_s=60.0,
  )
  return cfg
