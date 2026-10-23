# SPDX-License-Identifier: Apache-2.0
"""Reuse released right Hand 2 checkpoints for experimental left-hand playback."""

import copy
import functools

import mujoco
import numpy as np

import wuji_mjlab.assets.robots.wuji_hand2.wuji_hand2_cfg as hand
import wuji_mjlab.tasks.pen_spin.config.wuji_hand2.hand2_constants as pen
import wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants as cube
import wuji_mjlab.utils.hand_mirroring as mirroring

SUPPORTED_TASKS = ("WujiHand2_Reorient_50Hz", "WujiHand2_PenSpin")
POSITION_SIGNS = np.array((1.0, -1.0, 1.0))
QUATERNION_SIGNS = np.array((1.0, -1.0, 1.0, -1.0))


def _reflect_mount_child(mount, child):
  # Rig constants are expressed in the calibrated right palm. Reflect that pose,
  # then express it in the official left mount, which is not an exact CAD mirror.
  palm_pose = np.zeros(3), np.zeros(4)
  mujoco.mju_mulPose(*palm_pose, mount.pos, mount.quat, child.pos, child.quat)
  inverse_mount = np.zeros(3), np.zeros(4)
  mujoco.mju_negPose(*inverse_mount, mount.pos, mount.quat)
  local_pose = np.zeros(3), np.zeros(4)
  mujoco.mju_mulPose(
    *local_pose,
    *inverse_mount,
    palm_pose[0] * POSITION_SIGNS,
    palm_pose[1] * QUATERNION_SIGNS,
  )
  child.pos, child.quat = local_pose


def _left_spec(task_id):
  base_spec = hand.get_wuji_hand2_cfg("left").spec_fn
  if task_id == "WujiHand2_PenSpin":
    spec = pen.get_hand2_pen_spin_spec(base_spec)
    capsules = pen.HAND2_PEN_SPIN_CAPSULES
  else:
    spec = cube.get_hand2_reorient_spec(base_spec)
    capsules = cube.HAND2_REORIENT_CAPSULES
    mount = spec.body("right_mount")
    _reflect_mount_child(mount, spec.geom("jig_table_visual"))
    _reflect_mount_child(mount, spec.body("right_wrist_tag_frame"))
    spec.mesh("jig_table").scale *= POSITION_SIGNS
  for name, (fromto, _) in capsules.items():
    spec.geom(f"{name}_col").fromto = (
      np.asarray(fromto).reshape(2, 3) * POSITION_SIGNS
    ).flatten()
  return spec


def _reflect_initial_state(state):
  state.pos = tuple(np.asarray(state.pos) * POSITION_SIGNS)
  state.rot = tuple(np.asarray(state.rot) * QUATERNION_SIGNS)


def _policy_observation(original, name, env, **kwargs):
  value = original(env, **kwargs)
  if name == "cube_pos_in_tag":
    return mirroring.points_y(value)
  if name == "cube_ori_error":
    return mirroring.rotation_error_y(value)
  if name == "object_pose":
    return mirroring.points_y(value.reshape(*value.shape[:-1], -1, 3)).flatten(-2)
  if name == "motion":
    # The command contains joint targets, shaft points, then hand-link positions.
    joints = len(env.command_manager.get_term("motion").joint_names)
    value = value.clone()
    value[..., joints:] = mirroring.points_y(
      value[..., joints:].reshape(*value.shape[:-1], -1, 3)
    ).flatten(-2)
    return value
  raise ValueError(f"Unknown mirrored policy observation {name!r}")


def configure_left_hand_playback(
  cfg, task_id: str, *, mirror_policy_observations: bool
):
  """Return a left-hand play config without changing the input or global state.

  Both variants mirror the physical setup and references. Disabling observation
  mirroring is a direct-transfer control, not a different starting grasp.
  Joint values already correspond across the official left/right joint axes.
  """
  if task_id not in SUPPORTED_TASKS:
    raise ValueError(
      f"Left-hand playback supports only {SUPPORTED_TASKS}, got {task_id!r}"
    )
  if cfg.curriculum or cfg.observations["policy"].enable_corruption:
    raise ValueError("Left-hand transfer requires a play configuration")
  cfg = copy.deepcopy(cfg)
  robot = cfg.scene.entities["robot"]
  robot.spec_fn = functools.partial(_left_spec, task_id)
  _reflect_initial_state(robot.init_state)
  if task_id == "WujiHand2_PenSpin":
    motion = cfg.commands["motion"]
    motion.mirror_reference_y = True
    motion.fixed_wrist_pose_w = (*robot.init_state.pos, *robot.init_state.rot)
    observations = ("motion", "object_pose")
  else:
    _reflect_initial_state(cfg.scene.entities["object"].init_state)
    cfg.rewards["cage_escape"].params["up_sign"] = -1
    observations = ("cube_pos_in_tag", "cube_ori_error")
  if mirror_policy_observations:
    for name in observations:
      term = cfg.observations["policy"].terms[name]
      term.func = functools.partial(_policy_observation, term.func, name)
  return cfg
