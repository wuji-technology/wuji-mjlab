# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Scene construction and obs/action/goal utilities for reorient eval scripts.

Builds a MuJoCo scene by composing robot + cube specs via MjSpec attach,
matching the mjlab scene pipeline (scene.py) but standalone for eval.

The thin CLI wrappers (``scripts/eval_success_rate.py`` etc.) import from
this module so the building blocks can be exercised without launching the
viewer or full mjlab env stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from wuji_mjlab.tasks.reorient.robot_bindings import (
  REVO3_RIGHT_HAND_BINDING,
  WUJI_RIGHT_HAND_BINDING,
  ReorientRobotBinding,
)

_GOAL_VIS_Z_OFFSET = 0.15


# ---------------------------------------------------------------------------
# Quaternion utilities (numpy, wxyz = [w, x, y, z])
# ---------------------------------------------------------------------------


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
  """Multiply two quaternions (wxyz format)."""
  w1, x1, y1, z1 = q1
  w2, x2, y2, z2 = q2
  return np.array(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
  )


def quat_inv(q: np.ndarray) -> np.ndarray:
  """Inverse of a unit quaternion (wxyz format) = conjugate."""
  return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate vector v by quaternion q (wxyz format).

  Uses the formula: v' = v + 2w*(xyz × v) + 2*(xyz × (xyz × v)).
  Matches mjlab quat_apply semantics.
  """
  w, x, y, z = q
  xyz = np.array([x, y, z])
  t = 2.0 * np.cross(xyz, v)
  return v + w * t + np.cross(xyz, t)


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate vector v by the inverse of quaternion q (wxyz format).

  Equivalent to quat_apply(quat_inv(q), v).
  """
  return quat_apply(quat_inv(q), v)


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
  """Convert quaternion (wxyz) to 3x3 rotation matrix.

  Matches mjlab/utils/lab_api/math.py:168-198.
  """
  w, x, y, z = q
  two_s = 2.0 / (np.dot(q, q))
  return np.array(
    [
      1 - two_s * (y * y + z * z),
      two_s * (x * y - z * w),
      two_s * (x * z + y * w),
      two_s * (x * y + z * w),
      1 - two_s * (x * x + z * z),
      two_s * (y * z - x * w),
      two_s * (x * z - y * w),
      two_s * (y * z + x * w),
      1 - two_s * (x * x + y * y),
    ]
  )


def quat_error_magnitude(q1: np.ndarray, q2: np.ndarray) -> float:
  """Angular error between two quaternions in radians (wxyz format).

  Matches mjlab quat_box_minus → axis_angle → norm.
  """
  qd = quat_mul(q1, quat_inv(q2))
  # Ensure w >= 0 for shortest path
  if qd[0] < 0:
    qd = -qd
  # axis-angle: 2 * arctan2(|v|, w)
  vec_norm = np.linalg.norm(qd[1:])
  angle = 2.0 * np.arctan2(vec_norm, qd[0])
  return angle


def random_quat_uniform() -> np.ndarray:
  """Generate uniformly random unit quaternion (wxyz), w >= 0."""
  q = np.random.randn(4)
  q /= np.linalg.norm(q)
  if q[0] < 0:
    q = -q
  return q.astype(np.float64)


def quat_unique(q: np.ndarray) -> np.ndarray:
  """Ensure w >= 0 (canonical form)."""
  if q[0] < 0:
    return -q
  return q


# ---------------------------------------------------------------------------
# Scene metadata
# ---------------------------------------------------------------------------


@dataclass
class SceneMetadata:
  model: mujoco.MjModel
  data: mujoco.MjData
  joint_qpos_adr: np.ndarray
  ctrl_ids: np.ndarray
  cube_qpos_adr: int  # freejoint qpos start
  cube_body_id: int
  palm_body_id: int
  goal_mocap_id: int  # mocap body index
  default_joint_pos: np.ndarray
  default_cube_pos: np.ndarray  # (3,) from keyframe
  default_cube_quat: np.ndarray  # (4,) wxyz from keyframe
  soft_lower: np.ndarray
  soft_upper: np.ndarray
  robot_binding: ReorientRobotBinding
  tag_in_palm_pos: np.ndarray
  tag_in_palm_quat: np.ndarray

  # Timing
  sim_dt: float = 0.01
  ctrl_dt: float = 0.05
  n_substeps: int = 5

  @property
  def cube_pos(self) -> np.ndarray:
    """Current cube position from qpos (freejoint xyz)."""
    return self.data.qpos[self.cube_qpos_adr : self.cube_qpos_adr + 3].copy()

  @property
  def cube_quat(self) -> np.ndarray:
    """Current cube quaternion from qpos (freejoint wxyz)."""
    return self.data.qpos[self.cube_qpos_adr + 3 : self.cube_qpos_adr + 7].copy()

  @property
  def joint_pos(self) -> np.ndarray:
    """Current joint positions."""
    return self.data.qpos[self.joint_qpos_adr].copy()

  @property
  def action_dim(self) -> int:
    return int(self.default_joint_pos.shape[0])

  @property
  def palm_pos(self) -> np.ndarray:
    """Current palm_link position in world frame (3,) from xpos."""
    return self.data.xpos[self.palm_body_id].copy()

  @property
  def palm_quat(self) -> np.ndarray:
    """Current palm_link orientation in world frame (wxyz, 4,) from xquat."""
    return self.data.xquat[self.palm_body_id].copy()


def _resolve_scene_task(task_id: str | None):
  normalized = task_id or "WujiHand_Reorient"
  if normalized.startswith("Revo3RightHand"):
    from wuji_mjlab.assets.robots.revo3_hand.revo3_hand_cfg import (
      get_revo3_hand_cfg,
    )
    from wuji_mjlab.tasks.reorient.config.revo3_hand.env_cfgs import (
      REVO3_REORIENT_CUBE_INIT_STATE,
      REVO3_REORIENT_ROBOT_INIT_STATE,
    )

    return (
      normalized,
      REVO3_RIGHT_HAND_BINDING,
      get_revo3_hand_cfg,
      REVO3_REORIENT_ROBOT_INIT_STATE,
      REVO3_REORIENT_CUBE_INIT_STATE,
    )

  from wuji_mjlab.assets.robots.wuji_hand.wuji_hand_cfg import get_wuji_hand_cfg
  from wuji_mjlab.tasks.reorient.reorient_constants import (
    REORIENT_CUBE_INIT_STATE,
    REORIENT_ROBOT_INIT_STATE,
  )

  return (
    normalized,
    WUJI_RIGHT_HAND_BINDING,
    get_wuji_hand_cfg,
    REORIENT_ROBOT_INIT_STATE,
    REORIENT_CUBE_INIT_STATE,
  )


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------


def build_reorient_scene(
  sim_dt: float = 0.01,
  ctrl_dt: float = 0.05,
  soft_limit_factor: float = 0.9,
  cube_edge_m: float | None = None,
  task_id: str | None = None,
) -> SceneMetadata:
  """Build a MuJoCo scene for reorient eval by composing robot + cube specs.

  Follows the scene.py MjSpec attach pattern:
  1. Create empty MjSpec, set timestep
  2. Robot spec: extract keyframe → delete → attach(prefix="robot/")
  3. Cube spec: extract keyframe → delete → attach(prefix="object/")
  4. Merge assets (original + prefixed)
  5. Add goal mocap body (semi-transparent cube mesh)
  6. Merge keyframes → add_key("init_state", ...)
  7. Compile, resolve indices
  """
  from wuji_mjlab.assets.objects.inhand_object.object_cfg import get_inhand_object_cfg

  _, robot_binding, robot_cfg_factory, robot_init, cube_init = _resolve_scene_task(
    task_id
  )

  spec = mujoco.MjSpec()
  spec.option.timestep = sim_dt

  all_entity_assets: dict[str, bytes] = {}

  # Robot spec: attach with prefix at InitialStateCfg position
  robot_cfg = robot_cfg_factory()
  robot_spec = robot_cfg.spec_fn()
  if robot_spec.assets:
    all_entity_assets.update(robot_spec.assets)
  while robot_spec.keys:
    robot_spec.delete(robot_spec.keys[0])
  frame = spec.worldbody.add_frame()
  frame.pos = np.array(robot_init.pos)  # (0, 0, 0.5) — lift hand above ground
  frame.quat = np.array(robot_init.rot)  # (1, 0, 0, 0)
  spec.attach(robot_spec, prefix="robot/", frame=frame)

  # Cube spec: attach with prefix (parameterized size)
  cube_cfg = get_inhand_object_cfg(edge_m=cube_edge_m)
  cube_spec = cube_cfg.spec_fn()
  if cube_spec.assets:
    all_entity_assets.update(cube_spec.assets)
  while cube_spec.keys:
    cube_spec.delete(cube_spec.keys[0])
  frame = spec.worldbody.add_frame()
  spec.attach(cube_spec, prefix="object/", frame=frame)

  # Merge original entity assets into scene spec (scene.py)
  if all_entity_assets:
    existing = dict(spec.assets) if spec.assets else {}
    existing.update(all_entity_assets)
    spec.assets = existing

  # -- Add goal mocap body (semi-transparent textured cube for visualization) --
  # Reuses the attached cube's mesh + dexcube material (prefix "object/") so the
  # goal marker matches the live cube's appearance; alpha=0.6 keeps the texture
  # visible while letting the real cube show through behind it.
  goal_body = spec.worldbody.add_body()
  goal_body.name = "goal"
  goal_body.mocap = True
  goal_body.pos = np.array([0.0, 0.0, 0.65])  # Above hand, z+0.15 from cube default

  goal_geom = goal_body.add_geom()
  goal_geom.type = mujoco.mjtGeom.mjGEOM_MESH
  goal_geom.meshname = "object/cube_mesh"
  goal_geom.material = "object/dexcube"
  goal_geom.rgba = np.array(
    [1.0, 1.0, 1.0, 0.6]
  )  # Semi-transparent, preserves texture color
  goal_geom.contype = 0
  goal_geom.conaffinity = 0
  goal_geom.group = 2

  # -- Visual assets: skybox + checker groundplane + directional light --
  # Mirrors deploy/reorient/scripts/toreal_viewer.py:build_viewer_scene_xml()
  # so the eval viewer matches the deployment viewer's look.
  skybox = spec.add_texture()
  skybox.name = "skybox"
  skybox.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
  skybox.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
  skybox.rgb1 = np.array([0.3, 0.5, 0.9])
  skybox.rgb2 = np.array([0.9, 0.95, 1.0])
  skybox.width = 800
  skybox.height = 800

  gp_tex = spec.add_texture()
  gp_tex.name = "groundplane"
  gp_tex.type = mujoco.mjtTexture.mjTEXTURE_2D
  gp_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
  gp_tex.mark = mujoco.mjtMark.mjMARK_EDGE
  gp_tex.rgb1 = np.array([0.2, 0.3, 0.4])
  gp_tex.rgb2 = np.array([0.1, 0.2, 0.3])
  gp_tex.markrgb = np.array([0.8, 0.8, 0.8])
  gp_tex.width = 300
  gp_tex.height = 300

  gp_mat = spec.add_material()
  gp_mat.name = "groundplane"
  gp_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
  gp_mat.texrepeat = np.array([5.0, 5.0])
  gp_mat.texuniform = True
  gp_mat.reflectance = 0.2

  # Global visual tweaks (best-effort; harmless if unsupported)
  try:
    spec.visual.headlight.diffuse = np.array([0.8, 0.8, 0.8])
    spec.visual.headlight.ambient = np.array([0.2, 0.2, 0.2])
    spec.visual.headlight.specular = np.array([1.0, 1.0, 1.0])
    spec.visual.global_.azimuth = 120.0
    spec.visual.global_.elevation = -20.0
    spec.visual.quality.shadowsize = 8192
  except Exception:
    pass

  # Directional light from above so the checker floor casts shadow.
  light = spec.worldbody.add_light()
  light.pos = np.array([0.0, 0.0, 1.5])
  light.dir = np.array([0.0, 0.0, -1.0])
  light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

  # -- Add ground plane (textured via groundplane material) --
  ground_geom = spec.worldbody.add_geom()
  ground_geom.name = "floor"
  ground_geom.type = mujoco.mjtGeom.mjGEOM_PLANE
  ground_geom.size = np.array([5.0, 5.0, 0.05])
  ground_geom.material = "groundplane"
  ground_geom.contype = 1
  ground_geom.conaffinity = 1

  # -- Add a placeholder keyframe so model.nkey >= 1 for reset_scene --
  spec.add_key(name="init_state")

  # -- Compile first to resolve indices --
  model = spec.compile()
  data = mujoco.MjData(model)

  # -- Resolve indices --
  joint_names = [f"robot/{name}" for name in robot_binding.joint_names]
  joint_qpos_adr = np.array([model.joint(n).qposadr[0] for n in joint_names])

  # Actuator IDs (actuator names are <joint>_actuator in the XML)
  ctrl_ids = np.array(
    [model.actuator(f"robot/{name}_actuator").id for name in robot_binding.joint_names]
  )

  # Cube freejoint
  cube_jnt_name = "object/cube_freejoint"
  cube_qpos_adr = model.joint(cube_jnt_name).qposadr[0]
  cube_body_id = model.body("object/cube").id

  # Palm body (for tag-frame obs)
  palm_body_id = model.body(f"robot/{robot_binding.viewer_body_name}").id

  # Goal mocap body
  goal_body_id = model.body("goal").id
  goal_mocap_id = model.body_mocapid[goal_body_id]
  assert goal_mocap_id >= 0, "Goal body must be a mocap body"

  # -- Build init state from constants (specs have no keyframes; Entity builds them) --
  # Robot joint positions from task init state.
  robot_home = robot_init
  default_joint_pos = np.zeros(len(robot_binding.joint_names), dtype=np.float64)
  bare_joint_names = list(robot_binding.joint_names)
  for i, jname in enumerate(bare_joint_names):
    # joint_pos keys are regex patterns (e.g. ".*_finger1_joint1"); match against bare name
    for pattern, val in robot_home.joint_pos.items():
      import re

      if re.fullmatch(pattern, jname):
        default_joint_pos[i] = val
        break

  # Write robot joint qpos
  data.qpos[joint_qpos_adr] = default_joint_pos
  # Write robot joint ctrl (same as joint pos for position actuators)
  data.ctrl[ctrl_ids] = default_joint_pos

  # Cube initial state from task config.
  default_cube_pos = np.array(cube_init.pos, dtype=np.float64)
  default_cube_quat = np.array(cube_init.rot, dtype=np.float64)
  data.qpos[cube_qpos_adr : cube_qpos_adr + 3] = default_cube_pos
  data.qpos[cube_qpos_adr + 3 : cube_qpos_adr + 7] = default_cube_quat

  mujoco.mj_forward(model, data)

  # Save as keyframe for reset_scene()
  model.key_qpos[0] = data.qpos.copy()
  model.key_ctrl[0] = data.ctrl.copy()

  # -- Soft joint limits (entity.py) --
  raw_lower = np.array([model.jnt_range[model.joint(n).id][0] for n in joint_names])
  raw_upper = np.array([model.jnt_range[model.joint(n).id][1] for n in joint_names])
  mean = (raw_lower + raw_upper) / 2.0
  range_ = raw_upper - raw_lower
  soft_lower = mean - 0.5 * range_ * soft_limit_factor
  soft_upper = mean + 0.5 * range_ * soft_limit_factor

  n_substeps = int(round(ctrl_dt / sim_dt))

  return SceneMetadata(
    model=model,
    data=data,
    joint_qpos_adr=joint_qpos_adr,
    ctrl_ids=ctrl_ids,
    cube_qpos_adr=cube_qpos_adr,
    cube_body_id=cube_body_id,
    palm_body_id=palm_body_id,
    goal_mocap_id=goal_mocap_id,
    default_joint_pos=default_joint_pos,
    default_cube_pos=default_cube_pos,
    default_cube_quat=default_cube_quat,
    soft_lower=soft_lower,
    soft_upper=soft_upper,
    robot_binding=robot_binding,
    tag_in_palm_pos=np.array(robot_binding.tag_in_palm_pos, dtype=np.float64),
    tag_in_palm_quat=np.array(robot_binding.tag_in_palm_quat, dtype=np.float64),
    sim_dt=sim_dt,
    ctrl_dt=ctrl_dt,
    n_substeps=n_substeps,
  )


# ---------------------------------------------------------------------------
# Observation functions (numpy, matching mjlab training observations)
# ---------------------------------------------------------------------------


def compute_joint_pos_normalized(scene: SceneMetadata) -> np.ndarray:
  """Joint positions normalized by soft limits to [-1, 1].

  Matches joint_pos_limit_normalized in src/wuji_mjlab/tasks/reorient/mdp/observations.py.
  """
  pos = scene.joint_pos
  center = 0.5 * (scene.soft_lower + scene.soft_upper)
  half_range = 0.5 * (scene.soft_upper - scene.soft_lower)
  return np.clip((pos - center) / (half_range + 1e-6), -1.0, 1.0).astype(np.float32)


def compute_joint_pos_target_error(
  scene: SceneMetadata, target: np.ndarray
) -> np.ndarray:
  """Normalized joint position error: current_normalized - target_normalized.

  Matches joint_pos_target_error in src/wuji_mjlab/tasks/reorient/mdp/observations.py.
  """
  center = 0.5 * (scene.soft_lower + scene.soft_upper)
  half_range = 0.5 * (scene.soft_upper - scene.soft_lower)

  pos = scene.joint_pos
  norm_pos = np.clip((pos - center) / (half_range + 1e-6), -1.0, 1.0)
  norm_target = np.clip((target - center) / (half_range + 1e-6), -1.0, 1.0)
  return (norm_pos - norm_target).astype(np.float32)


def compute_cube_pos_in_tag(scene: SceneMetadata) -> np.ndarray:
  """Cube position in tag (wrist marker) frame. Shape: (3,).

  Mirrors cube_pos_in_tag in src/wuji_mjlab/tasks/reorient/mdp/observations.py:
    tag_pos_w = palm_pos_w + quat_apply(palm_quat_w, TAG_IN_PALM_POS)
    tag_quat_w = quat_mul(palm_quat_w, TAG_IN_PALM_QUAT_WXYZ)
    cube_pos_in_tag = quat_apply_inverse(tag_quat_w, cube_pos_w - tag_pos_w)

  The tag transform comes from the selected robot binding.
  """
  palm_pos_w = scene.palm_pos
  palm_quat_w = scene.palm_quat
  tag_pos_w = palm_pos_w + quat_apply(palm_quat_w, scene.tag_in_palm_pos)
  tag_quat_w = quat_mul(palm_quat_w, scene.tag_in_palm_quat)
  cube_pos_tag = quat_apply_inverse(tag_quat_w, scene.cube_pos - tag_pos_w)
  return cube_pos_tag.astype(np.float32)


def compute_cube_ori_error_6d(
  scene: SceneMetadata, goal_quat: np.ndarray
) -> np.ndarray:
  """6D rotation error: flatten(rot_matrix)[3:]. Shape: (6,).

  q_err = quat_mul(cube_quat, quat_inv(goal_quat))
  rot = matrix_from_quat(q_err)  → 9 values row-major
  return rot[3:]  → [M10, M11, M12, M20, M21, M22]

  Matches goal_rot_err_6d in src/wuji_mjlab/tasks/reorient/mdp/observations.py.
  """
  tag_quat_inv = quat_inv(quat_mul(scene.palm_quat, scene.tag_in_palm_quat))
  cube_in_tag = quat_mul(tag_quat_inv, scene.cube_quat)
  goal_in_tag = quat_mul(tag_quat_inv, goal_quat)
  q_err = quat_mul(cube_in_tag, quat_inv(goal_in_tag))
  mat = matrix_from_quat(q_err)  # (9,)
  return mat[3:].astype(np.float32)


# ---------------------------------------------------------------------------
# Action control law
# ---------------------------------------------------------------------------


def apply_action(
  scene: SceneMetadata,
  action: np.ndarray,
  prev_target: np.ndarray,
  episode_step: int,
  action_scale: float = 0.5,
  ema_alpha: float = 0.5,
  warmup_time_s: float = 0.4,
) -> np.ndarray:
  """Apply action to scene, returns new prev_target.

  Matches JointPositionOffsetEMAAction.process_actions() in actions.py.

  raw_target = default_pos + clamp(action, -1, 1) * action_scale
  clip to soft limits
  smoothed = ema_alpha * raw_target + (1 - ema_alpha) * prev_target
  During warmup: hold default_pos
  """
  clamped = np.clip(action, -1.0, 1.0)
  raw_target = scene.default_joint_pos + clamped * action_scale
  raw_target = np.clip(raw_target, scene.soft_lower, scene.soft_upper)

  smoothed = ema_alpha * raw_target + (1.0 - ema_alpha) * prev_target

  in_warmup = episode_step * scene.ctrl_dt < warmup_time_s
  if in_warmup:
    processed = scene.default_joint_pos.copy()
  else:
    processed = smoothed

  # Write to ctrl
  scene.data.ctrl[scene.ctrl_ids] = processed
  return processed.copy()


# ---------------------------------------------------------------------------
# Goal management
# ---------------------------------------------------------------------------


def goal_mocap_position_above_cube(
  cube_pos: np.ndarray,
  z_offset: float = _GOAL_VIS_Z_OFFSET,
) -> np.ndarray:
  """Place eval goal visualization directly above the live cube position."""
  vis_pos = np.array(cube_pos, copy=True)
  vis_pos[2] += z_offset
  return vis_pos


def set_goal_mocap(scene: SceneMetadata, goal_quat: np.ndarray) -> None:
  """Set goal mocap pose above the current cube with the target orientation."""
  scene.data.mocap_pos[scene.goal_mocap_id] = goal_mocap_position_above_cube(
    scene.cube_pos
  )
  scene.data.mocap_quat[scene.goal_mocap_id] = goal_quat


@dataclass
class HoldState:
  """Two-phase hold state machine state."""

  hold_counter: int = 0
  in_success_window: bool = False
  window_timer: int = 0


@dataclass
class HoldEvents:
  """One-shot events from a single check_hold step."""

  success_achieved: bool = False
  goal_switched: bool = False


def check_hold(
  ori_error: float,
  state: HoldState,
  threshold: float = 0.2,
  success_hold_steps: int = 5,
  goal_switch_delay: int = 20,
) -> tuple[HoldState, HoldEvents]:
  """Two-phase hold check mirroring InHandReorientCommand._update_command.

  APPROACHING: count consecutive within-threshold steps.
  SUCCESS_WINDOW: wait goal_switch_delay steps then signal goal switch.

  Returns (new_state, events).
  """
  events = HoldEvents()
  within_threshold = ori_error < threshold

  if not state.in_success_window:
    # APPROACHING phase
    if within_threshold:
      state.hold_counter += 1
      if state.hold_counter >= success_hold_steps:
        # Transition to SUCCESS_WINDOW
        state.in_success_window = True
        state.window_timer = 0
        state.hold_counter = 0
        events.success_achieved = True
    else:
      state.hold_counter = 0
  else:
    # SUCCESS_WINDOW phase
    state.window_timer += 1
    if state.window_timer >= goal_switch_delay:
      # Goal switch
      state.in_success_window = False
      state.window_timer = 0
      state.hold_counter = 0
      events.goal_switched = True

  return state, events


# ---------------------------------------------------------------------------
# Scene reset + config loading
# ---------------------------------------------------------------------------


def reset_scene(scene: SceneMetadata) -> None:
  """Reset scene to init state (keyframe 0)."""
  mujoco.mj_resetDataKeyframe(scene.model, scene.data, 0)
  mujoco.mj_forward(scene.model, scene.data)


def load_config(run_dir: str | Path) -> dict:
  """Load config.json from a training run directory.

  Searches in standard locations:
  - <run_dir>/config.json
  - <run_dir>/checkpoints/config.json
  """
  run_dir = Path(run_dir)
  candidates = [
    run_dir / "checkpoints" / "config.json",
    run_dir / "config.json",
  ]
  for p in candidates:
    if p.exists():
      with open(p) as f:
        return json.load(f)
  return {}


# ---------------------------------------------------------------------------
# Contact info (for terminal display)
# ---------------------------------------------------------------------------


def get_contact_info(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
  """Extract active contacts and compute world-frame resultant force/torque."""
  contacts = []
  force_buf = np.zeros(6, dtype=np.float64)
  resultant_force = np.zeros(3, dtype=np.float64)
  resultant_torque = np.zeros(3, dtype=np.float64)

  for i in range(data.ncon):
    c = data.contact[i]
    mujoco.mj_contactForce(model, data, i, force_buf)
    normal_force = force_buf[0]
    total_force = np.linalg.norm(force_buf[:3])
    if total_force < 1e-6:
      continue

    frame = c.frame.reshape(3, 3)
    force_world = frame.T @ force_buf[:3]
    resultant_force += force_world
    resultant_torque += np.cross(c.pos, force_world)

    g1_name = (
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}"
    )
    g2_name = (
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}"
    )
    contacts.append(
      {
        "geom1": g1_name,
        "geom2": g2_name,
        "normal_force": normal_force,
        "total_force": total_force,
        "pos": c.pos.copy(),
      }
    )
  contacts.sort(key=lambda x: x["total_force"], reverse=True)
  return {
    "contacts": contacts,
    "resultant_force": resultant_force,
    "resultant_torque": resultant_torque,
  }
