# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Interactive Revo3 right-hand initial-state tuner.

This tool is deliberately standalone: it launches the MJLab play environment,
applies a locally tuned hand/object state, and can save/reload that state as
YAML. It does not mutate registered task defaults, training configs, rewards,
resets, terminations, randomization, policies, deployment code, or artifacts
unless the user explicitly saves to a chosen path.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

# CI and sandboxed shells can make default import-time cache locations read-only.
# Keep these before importing mjlab.
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import torch
import wuji_mjlab.tasks  # noqa: F401  Registers tasks.
import yaml
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from scripts.play.play_rsl_rl_zero_wrist_pitch import (
  _force_fixed_reorient_goal,
  _force_single_reorient_goal,
  _force_zero_robot_pitch_reset,
)
from wuji_mjlab.tasks.reorient.config.revo3_hand.env_cfgs import (
  REVO3_REORIENT_CUBE_INIT_STATE,
  REVO3_REORIENT_ROBOT_INIT_STATE,
)
from wuji_mjlab.tasks.reorient.robot_bindings import REVO3_RIGHT_HAND_BINDING
from wuji_mjlab.utils.cli_override_utils import DEFAULT_TASK_ALIASES, resolve_task
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

SCHEMA = "revo3_initial_state_tuning_v1"
DEFAULT_TASK = "Revo3RightHand_Reorient_ReposeReward_FineTune"
DEFAULT_OUTPUT = Path("artifacts/revo3_initial_state_tuned.yaml")
ViewerKind = Literal["native", "viser", "none"]


@dataclass
class TuningState:
  task: str
  joint_order: list[str]
  hand_joint_pos_rad: dict[str, float]
  object_position_xyz: list[float]
  object_orientation_xyzw: list[float]
  object_orientation_rpy: list[float]
  collision_enabled: bool
  notes: str = ""


def _finite_float(value: Any, *, name: str) -> float:
  try:
    out = float(value)
  except (TypeError, ValueError) as exc:
    raise ValueError(f"{name} must be a finite float, got {value!r}.") from exc
  if not math.isfinite(out):
    raise ValueError(f"{name} must be finite, got {value!r}.")
  return out


def _finite_vector(values: Any, *, name: str, length: int) -> list[float]:
  if not isinstance(values, (list, tuple)) or len(values) != length:
    raise ValueError(f"{name} must be a length-{length} list.")
  return [_finite_float(v, name=f"{name}[{i}]") for i, v in enumerate(values)]


def _normalize_quat_xyzw(quat: list[float]) -> list[float]:
  norm = math.sqrt(sum(v * v for v in quat))
  if not math.isfinite(norm) or norm <= 1.0e-12:
    raise ValueError("object.orientation_xyzw must have non-zero finite norm.")
  return [v / norm for v in quat]


def _xyzw_to_wxyz(quat_xyzw: list[float]) -> list[float]:
  x, y, z, w = quat_xyzw
  return [w, x, y, z]


def _wxyz_to_xyzw(quat_wxyz: list[float]) -> list[float]:
  w, x, y, z = quat_wxyz
  return [x, y, z, w]


def _rpy_to_xyzw(rpy: list[float]) -> list[float]:
  roll, pitch, yaw = rpy
  cr = math.cos(roll * 0.5)
  sr = math.sin(roll * 0.5)
  cp = math.cos(pitch * 0.5)
  sp = math.sin(pitch * 0.5)
  cy = math.cos(yaw * 0.5)
  sy = math.sin(yaw * 0.5)
  w = cr * cp * cy + sr * sp * sy
  x = sr * cp * cy - cr * sp * sy
  y = cr * sp * cy + sr * cp * sy
  z = cr * cp * sy - sr * sp * cy
  return _normalize_quat_xyzw([x, y, z, w])


def _xyzw_to_rpy(quat_xyzw: list[float]) -> list[float]:
  x, y, z, w = _normalize_quat_xyzw(quat_xyzw)
  sinr_cosp = 2.0 * (w * x + y * z)
  cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
  roll = math.atan2(sinr_cosp, cosr_cosp)

  sinp = 2.0 * (w * y - z * x)
  if abs(sinp) >= 1.0:
    pitch = math.copysign(math.pi / 2.0, sinp)
  else:
    pitch = math.asin(sinp)

  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  yaw = math.atan2(siny_cosp, cosy_cosp)
  return [roll, pitch, yaw]


def _task_default_state(task: str, collision_enabled: bool, notes: str = "") -> TuningState:
  joint_order = list(REVO3_RIGHT_HAND_BINDING.joint_names)
  joint_pos = {}
  for name in joint_order:
    if name not in REVO3_REORIENT_ROBOT_INIT_STATE.joint_pos:
      raise ValueError(f"Task default is missing Revo3 joint '{name}'.")
    joint_pos[name] = float(REVO3_REORIENT_ROBOT_INIT_STATE.joint_pos[name])

  object_quat_xyzw = _wxyz_to_xyzw(list(REVO3_REORIENT_CUBE_INIT_STATE.rot))
  return TuningState(
    task=task,
    joint_order=joint_order,
    hand_joint_pos_rad=joint_pos,
    object_position_xyz=[float(v) for v in REVO3_REORIENT_CUBE_INIT_STATE.pos],
    object_orientation_xyzw=object_quat_xyzw,
    object_orientation_rpy=_xyzw_to_rpy(object_quat_xyzw),
    collision_enabled=collision_enabled,
    notes=notes,
  )


def _state_to_yaml_payload(state: TuningState) -> dict[str, Any]:
  return {
    "schema": SCHEMA,
    "task": state.task,
    "hand_side": "right",
    "units": {
      "joint_position": "rad",
      "object_position": "m",
      "object_orientation": "xyzw quaternion and rpy radians",
    },
    "joint_order": state.joint_order,
    "hand_joint_pos_rad": {
      name: float(state.hand_joint_pos_rad[name]) for name in state.joint_order
    },
    "object": {
      "position_frame": "world",
      "position_xyz": [float(v) for v in state.object_position_xyz],
      "orientation_xyzw": [float(v) for v in state.object_orientation_xyzw],
      "orientation_rpy": [float(v) for v in state.object_orientation_rpy],
    },
    "collision": {
      "enabled": bool(state.collision_enabled),
      "applied_to_simulation": False,
      "implementation": "metadata_only",
      "notes": (
        "Runtime collision toggling is not cleanly supported by this tool; "
        "this field records user intent only."
      ),
    },
    "metadata": {
      "created_by": Path(__file__).name,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "notes": state.notes,
      "not_training_default": True,
    },
  }


def _state_from_yaml_payload(payload: dict[str, Any]) -> TuningState:
  if not isinstance(payload, dict):
    raise ValueError("YAML root must be a mapping.")
  if payload.get("schema") != SCHEMA:
    raise ValueError(f"Expected schema {SCHEMA!r}, got {payload.get('schema')!r}.")

  task = str(payload.get("task") or DEFAULT_TASK)
  joint_order = payload.get("joint_order")
  if not isinstance(joint_order, list):
    raise ValueError("joint_order must be a list.")
  joint_order = [str(name) for name in joint_order]
  expected_order = list(REVO3_RIGHT_HAND_BINDING.joint_names)
  if joint_order != expected_order:
    raise ValueError(
      "YAML joint_order does not match Revo3 policy/tool order.\n"
      f"Expected: {expected_order}\n"
      f"Got:      {joint_order}"
    )

  raw_joint_pos = payload.get("hand_joint_pos_rad")
  if not isinstance(raw_joint_pos, dict):
    raise ValueError("hand_joint_pos_rad must be a mapping.")
  joint_pos: dict[str, float] = {}
  for name in joint_order:
    if name not in raw_joint_pos:
      raise ValueError(f"hand_joint_pos_rad is missing '{name}'.")
    joint_pos[name] = _finite_float(raw_joint_pos[name], name=f"joint {name}")

  object_payload = payload.get("object")
  if not isinstance(object_payload, dict):
    raise ValueError("object must be a mapping.")
  position_xyz = _finite_vector(
    object_payload.get("position_xyz"),
    name="object.position_xyz",
    length=3,
  )

  if "orientation_xyzw" in object_payload:
    quat_xyzw = _normalize_quat_xyzw(
      _finite_vector(
        object_payload.get("orientation_xyzw"),
        name="object.orientation_xyzw",
        length=4,
      )
    )
    rpy = _xyzw_to_rpy(quat_xyzw)
  elif "orientation_rpy" in object_payload:
    rpy = _finite_vector(
      object_payload.get("orientation_rpy"),
      name="object.orientation_rpy",
      length=3,
    )
    quat_xyzw = _rpy_to_xyzw(rpy)
  else:
    raise ValueError("object must contain orientation_xyzw or orientation_rpy.")

  collision_payload = payload.get("collision", {})
  collision_enabled = True
  if isinstance(collision_payload, dict) and "enabled" in collision_payload:
    collision_enabled = bool(collision_payload["enabled"])

  metadata = payload.get("metadata", {})
  notes = ""
  if isinstance(metadata, dict):
    notes = str(metadata.get("notes") or "")

  return TuningState(
    task=task,
    joint_order=joint_order,
    hand_joint_pos_rad=joint_pos,
    object_position_xyz=position_xyz,
    object_orientation_xyzw=quat_xyzw,
    object_orientation_rpy=rpy,
    collision_enabled=collision_enabled,
    notes=notes,
  )


def _load_state(path: Path) -> TuningState:
  with path.open("r", encoding="utf-8") as f:
    payload = yaml.safe_load(f)
  return _state_from_yaml_payload(payload)


def _save_state(path: Path, state: TuningState) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = _state_to_yaml_payload(state)
  with path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(payload, f, sort_keys=False)


def _format_floats(values: list[float]) -> str:
  return "[" + ", ".join(f"{v:+.6f}" for v in values) + "]"


class Revo3InitialStateController:
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    *,
    task: str,
    output: Path,
    collision_enabled: bool,
  ) -> None:
    self.env = env
    self.task = task
    self.output = output
    self.collision_enabled = collision_enabled
    self.collision_applied_to_simulation = False
    self.joint_order = list(REVO3_RIGHT_HAND_BINDING.joint_names)

    robot = self.env.scene["robot"]
    joint_ids, joint_names = robot.find_joints(self.joint_order, preserve_order=True)
    if joint_names != self.joint_order:
      raise ValueError(
        "Resolved Revo3 joint order does not match binding.\n"
        f"Expected: {self.joint_order}\nGot:      {joint_names}"
      )
    self.joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.env.device)
    self.env_id = torch.tensor([0], dtype=torch.long, device=self.env.device)

  def print_joint_order(self) -> None:
    print("[tuner] joint order:")
    for idx, name in enumerate(self.joint_order):
      print(f"  {idx:02d}: {name}")

  def _joint_limits(self) -> tuple[list[float], list[float]]:
    robot = self.env.scene["robot"]
    limits = robot.data.soft_joint_pos_limits[0, self.joint_ids].detach().cpu()
    lower = [float(v) for v in limits[:, 0].tolist()]
    upper = [float(v) for v in limits[:, 1].tolist()]
    return lower, upper

  def joint_slider_ranges(self) -> dict[str, tuple[float, float]]:
    state = self.current_state()
    lower, upper = self._joint_limits()
    ranges: dict[str, tuple[float, float]] = {}
    for idx, name in enumerate(self.joint_order):
      value = float(state.hand_joint_pos_rad[name])
      lo = lower[idx]
      hi = upper[idx]
      if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
        lo, hi = value - 1.0, value + 1.0
      lo = min(lo - 0.25, value - 0.25, -2.5)
      hi = max(hi + 0.25, value + 0.25, 2.5)
      ranges[name] = (lo, hi)
    return ranges

  def validate_state(self, state: TuningState) -> list[str]:
    warnings: list[str] = []
    if state.joint_order != self.joint_order:
      raise ValueError("Tuning state joint_order does not match this Revo3 tool.")
    for name in self.joint_order:
      _finite_float(state.hand_joint_pos_rad[name], name=f"joint {name}")
    _finite_vector(state.object_position_xyz, name="object.position_xyz", length=3)
    state.object_orientation_xyzw = _normalize_quat_xyzw(
      _finite_vector(
        state.object_orientation_xyzw,
        name="object.orientation_xyzw",
        length=4,
      )
    )
    state.object_orientation_rpy = _xyzw_to_rpy(state.object_orientation_xyzw)

    lower, upper = self._joint_limits()
    for idx, name in enumerate(self.joint_order):
      value = float(state.hand_joint_pos_rad[name])
      if value < lower[idx] or value > upper[idx]:
        warnings.append(
          f"{name}={value:+.6f} rad is outside soft limit "
          f"[{lower[idx]:+.6f}, {upper[idx]:+.6f}]"
        )
    return warnings

  def current_state(self) -> TuningState:
    self.env.sim.forward()
    robot = self.env.scene["robot"]
    obj = self.env.scene["object"]
    joint_values = (
      robot.data.joint_pos[0, self.joint_ids].detach().cpu().tolist()
    )
    object_pos = obj.data.root_link_pos_w[0].detach().cpu().tolist()
    object_quat_wxyz = obj.data.root_link_quat_w[0].detach().cpu().tolist()
    object_quat_xyzw = _normalize_quat_xyzw(_wxyz_to_xyzw(object_quat_wxyz))
    return TuningState(
      task=self.task,
      joint_order=list(self.joint_order),
      hand_joint_pos_rad={
        name: float(joint_values[idx]) for idx, name in enumerate(self.joint_order)
      },
      object_position_xyz=[float(v) for v in object_pos],
      object_orientation_xyzw=object_quat_xyzw,
      object_orientation_rpy=_xyzw_to_rpy(object_quat_xyzw),
      collision_enabled=self.collision_enabled,
    )

  def apply_state(self, state: TuningState) -> None:
    warnings = self.validate_state(state)
    for warning in warnings:
      print(f"[tuner][WARN] {warning}", flush=True)

    robot = self.env.scene["robot"]
    obj = self.env.scene["object"]

    joint_tensor = torch.tensor(
      [[state.hand_joint_pos_rad[name] for name in self.joint_order]],
      dtype=torch.float32,
      device=self.env.device,
    )
    zero_joint_vel = torch.zeros_like(joint_tensor)
    robot.write_joint_position_to_sim(
      joint_tensor,
      joint_ids=self.joint_ids,
      env_ids=self.env_id,
    )
    robot.write_joint_velocity_to_sim(
      zero_joint_vel,
      joint_ids=self.joint_ids,
      env_ids=self.env_id,
    )
    robot.set_joint_position_target(
      joint_tensor,
      joint_ids=self.joint_ids,
    )

    object_pose = torch.tensor(
      [state.object_position_xyz + _xyzw_to_wxyz(state.object_orientation_xyzw)],
      dtype=torch.float32,
      device=self.env.device,
    )
    object_vel = torch.zeros((1, 6), dtype=torch.float32, device=self.env.device)
    obj.write_root_link_pose_to_sim(object_pose, env_ids=self.env_id)
    obj.write_root_link_velocity_to_sim(object_vel, env_ids=self.env_id)

    self.env.scene.write_data_to_sim()
    self.env.sim.forward()
    self.env.sim.sense()
    self.collision_enabled = bool(state.collision_enabled)
    if self.env.command_manager.active_terms:
      self.env.command_manager.compute(dt=0.0)

  def apply_task_default_state(self) -> None:
    state = _task_default_state(
      self.task,
      self.collision_enabled,
      notes="Reset by tune_revo3_initial_state.py",
    )
    self.apply_state(state)
    print("[tuner] reset to deterministic Revo3 task initial state.", flush=True)

  def save_current_state(self, path: Path | None = None) -> Path:
    out = path or self.output
    state = self.current_state()
    _save_state(out, state)
    print(f"[tuner] saved tuned state: {out}", flush=True)
    self.print_state_summary(state)
    print(
      "[tuner] collision toggle: metadata-only; not applied to simulation.",
      flush=True,
    )
    return out

  def load_and_apply(self, path: Path) -> None:
    state = _load_state(path)
    if state.task != self.task:
      print(
        f"[tuner][WARN] YAML task is {state.task!r}; current task is {self.task!r}.",
        flush=True,
      )
      state.task = self.task
    self.apply_state(state)
    self.collision_enabled = state.collision_enabled
    print(f"[tuner] loaded and applied tuned state: {path}", flush=True)
    print(
      "[tuner] collision toggle from YAML is metadata-only; not applied to simulation.",
      flush=True,
    )

  def print_state_summary(self, state: TuningState | None = None) -> None:
    state = state or self.current_state()
    print("[tuner] current state summary:", flush=True)
    print(f"  task: {state.task}", flush=True)
    print("  object.position_frame: world", flush=True)
    print(f"  object.position_xyz: {_format_floats(state.object_position_xyz)}", flush=True)
    print(
      f"  object.orientation_xyzw: {_format_floats(state.object_orientation_xyzw)}",
      flush=True,
    )
    print(
      f"  object.orientation_rpy_rad: {_format_floats(state.object_orientation_rpy)}",
      flush=True,
    )
    print(f"  collision.enabled_metadata: {state.collision_enabled}", flush=True)
    print("  hand_joint_pos_rad:", flush=True)
    for idx, name in enumerate(state.joint_order):
      print(f"    {idx:02d} {name}: {state.hand_joint_pos_rad[name]:+.6f}", flush=True)

  def set_joint(self, key: str, value: float) -> None:
    state = self.current_state()
    if key.isdigit():
      idx = int(key)
      if idx < 0 or idx >= len(self.joint_order):
        raise ValueError(f"Joint index out of range: {idx}")
      name = self.joint_order[idx]
    else:
      name = key
      if name not in self.joint_order:
        raise ValueError(f"Unknown Revo3 joint '{name}'.")
    state.hand_joint_pos_rad[name] = value
    self.apply_state(state)
    print(f"[tuner] set {name} = {value:+.6f} rad", flush=True)

  def set_position(self, xyz: list[float]) -> None:
    state = self.current_state()
    state.object_position_xyz = xyz
    self.apply_state(state)
    print(f"[tuner] set object.position_xyz = {_format_floats(xyz)}", flush=True)

  def set_position_axis(self, axis: str, value: float) -> None:
    axis_to_idx = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_to_idx:
      raise ValueError("axis must be x, y, or z.")
    state = self.current_state()
    state.object_position_xyz[axis_to_idx[axis]] = value
    self.apply_state(state)
    print(f"[tuner] set object.position_{axis} = {value:+.6f} m", flush=True)

  def set_quat_xyzw(self, quat_xyzw: list[float]) -> None:
    state = self.current_state()
    state.object_orientation_xyzw = _normalize_quat_xyzw(quat_xyzw)
    state.object_orientation_rpy = _xyzw_to_rpy(state.object_orientation_xyzw)
    self.apply_state(state)
    print(
      "[tuner] set object.orientation_xyzw = "
      + _format_floats(state.object_orientation_xyzw),
      flush=True,
    )

  def set_rpy(self, rpy: list[float]) -> None:
    state = self.current_state()
    state.object_orientation_rpy = rpy
    state.object_orientation_xyzw = _rpy_to_xyzw(rpy)
    self.apply_state(state)
    print(f"[tuner] set object.orientation_rpy_rad = {_format_floats(rpy)}", flush=True)

  def set_collision_metadata(self, enabled: bool) -> None:
    self.collision_enabled = enabled
    print(
      "[tuner] collision.enabled metadata set to "
      f"{enabled}; simulation collision model unchanged.",
      flush=True,
    )


class _TuningViewerMixin:
  controller: Revo3InitialStateController

  def _handle_tuning_payload(self, payload: Any) -> bool:
    if not isinstance(payload, dict) or payload.get("type") != "tuner":
      return False
    try:
      self._dispatch_tuning_payload(payload)
    except Exception as exc:
      print(f"[tuner][ERROR] {exc}", flush=True)
    return True

  def _dispatch_tuning_payload(self, payload: dict[str, Any]) -> None:
    op = payload.get("op")
    if op == "show":
      self.controller.print_state_summary()
    elif op == "joint_order":
      self.controller.print_joint_order()
    elif op == "set_joint":
      self.controller.set_joint(
        str(payload["joint"]),
        _finite_float(payload["value"], name="joint value"),
      )
    elif op == "set_position":
      self.controller.set_position(payload["xyz"])
    elif op == "set_position_axis":
      self.controller.set_position_axis(
        str(payload["axis"]),
        _finite_float(payload["value"], name="position value"),
      )
    elif op == "set_quat":
      self.controller.set_quat_xyzw(payload["quat_xyzw"])
    elif op == "set_rpy":
      self.controller.set_rpy(payload["rpy"])
    elif op == "apply_state":
      state = self.controller.current_state()
      raw_joints = payload.get("hand_joint_pos_rad", {})
      if not isinstance(raw_joints, dict):
        raise ValueError("apply_state hand_joint_pos_rad must be a mapping.")
      for name in self.controller.joint_order:
        if name in raw_joints:
          state.hand_joint_pos_rad[name] = _finite_float(
            raw_joints[name],
            name=f"joint {name}",
          )
      if "object_position_xyz" in payload:
        state.object_position_xyz = _finite_vector(
          payload["object_position_xyz"],
          name="object_position_xyz",
          length=3,
        )
      if "object_orientation_rpy" in payload:
        state.object_orientation_rpy = _finite_vector(
          payload["object_orientation_rpy"],
          name="object_orientation_rpy",
          length=3,
        )
        state.object_orientation_xyzw = _rpy_to_xyzw(state.object_orientation_rpy)
      if "collision_enabled" in payload:
        state.collision_enabled = bool(payload["collision_enabled"])
      self.controller.apply_state(state)
    elif op == "reset_task":
      self.controller.apply_task_default_state()
    elif op == "save":
      raw_path = payload.get("path")
      path = Path(raw_path) if raw_path else None
      self.controller.save_current_state(path)
    elif op == "load":
      self.controller.load_and_apply(Path(str(payload["path"])))
    elif op == "collision":
      self.controller.set_collision_metadata(bool(payload["enabled"]))
    elif op == "exit":
      self._interrupted = True
    else:
      raise ValueError(f"Unsupported tuner operation: {op!r}")


class TuningNativeViewer(_TuningViewerMixin, NativeMujocoViewer):
  def __init__(self, *args, controller: Revo3InitialStateController, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.controller = controller

  def setup(self) -> None:
    super().setup()
    self.pause()
    print("[tuner] native viewer launched paused.", flush=True)
    print("[tuner] type 'help' in this terminal for tuning commands.", flush=True)

  def _handle_custom_action(self, action, payload) -> bool:
    if self._handle_tuning_payload(payload):
      return True
    return super()._handle_custom_action(action, payload)


class TuningViserViewer(_TuningViewerMixin, ViserPlayViewer):
  def __init__(self, *args, controller: Revo3InitialStateController, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.controller = controller
    self._tuner_gui_ready = False
    self._tuner_syncing_gui = False
    self._tuner_joint_sliders: dict[str, Any] = {}
    self._tuner_position_sliders: dict[str, Any] = {}
    self._tuner_rpy_deg_sliders: dict[str, Any] = {}
    self._tuner_live_apply = None
    self._tuner_collision_checkbox = None
    self._tuner_yaml_path = None

  def setup(self) -> None:
    super().setup()
    self.pause()
    self._create_tuner_gui()
    self._tuner_gui_ready = True
    self._sync_ui_state()
    print("[tuner] Viser viewer launched paused.", flush=True)
    print(
      "[tuner] Viser page includes Initial State Tuner sliders.",
      flush=True,
    )
    print("[tuner] terminal commands are still available; type 'help'.", flush=True)

  def _create_tuner_gui(self) -> None:
    state = self.controller.current_state()
    joint_ranges = self.controller.joint_slider_ranges()

    with self._server.gui.add_folder("Initial State Tuner", expand_by_default=True):
      self._tuner_live_apply = self._server.gui.add_checkbox(
        "Live Apply",
        initial_value=True,
      )
      self._tuner_yaml_path = self._server.gui.add_text(
        "YAML path",
        initial_value=str(self.controller.output),
      )

      with self._server.gui.add_folder("Hand Joints", expand_by_default=True):
        for idx, name in enumerate(self.controller.joint_order):
          lo, hi = joint_ranges[name]
          slider = self._server.gui.add_slider(
            f"{idx:02d} {name}",
            min=lo,
            max=hi,
            step=0.001,
            initial_value=float(state.hand_joint_pos_rad[name]),
            hint="rad",
          )
          self._tuner_joint_sliders[name] = slider

          @slider.on_update
          def _(_, joint_name=name, handle=slider) -> None:
            if self._tuner_syncing_gui or not self._live_apply_enabled():
              return
            self.request_action(
              "CUSTOM",
              {
                "type": "tuner",
                "op": "set_joint",
                "joint": joint_name,
                "value": float(handle.value),
              },
            )

      with self._server.gui.add_folder("Object Position", expand_by_default=True):
        position_ranges = {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 1.2),
        }
        for idx, axis in enumerate(("x", "y", "z")):
          lo, hi = position_ranges[axis]
          value = float(state.object_position_xyz[idx])
          slider = self._server.gui.add_slider(
            f"{axis} m",
            min=min(lo, value - 0.05),
            max=max(hi, value + 0.05),
            step=0.001,
            initial_value=value,
          )
          self._tuner_position_sliders[axis] = slider

          @slider.on_update
          def _(_) -> None:
            if self._tuner_syncing_gui or not self._live_apply_enabled():
              return
            self.request_action(
              "CUSTOM",
              {
                "type": "tuner",
                "op": "set_position",
                "xyz": self._position_xyz_from_gui(),
              },
            )

      with self._server.gui.add_folder("Object Orientation", expand_by_default=True):
        for idx, axis in enumerate(("roll", "pitch", "yaw")):
          value_deg = math.degrees(float(state.object_orientation_rpy[idx]))
          slider = self._server.gui.add_slider(
            f"{axis} deg",
            min=-180.0,
            max=180.0,
            step=0.5,
            initial_value=value_deg,
          )
          self._tuner_rpy_deg_sliders[axis] = slider

          @slider.on_update
          def _(_) -> None:
            if self._tuner_syncing_gui or not self._live_apply_enabled():
              return
            self.request_action(
              "CUSTOM",
              {
                "type": "tuner",
                "op": "set_rpy",
                "rpy": self._rpy_rad_from_gui(),
              },
            )

      self._tuner_collision_checkbox = self._server.gui.add_checkbox(
        "Collision Metadata",
        initial_value=state.collision_enabled,
        hint="Saved to YAML only; simulation collision model is unchanged.",
      )

      @self._tuner_collision_checkbox.on_update
      def _(_) -> None:
        if self._tuner_syncing_gui:
          return
        self.request_action(
          "CUSTOM",
          {
            "type": "tuner",
            "op": "collision",
            "enabled": bool(self._tuner_collision_checkbox.value),
          },
        )

      apply_button = self._server.gui.add_button("Apply Sliders")

      @apply_button.on_click
      def _(_) -> None:
        self._request_apply_gui_state()

      save_button = self._server.gui.add_button("Save YAML")

      @save_button.on_click
      def _(_) -> None:
        self.request_action(
          "CUSTOM",
          {
            "type": "tuner",
            "op": "save",
            "path": self._yaml_path_from_gui(),
          },
        )

      load_button = self._server.gui.add_button("Load YAML")

      @load_button.on_click
      def _(_) -> None:
        self.request_action(
          "CUSTOM",
          {
            "type": "tuner",
            "op": "load",
            "path": self._yaml_path_from_gui(),
          },
        )

      reset_button = self._server.gui.add_button("Reset Task State")

      @reset_button.on_click
      def _(_) -> None:
        self.request_action("CUSTOM", {"type": "tuner", "op": "reset_task"})

      summary_button = self._server.gui.add_button("Print Summary")

      @summary_button.on_click
      def _(_) -> None:
        self.request_action("CUSTOM", {"type": "tuner", "op": "show"})

  def _live_apply_enabled(self) -> bool:
    return self._tuner_live_apply is None or bool(self._tuner_live_apply.value)

  def _yaml_path_from_gui(self) -> str:
    if self._tuner_yaml_path is None:
      return str(self.controller.output)
    path = str(self._tuner_yaml_path.value).strip()
    return path or str(self.controller.output)

  def _position_xyz_from_gui(self) -> list[float]:
    return [
      float(self._tuner_position_sliders[axis].value) for axis in ("x", "y", "z")
    ]

  def _rpy_rad_from_gui(self) -> list[float]:
    return [
      math.radians(float(self._tuner_rpy_deg_sliders[axis].value))
      for axis in ("roll", "pitch", "yaw")
    ]

  def _request_apply_gui_state(self) -> None:
    self.request_action(
      "CUSTOM",
      {
        "type": "tuner",
        "op": "apply_state",
        "hand_joint_pos_rad": {
          name: float(slider.value)
          for name, slider in self._tuner_joint_sliders.items()
        },
        "object_position_xyz": self._position_xyz_from_gui(),
        "object_orientation_rpy": self._rpy_rad_from_gui(),
        "collision_enabled": (
          bool(self._tuner_collision_checkbox.value)
          if self._tuner_collision_checkbox is not None
          else self.controller.collision_enabled
        ),
      },
    )

  def _sync_tuner_gui_state(self) -> None:
    if not self._tuner_gui_ready:
      return
    with self._sim_lock:
      state = self.controller.current_state()
    self._tuner_syncing_gui = True
    try:
      for name, slider in self._tuner_joint_sliders.items():
        slider.value = float(state.hand_joint_pos_rad[name])
      for idx, axis in enumerate(("x", "y", "z")):
        self._tuner_position_sliders[axis].value = float(
          state.object_position_xyz[idx]
        )
      for idx, axis in enumerate(("roll", "pitch", "yaw")):
        self._tuner_rpy_deg_sliders[axis].value = math.degrees(
          float(state.object_orientation_rpy[idx])
        )
      if self._tuner_collision_checkbox is not None:
        self._tuner_collision_checkbox.value = bool(state.collision_enabled)
    finally:
      self._tuner_syncing_gui = False

  def _sync_ui_state(self) -> None:
    super()._sync_ui_state()
    self._sync_tuner_gui_state()

  def _handle_custom_action(self, action, payload) -> bool:
    if isinstance(payload, dict) and payload.get("type") == "tuner":
      with self._sim_lock:
        handled = self._handle_tuning_payload(payload)
      if handled and hasattr(self, "_scene"):
        self._scene.request_update()
      return True
    return super()._handle_custom_action(action, payload)


class PolicyZero:
  def __init__(self, env: RslRlVecEnvWrapper) -> None:
    self._env = env
    self._action_shape: tuple[int, ...] = env.unwrapped.action_space.shape

  def __call__(self, obs) -> torch.Tensor:
    del obs
    return torch.zeros(self._action_shape, device=self._env.unwrapped.device)


def _help_text() -> str:
  return """Commands:
  help
  show
  joint-order
  joint <index|name> <rad>
  pos <x> <y> <z>
  pos-x <m> | pos-y <m> | pos-z <m>
  rpy <roll> <pitch> <yaw>           # radians
  rpy-deg <roll> <pitch> <yaw>       # degrees
  quat <x> <y> <z> <w>
  collision on|off                   # metadata only
  reset-task
  save [path]
  load <path>
  quit
"""


def _parse_command(line: str, default_output: Path) -> dict[str, Any] | None:
  parts = line.strip().split()
  if not parts:
    return None
  cmd = parts[0].lower()

  if cmd == "help":
    print(_help_text(), flush=True)
    return None
  if cmd == "show":
    return {"type": "tuner", "op": "show"}
  if cmd in {"joint-order", "joints"}:
    return {"type": "tuner", "op": "joint_order"}
  if cmd in {"joint", "j"}:
    if len(parts) != 3:
      raise ValueError("usage: joint <index|name> <rad>")
    return {
      "type": "tuner",
      "op": "set_joint",
      "joint": parts[1],
      "value": _finite_float(parts[2], name="joint value"),
    }
  if cmd == "pos":
    if len(parts) != 4:
      raise ValueError("usage: pos <x> <y> <z>")
    return {
      "type": "tuner",
      "op": "set_position",
      "xyz": [_finite_float(v, name="position") for v in parts[1:4]],
    }
  if cmd in {"pos-x", "pos-y", "pos-z"}:
    if len(parts) != 2:
      raise ValueError(f"usage: {cmd} <m>")
    return {
      "type": "tuner",
      "op": "set_position_axis",
      "axis": cmd[-1],
      "value": _finite_float(parts[1], name=cmd),
    }
  if cmd == "rpy":
    if len(parts) != 4:
      raise ValueError("usage: rpy <roll> <pitch> <yaw>")
    return {
      "type": "tuner",
      "op": "set_rpy",
      "rpy": [_finite_float(v, name="rpy") for v in parts[1:4]],
    }
  if cmd == "rpy-deg":
    if len(parts) != 4:
      raise ValueError("usage: rpy-deg <roll> <pitch> <yaw>")
    return {
      "type": "tuner",
      "op": "set_rpy",
      "rpy": [
        math.radians(_finite_float(v, name="rpy-deg")) for v in parts[1:4]
      ],
    }
  if cmd == "quat":
    if len(parts) != 5:
      raise ValueError("usage: quat <x> <y> <z> <w>")
    return {
      "type": "tuner",
      "op": "set_quat",
      "quat_xyzw": [_finite_float(v, name="quat") for v in parts[1:5]],
    }
  if cmd == "collision":
    if len(parts) != 2 or parts[1].lower() not in {"on", "off", "true", "false"}:
      raise ValueError("usage: collision on|off")
    enabled = parts[1].lower() in {"on", "true"}
    return {"type": "tuner", "op": "collision", "enabled": enabled}
  if cmd == "reset-task":
    return {"type": "tuner", "op": "reset_task"}
  if cmd == "save":
    if len(parts) > 2:
      raise ValueError("usage: save [path]")
    return {
      "type": "tuner",
      "op": "save",
      "path": str(Path(parts[1])) if len(parts) == 2 else str(default_output),
    }
  if cmd == "load":
    if len(parts) != 2:
      raise ValueError("usage: load <path>")
    return {"type": "tuner", "op": "load", "path": parts[1]}
  if cmd in {"quit", "exit"}:
    return {"type": "tuner", "op": "exit"}

  raise ValueError(f"Unknown command {cmd!r}. Type 'help'.")


def _start_terminal_thread(viewer, output: Path) -> threading.Thread | None:
  if not sys.stdin.isatty():
    print("[tuner] stdin is not interactive; terminal controls disabled.", flush=True)
    return None

  def loop() -> None:
    print(_help_text(), flush=True)
    while True:
      try:
        line = input("tuner> ")
      except EOFError:
        viewer.request_action("CUSTOM", {"type": "tuner", "op": "exit"})
        return
      try:
        payload = _parse_command(line, output)
      except Exception as exc:
        print(f"[tuner][ERROR] {exc}", flush=True)
        continue
      if payload is None:
        continue
      viewer.request_action("CUSTOM", payload)
      if payload.get("op") == "exit":
        return

  thread = threading.Thread(target=loop, daemon=True, name="revo3-tuner-input")
  thread.start()
  return thread


def _prepare_env_cfg(task_id: str, args: argparse.Namespace) -> tuple[Any, Any]:
  env_cfg, agent_cfg = prepare_task_cfgs(task_id, [], play=True)
  env_cfg.scene.num_envs = args.num_envs

  if args.zero_wrist_pitch_reset:
    _force_zero_robot_pitch_reset(env_cfg)
  if args.single_goal:
    _force_single_reorient_goal(env_cfg)
  if args.fixed_goal:
    _force_fixed_reorient_goal(
      env_cfg,
      axis=args.fixed_goal_axis,
      deg=args.fixed_goal_deg,
    )
  if args.no_terminations:
    env_cfg.terminations = {}
  return env_cfg, agent_cfg


def _build_env(task_id: str, args: argparse.Namespace) -> tuple[RslRlVecEnvWrapper, Any]:
  env_cfg, agent_cfg = _prepare_env_cfg(task_id, args)
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  return wrapped, env_cfg


def _reset_and_apply_initial_state(
  wrapped: RslRlVecEnvWrapper,
  controller: Revo3InitialStateController,
  *,
  load_path: Path | None,
) -> None:
  wrapped.reset()
  if load_path is not None:
    controller.load_and_apply(load_path)
  else:
    controller.apply_task_default_state()


def run(args: argparse.Namespace) -> None:
  if args.num_envs != 1:
    raise ValueError("This tuner is intentionally scoped to --num-envs 1.")

  task_id = resolve_task(
    args.task,
    task_aliases=DEFAULT_TASK_ALIASES,
    all_tasks=list_tasks(),
  )
  if not task_id.startswith("Revo3RightHand_"):
    raise ValueError(f"This Revo3 tuner does not support non-Revo3 task {task_id!r}.")

  print(f"[tuner] task: {task_id}", flush=True)
  print(f"[tuner] viewer: {args.viewer}", flush=True)
  print(
    "[tuner] collision toggle: metadata-only; simulation collision model unchanged.",
    flush=True,
  )

  wrapped, _env_cfg = _build_env(task_id, args)
  controller = Revo3InitialStateController(
    wrapped.unwrapped,
    task=task_id,
    output=args.output,
    collision_enabled=args.collision_enabled,
  )
  controller.print_joint_order()

  try:
    _reset_and_apply_initial_state(wrapped, controller, load_path=args.load)
    controller.print_state_summary()
    if args.save_on_exit:
      controller.save_current_state(args.output)
    if args.exit_after_init or args.viewer == "none":
      return

    policy = PolicyZero(wrapped)
    if args.viewer == "native":
      viewer = TuningNativeViewer(wrapped, policy, controller=controller)
    elif args.viewer == "viser":
      viewer = TuningViserViewer(wrapped, policy, controller=controller)
    else:
      raise RuntimeError(f"Unsupported viewer: {args.viewer}")

    _start_terminal_thread(viewer, args.output)
    viewer.run()
  finally:
    wrapped.close()


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default=DEFAULT_TASK)
  parser.add_argument("--viewer", choices=("native", "viser", "none"), default="native")
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--device", default=None)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--load", type=Path, default=None, help="YAML state to load.")
  parser.add_argument("--save-on-exit", action="store_true")
  parser.add_argument("--exit-after-init", action="store_true")
  parser.add_argument("--fixed-goal", action="store_true")
  parser.add_argument("--single-goal", action="store_true")
  parser.add_argument("--fixed-goal-axis", choices=("x", "y", "z"), default="z")
  parser.add_argument("--fixed-goal-deg", type=float, default=90.0)
  parser.add_argument(
    "--allow-wrist-pitch-reset",
    dest="zero_wrist_pitch_reset",
    action="store_false",
    default=True,
    help="Do not apply the play-script zero wrist-pitch reset override.",
  )
  parser.add_argument(
    "--allow-terminations",
    dest="no_terminations",
    action="store_false",
    default=True,
    help="Keep task terminations enabled in the local tuner env.",
  )
  parser.add_argument(
    "--collision-enabled",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Record collision intent in YAML. Runtime physics toggle is metadata-only.",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  run(args)


if __name__ == "__main__":
  main()
