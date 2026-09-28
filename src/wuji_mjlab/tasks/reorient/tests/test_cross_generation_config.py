# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator

import mujoco
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from wuji_mjlab.tasks.reorient.config.wuji_hand.env_cfgs import (
  wuji_hand_reorient_env_cfg,
)
from wuji_mjlab.tasks.reorient.config.wuji_hand2 import _reorient_hand2_50hz_env_cfg
from wuji_mjlab.tasks.reorient.config.wuji_hand2.env_cfgs import (
  wuji_hand2_reorient_env_cfg,
)
from wuji_mjlab.tasks.reorient.mdp.metrics import torque_saturation_ratio
from wuji_mjlab.tasks.reorient.mdp.observations import joint_pos_target_error
from wuji_mjlab.tasks.reorient.tooling.scene_builder import build_reorient_scene

_GENERATIONS = ("hand1", "hand2")
_NAME_FIELDS = (
  "body_names",
  "geom_names",
  "site_names",
  "joint_names",
  "actuator_names",
  "tendon_names",
)
_OBJECT_TYPES = {
  "body_names": mujoco.mjtObj.mjOBJ_BODY,
  "geom_names": mujoco.mjtObj.mjOBJ_GEOM,
  "site_names": mujoco.mjtObj.mjOBJ_SITE,
  "joint_names": mujoco.mjtObj.mjOBJ_JOINT,
  "actuator_names": mujoco.mjtObj.mjOBJ_ACTUATOR,
  "tendon_names": mujoco.mjtObj.mjOBJ_TENDON,
  "contact-body": mujoco.mjtObj.mjOBJ_BODY,
  "contact-geom": mujoco.mjtObj.mjOBJ_GEOM,
  "contact-subtree": mujoco.mjtObj.mjOBJ_BODY,
}


def _make_cfg(generation: str, num_envs: int) -> ManagerBasedRlEnvCfg:
  if generation == "hand1":
    return wuji_hand_reorient_env_cfg(num_envs=num_envs)
  if generation == "hand2":
    return wuji_hand2_reorient_env_cfg(num_envs=num_envs)
  raise ValueError(f"unsupported generation {generation!r}")


def _iter_cfg_nodes(obj: object, seen: set[int] | None = None) -> Iterator[object]:
  if seen is None:
    seen = set()
  if id(obj) in seen:
    return
  seen.add(id(obj))
  yield obj

  if dataclasses.is_dataclass(obj):
    for field in dataclasses.fields(obj):
      yield from _iter_cfg_nodes(getattr(obj, field.name), seen)
  elif isinstance(obj, dict):
    for value in obj.values():
      yield from _iter_cfg_nodes(value, seen)
  elif isinstance(obj, (list, tuple, set, frozenset)):
    for value in obj:
      yield from _iter_cfg_nodes(value, seen)


def _as_patterns(value: object) -> tuple[str, ...]:
  if isinstance(value, str):
    return (value,)
  if isinstance(value, (list, tuple)):
    return tuple(value)
  return ()


def _collect_name_patterns(cfg: ManagerBasedRlEnvCfg) -> list[tuple[str, str, str]]:
  patterns: set[tuple[str, str, str]] = set()
  for node in _iter_cfg_nodes(cfg):
    node_type = type(node).__name__
    if node_type == "SceneEntityCfg":
      entity = getattr(node, "name", "?")
      for field_name in _NAME_FIELDS:
        for pattern in _as_patterns(getattr(node, field_name, None)):
          patterns.add((entity, field_name, pattern))
    elif node_type == "ContactMatch":
      entity = getattr(node, "entity", "?")
      mode = getattr(node, "mode", "?")
      for pattern in _as_patterns(getattr(node, "pattern", None)):
        patterns.add((entity, f"contact-{mode}", pattern))
  return sorted(patterns)


def _model_names(model: mujoco.MjModel, kind: str) -> tuple[str, ...]:
  try:
    object_type = _OBJECT_TYPES[kind]
  except KeyError as exc:
    raise AssertionError(
      f"name-pattern scanner does not support kind {kind!r}"
    ) from exc

  counts = {
    mujoco.mjtObj.mjOBJ_BODY: model.nbody,
    mujoco.mjtObj.mjOBJ_GEOM: model.ngeom,
    mujoco.mjtObj.mjOBJ_SITE: model.nsite,
    mujoco.mjtObj.mjOBJ_JOINT: model.njnt,
    mujoco.mjtObj.mjOBJ_ACTUATOR: model.nu,
    mujoco.mjtObj.mjOBJ_TENDON: model.ntendon,
  }
  return tuple(
    mujoco.mj_id2name(model, object_type, index) or ""
    for index in range(counts[object_type])
  )


@pytest.mark.parametrize("generation", _GENERATIONS)
def test_initial_action_target_matches_default_pose(generation: str) -> None:
  env = ManagerBasedRlEnv(
    cfg=_make_cfg(generation, num_envs=2),
    device="cpu",
    render_mode=None,
  )
  try:
    torch.testing.assert_close(
      env.action_manager.get_term("joint_pos").processed_action,
      env.scene["robot"].data.default_joint_pos,
    )
  finally:
    env.close()


@pytest.mark.parametrize("generation", _GENERATIONS)
def test_every_reorient_name_pattern_matches_each_generation(generation: str) -> None:
  cfg = _make_cfg(generation, num_envs=2)
  model = build_reorient_scene(hand=generation).model
  missing: list[tuple[str, str, str]] = []

  for entity, kind, pattern in _collect_name_patterns(cfg):
    names = _model_names(model, kind)
    prefix = f"{entity}/" if entity in ("robot", "object") else ""
    candidates = (
      name.removeprefix(prefix)
      for name in names
      if name and (not prefix or name.startswith(prefix))
    )
    if not any(re.fullmatch(pattern, name) for name in candidates):
      missing.append((entity, kind, pattern))

  assert not missing, "name patterns matched no model element:\n" + "\n".join(
    f"[{generation}] {entity} {kind}: {pattern}" for entity, kind, pattern in missing
  )


@pytest.fixture(
  scope="module",
  params=[
    wuji_hand_reorient_env_cfg,
    wuji_hand2_reorient_env_cfg,
    _reorient_hand2_50hz_env_cfg,
  ],
  ids=["hand1", "hand2-20hz", "hand2-50hz"],
)
def play_env(request):
  cfg = request.param(play=True)
  cfg.scene.num_envs = 2
  cfg.seed = 20260927
  cfg.commands["reorient_command"].debug_vis = False
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
  try:
    yield env
  finally:
    env.close()


@pytest.mark.parametrize("previous_episode", [False, True])
def test_reset_restores_target_and_entire_qpos_error_history(
  play_env, previous_episode
):
  env = play_env
  robot = env.scene["robot"]
  action = env.action_manager.get_term("joint_pos")
  if previous_episode:
    env.episode_length_buf.fill_(action._warmup_steps)
    action.process_actions(torch.full((2, 20), 0.8))
  obs, _ = env.reset(seed=20260927)
  limits = robot.data.soft_joint_pos_limits
  center = limits.mean(dim=-1)
  half = (limits[..., 1] - limits[..., 0]) * 0.5
  expected = ((robot.data.joint_pos - center) / (half + 1e-6)).clamp(-1, 1) - (
    (robot.data.default_joint_pos - center) / (half + 1e-6)
  ).clamp(-1, 1)
  torch.testing.assert_close(joint_pos_target_error(env), expected)
  torch.testing.assert_close(action.processed_action, robot.data.default_joint_pos)
  for group in ("policy", "critic"):
    names = env.observation_manager.active_terms[group]
    dims = env.observation_manager.group_obs_term_dim[group]
    index = names.index("qpos_error")
    offset = sum(dim[0] for dim in dims[:index])
    width = dims[index][0]
    num_joints = expected.shape[-1]
    history_length = width // num_joints
    actual = obs[group][:, offset : offset + width].reshape(
      env.num_envs, history_length, num_joints
    )
    torch.testing.assert_close(
      actual, expected[:, None, :].expand(-1, history_length, -1)
    )


def test_partial_env_reset_preserves_other_world_command_and_target(play_env):
  env = play_env
  env.reset(seed=20260927)
  command = env.command_manager.get_term("reorient_command")
  action = env.action_manager.get_term("joint_pos")
  env.episode_length_buf.fill_(action._warmup_steps)
  action.process_actions(torch.full((2, 20), 0.8))
  target_before = action.processed_action[1].clone()
  command.goal_timer[:] = torch.tensor([31, 79])
  command.hold_counter[:] = 2
  command.window_timer[:] = 3
  before = {
    name: value[1].clone()
    for name, value in vars(command).items()
    if isinstance(value, torch.Tensor) and value.shape[0] == 2
  }
  env.reset(env_ids=torch.tensor([0]))
  for name, value in before.items():
    torch.testing.assert_close(getattr(command, name)[1], value, msg=name)
  torch.testing.assert_close(action.processed_action[1], target_before)
  torch.testing.assert_close(
    action.processed_action[0], env.scene["robot"].data.default_joint_pos[0]
  )


def test_torque_saturation_detects_real_bounded_forces(play_env):
  env = play_env
  env.reset(seed=20260927)
  robot = env.scene["robot"]
  action = env.action_manager.get_term("joint_pos")
  q = robot.data.soft_joint_pos_limits[:, :, 0] + 0.02
  robot.write_joint_state_to_sim(q, torch.zeros_like(q))
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.episode_length_buf.fill_(action._warmup_steps)
  limits = torch.as_tensor(env.sim.mj_model.actuator_forcerange).abs().amax(dim=-1)
  limits = limits[robot.indexing.ctrl_ids]
  actual, expected = [], []
  for _ in range(5):
    env.step(torch.ones(2, 20))
    for ids in (slice(None), [0, 3, 7, 11, 19]):
      cfg = SceneEntityCfg("robot", actuator_ids=ids)
      actual.append(torque_saturation_ratio(env, asset_cfg=cfg))
      expected.append(
        (robot.data.actuator_force[:, ids].abs() > 0.9 * limits[ids])
        .float()
        .mean(dim=-1)
      )
  actual, expected = torch.stack(actual), torch.stack(expected)
  assert torch.any(expected > 0)
  torch.testing.assert_close(actual, expected)
