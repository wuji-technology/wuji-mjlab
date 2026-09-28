# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Correctness gate: numpy obs_builder / ActionPostprocessor vs the mjlab sim."""

from __future__ import annotations

import argparse
import importlib
import sys

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_inv, quat_mul
from wuji_reorient_deploy.constants import load_constants
from wuji_reorient_deploy.obs_builder import ActionPostprocessor, ObsAssembler

from wuji_mjlab.assets.robots.wuji_hand.wuji_hand_cfg import wuji_hand_xml_path

_GEN1_ASSET_XML = wuji_hand_xml_path("right")


def _skip_if_gen1_asset_missing(gen: int) -> None:
  if gen == 1 and not _GEN1_ASSET_XML.is_file():
    print(
      "SKIP: gen1 asset file is unavailable "
      f"({_GEN1_ASSET_XML}); this environmental prerequisite is the only "
      "allowed gen1 skip"
    )
    sys.exit(0)


def _tag_frame(env, robot):
  from wuji_mjlab.tasks.reorient.mdp.observations import (
    _DEFAULT_TAG_CFG,
    _resolve_tag_site_ids,
    tag_pose_w,
  )

  site_ids = _resolve_tag_site_ids(robot, _DEFAULT_TAG_CFG)
  tag_pos_w, tag_quat_w = tag_pose_w(robot, site_ids)
  return tag_pos_w, tag_quat_w


def _build_cfg(gen: int, hand: str):
  if gen == 2:
    from wuji_mjlab.tasks.reorient.config.wuji_hand2.env_cfgs import (
      wuji_hand2_reorient_env_cfg,
    )

    return wuji_hand2_reorient_env_cfg(hand_side=hand, play=True, num_envs=1)
  from wuji_mjlab.tasks.reorient.config.wuji_hand.env_cfgs import (
    wuji_hand_reorient_env_cfg,
  )

  return wuji_hand_reorient_env_cfg(play=True, num_envs=1)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument(
    "--gen", type=int, choices=(1, 2), default=2, help="hand generation (default: 2)"
  )
  ap.add_argument("--hand", choices=["right"], default="right")
  ap.add_argument("--steps", type=int, default=60)
  ap.add_argument("--task", help="registered task ID to validate in play mode")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  _skip_if_gen1_asset_missing(args.gen)
  try:
    if args.task:
      registered_generations = {
        "WujiHand_Reorient": 1,
        "WujiHand2_Reorient": 2,
        "WujiHand2_Reorient_50Hz": 2,
      }
      if registered_generations.get(args.task) != args.gen:
        raise ValueError(f"Task {args.task!r} does not match --gen {args.gen}")
      package = "wuji_hand2" if args.gen == 2 else "wuji_hand"
      importlib.import_module(f"wuji_mjlab.tasks.reorient.config.{package}")
      cfg = load_env_cfg(args.task, play=True)
      cfg.scene.num_envs = 1
    else:
      cfg = _build_cfg(args.gen, args.hand)
  except Exception as e:
    if args.gen == 1:
      raise RuntimeError(
        "Gen 1 asset exists, but its environment configuration failed to build; "
        "this is a correctness-gate failure, not a skip"
      ) from e
    raise

  cfg.observations["policy"].enable_corruption = False
  cfg.terminations = {}
  # Deploy never corrupts the cube signal, so drop the training-only obs injection.
  for _term in cfg.observations["policy"].terms.values():
    if _term.params and "injection_prob" in _term.params:
      _term.params["injection_prob"] = 0.0
  const = load_constants(args.gen, args.hand)
  history_lengths = {
    name: term.history_length for name, term in cfg.observations["policy"].terms.items()
  }
  if any(length != const.history_length for length in history_lengths.values()):
    raise ValueError(
      f"Task {args.task!r} observation histories {history_lengths} do not match "
      f"deploy history_length={const.history_length}"
    )

  try:
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
  except Exception as e:
    if args.gen == 1:
      raise RuntimeError(
        "Gen 1 asset exists, but its environment failed to instantiate; "
        "this is a correctness-gate failure, not a skip"
      ) from e
    raise
  robot = env.scene["robot"]
  action_term = env.action_manager.get_term("joint_pos")
  target_ids = action_term._target_ids
  cmd = env.command_manager.get_term("reorient_command")
  print(
    f"Task: {args.task or ('WujiHand2_Reorient' if args.gen == 2 else 'WujiHand_Reorient')} "
    f"(play=True), ctrl_dt={env.step_dt:.2f}, ema_alpha={action_term._ema_alpha}",
    flush=True,
  )

  post = ActionPostprocessor(
    const,
    action_scale=action_term._action_scale,
    ema_alpha=action_term._ema_alpha,
    warmup_time_s=action_term._warmup_time_s,
    ctrl_dt=env.step_dt,
  )
  asm = ObsAssembler(const)

  def sim_intermediates():
    jp = robot.data.joint_pos[0, target_ids].cpu().numpy().astype(np.float64)
    tgt = action_term.processed_action[0, target_ids].cpu().numpy().astype(np.float64)
    prev_raw = env.action_manager.prev_action[0].cpu().numpy().astype(np.float64)
    tag_pos_w, tag_quat_w = _tag_frame(env, robot)
    cube_pos_w = env.scene["object"].data.root_link_pos_w
    cube_quat_w = env.scene["object"].data.root_link_quat_w
    cube_pos_tag = (
      quat_apply_inverse(tag_quat_w, cube_pos_w - tag_pos_w)[0].cpu().numpy()
    )
    tag_inv = quat_inv(tag_quat_w)
    cube_quat_tag = quat_mul(tag_inv, cube_quat_w)[0].cpu().numpy()
    goal_quat_tag = quat_mul(tag_inv, cmd.goal_quat)[0].cpu().numpy()
    return jp, tgt, cube_pos_tag, cube_quat_tag, goal_quat_tag, prev_raw

  obs_dict, _ = env.reset()
  max_obs_err = 0.0
  max_act_err = 0.0

  for _k in range(args.steps):
    sim_obs = obs_dict["policy"][0].cpu().numpy().astype(np.float64)
    jp, tgt, cpos, cquat, gquat, prev_raw = sim_intermediates()
    my_obs = asm.build(jp, tgt, cpos, cquat, gquat, prev_raw).astype(np.float64)
    max_obs_err = max(max_obs_err, float(np.max(np.abs(my_obs - sim_obs))))

    action = torch.empty(1, const.action_dim).uniform_(-1.0, 1.0)
    my_target = post.process(action[0].cpu().numpy().astype(np.float64))
    obs_dict, _, _, _, _ = env.step(action)
    sim_target = (
      action_term.processed_action[0, target_ids].cpu().numpy().astype(np.float64)
    )
    max_act_err = max(max_act_err, float(np.max(np.abs(my_target - sim_target))))

  env.close()
  print(f"[gen{args.gen} {args.hand}] steps={args.steps}")
  print(f"  obs    max|Δ| = {max_obs_err:.3e}   (gate < 1e-4)")
  print(f"  action max|Δ| = {max_act_err:.3e}   (gate < 1e-5)")
  ok = max_obs_err < 1e-4 and max_act_err < 1e-5
  print("  RESULT:", "PASS ✓" if ok else "FAIL ✗")
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()
