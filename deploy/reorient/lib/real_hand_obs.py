# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Deploy-side ObsTerm overrides — read tag-frame cube pose directly from ZMQ.

The training-side obs functions (src/wuji_mjlab/tasks/reorient/mdp/observations.py)
compute cube_pos_in_tag by transforming scene["object"].root_link_pos_w (mjworld)
through palm + TAG_IN_PALM. For deploy, the observer's vision pipeline natively
outputs cube poses in the wrist-tag frame, which IS exactly the frame the policy
was trained on. So we override the obs term funcs to bypass the mjworld dance.

These functions are configclass-level swaps in real_hand_env_cfg.make_real_hand_env_cfg.

For ZMQ-less smoke tests, fall back to zeros/identity (matches RealHandEnv default
when _cube_zmq is None — see _cube_zmq handling in real_hand_env.py).
"""
from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_inv, quat_mul


def cube_pos_in_tag_from_zmq(
    env,
    object_cfg: SceneEntityCfg | None = None,
    robot_cfg: SceneEntityCfg | None = None,
    injection_prob: float = 0.0,
    tag_in_palm_pos: tuple[float, float, float] | None = None,
    tag_in_palm_quat: tuple[float, float, float, float] | None = None,
) -> torch.Tensor:
    """Tag-frame cube position, read directly from observer ZMQ feed.

    Args mirror training cube_pos_in_tag for ObservationTermCfg drop-in.
    """
    if getattr(env, "_cube_zmq", None) is None:
        return torch.zeros((env.num_envs, 3), device=env.device)
    cube_pos, _cube_quat = env._cube_zmq.latest()
    out = torch.from_numpy(cube_pos).float().unsqueeze(0).to(env.device)
    if injection_prob > 0 and env.scene.device != "meta":
        n = out.shape[0]
        mask = torch.rand(n, 1, device=out.device) < injection_prob
        rand_pos = torch.empty_like(out).uniform_(-0.5, 0.5)
        out = torch.where(mask, rand_pos, out)
    return out


def goal_rot_err_6d_from_zmq(
    env,
    command_name: str,
    object_cfg: SceneEntityCfg | None = None,
    robot_cfg: SceneEntityCfg | None = None,
    injection_prob: float = 0.0,
    tag_in_palm_pos: tuple[float, float, float] | None = None,
    tag_in_palm_quat: tuple[float, float, float, float] | None = None,
) -> torch.Tensor:
    """6D rotation error in tag frame, computed from tag-frame cube_quat (ZMQ)
    and tag-frame goal_quat (command term).

    Mirrors training goal_rot_err_6d exactly, except both quats are already in
    tag frame (no mjworld→tag conversion needed).
    """
    if getattr(env, "_cube_zmq", None) is None:
        cube_quat_tag = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
            .unsqueeze(0)
            .expand(env.num_envs, 4)
        )
    else:
        _cube_pos, cube_quat = env._cube_zmq.latest()
        cube_quat_tag = (
            torch.from_numpy(cube_quat).float().unsqueeze(0).to(env.device)
        )

    command = env.command_manager.get_term(command_name)
    # goal_quat from the command is already in tag frame (deploy convention:
    # external/fixed/random goal modes all express the goal quat as the target
    # cube orientation in tag frame).
    goal_quat_tag = command.goal_quat

    q_err_tag = quat_mul(cube_quat_tag, quat_inv(goal_quat_tag))
    rot = matrix_from_quat(q_err_tag)
    ori_error = rot.reshape(*rot.shape[:-2], 9)[..., 3:]

    if injection_prob > 0 and env.scene.device != "meta":
        from wuji_mjlab.utils.math import random_quat_uniform

        n = ori_error.shape[0]
        mask = torch.rand(n, 1, device=ori_error.device) < injection_prob
        rand_quat = random_quat_uniform(n, device=ori_error.device)
        rand_rot = matrix_from_quat(rand_quat)
        rand_error = rand_rot.reshape(*rand_rot.shape[:-2], 9)[..., 3:]
        ori_error = torch.where(mask, rand_error, ori_error)

    return ori_error
