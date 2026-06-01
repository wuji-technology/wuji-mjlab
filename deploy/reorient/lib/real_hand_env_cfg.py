# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""make_real_hand_env_cfg: take the training reorient cfg as base and trim managers not needed for deploy.

Key points:
  - Critic obs group **must** be trimmed (contains InHandReorient state machine
    fields such as hold_counter / reward_window_timer that ExternalGoalCommandTerm
    does not provide).
  - events / rewards / terminations / curriculum / metrics / recorders are all
    cleared (real hand does not reset sim state, does not compute reward /
    termination conditions).
  - actions keep the original training class (process_action runs EMA + clamp +
    warmup; RealHandEnv reads the result from processed_action itself instead of
    calling apply_action).
  - commands is replaced with ExternalGoalCommandTerm (keep the key
    "reorient_command" unchanged; obs reads goal_quat via this key).
"""
from __future__ import annotations

from wuji_mjlab.tasks.reorient.reorient_env_cfg import make_reorient_env_cfg
from wuji_mjlab.assets.robots.wuji_hand.wuji_hand_cfg import get_wuji_hand_cfg
from wuji_mjlab.assets.objects.inhand_object.object_cfg import get_inhand_object_cfg
from wuji_mjlab.tasks.reorient.reorient_constants import (
    REORIENT_CUBE_INIT_STATE,
    REORIENT_ROBOT_INIT_STATE,
)

from .external_goal_command import ExternalGoalCommandTermCfg
from .real_hand_obs import (
    cube_pos_in_tag_from_zmq,
    goal_rot_err_6d_from_zmq,
)


def _apply_policy_config_overrides(cfg, pc: dict) -> None:
    """Override action term + decimation + obs history to match sidecar JSON.

    Model-intrinsic params (locked to the trained checkpoint) are read from
    the ONNX sidecar JSON via ``ONNXPolicy.config`` and applied here. The
    deploy env is therefore built FROM the policy, so ``validate_against_env``
    becomes a (cheap) belt-and-suspenders sanity check rather than a brittle
    cross-check between two independent sources.

    Keys consumed (all optional — missing ones are skipped):
      action_scale, ema_alpha, warmup_time_s → action_term.cfg
      ctrl_dt                                → cfg.decimation
                                                = round(ctrl_dt / cfg.sim.mujoco.timestep)
      history_len                            → policy obs group's history_length
                                                on every term with history_length>0
      control_mode                           → assert == 'absolute'
                                                (only mode supported in deploy)
    """
    action_term = cfg.actions["joint_pos"]
    for key, attr in (
        ("action_scale", "action_scale"),
        ("ema_alpha", "ema_alpha"),
        ("warmup_time_s", "warmup_time_s"),
    ):
        if key in pc:
            setattr(action_term, attr, float(pc[key]))

    if "ctrl_dt" in pc:
        sim_dt = cfg.sim.mujoco.timestep
        new_decim = max(1, int(round(float(pc["ctrl_dt"]) / sim_dt)))
        cfg.decimation = new_decim

    if "history_len" in pc:
        target_h = int(pc["history_len"])
        for _group_name, group_cfg in cfg.observations.items():
            terms = getattr(group_cfg, "terms", None) or vars(group_cfg)
            items = terms.items() if hasattr(terms, "items") else []
            for _tname, tcfg in items:
                if hasattr(tcfg, "history_length") and getattr(tcfg, "history_length", 0) > 0:
                    tcfg.history_length = target_h

    if "control_mode" in pc:
        assert pc["control_mode"] == "absolute", (
            f"Only control_mode='absolute' supported; sidecar JSON says "
            f"{pc['control_mode']!r}"
        )


def _apply_deploy_obs_overrides(cfg) -> None:
    """Swap training cube_pos_in_tag / goal_rot_err_6d to ZMQ-direct deploy variants.

    The vision pipeline natively detects cube poses in the wrist-AprilTag frame
    — exactly the frame the policy expects. Bypassing the mjworld dance avoids
    the inverse∘forward round-trip whose constants (TAG_IN_PALM) don't match
    the physical AprilTag location precisely. See lib/real_hand_obs.py.
    """
    from wuji_mjlab.tasks.reorient.mdp import observations as _train_obs

    func_swaps = {
        _train_obs.cube_pos_in_tag: cube_pos_in_tag_from_zmq,
        _train_obs.goal_rot_err_6d: goal_rot_err_6d_from_zmq,
    }

    for _group_name, group_cfg in cfg.observations.items():
        terms = getattr(group_cfg, "terms", None)
        if terms is None:
            continue
        for _term_name, term_cfg in terms.items():
            if term_cfg.func in func_swaps:
                term_cfg.func = func_swaps[term_cfg.func]


def attach_wuji_hand_entities(cfg) -> None:
    """Copy the entity attach block from the training wuji_hand_reorient_env_cfg (Wuji Hand).

    Not copied: reward overrides (finger_collision/palm_detach), play branch.
    When the training-side attach logic changes, this function must be kept in sync.
    """
    cfg.scene.entities = {
        "robot": get_wuji_hand_cfg(),
        "object": get_inhand_object_cfg(),
    }
    cfg.scene.entities["robot"].init_state = REORIENT_ROBOT_INIT_STATE
    cfg.scene.entities["object"].init_state = REORIENT_CUBE_INIT_STATE
    cfg.viewer.body_name = "right_palm_link"


def make_real_hand_env_cfg(
    robot_variant: str = "wuji_hand",
    policy_config: dict | None = None,
):
    """Construct deploy env cfg by trimming training cfg.

    Args:
        robot_variant: only "wuji_hand" (Wuji Hand) is supported for now; other
            models can be added in the future.
        policy_config: optional dict from ONNX sidecar JSON
            (``ONNXPolicy.config``). When provided, model-intrinsic params
            (action_scale, ema_alpha, warmup_time_s, ctrl_dt, history_len,
            control_mode) override the cfg so the env matches the checkpoint
            exactly. See ``_apply_policy_config_overrides``. Missing keys are
            skipped — the cfg keeps the training defaults for those fields.
    """
    cfg = make_reorient_env_cfg()

    if robot_variant == "wuji_hand":
        attach_wuji_hand_entities(cfg)
    else:
        raise NotImplementedError(f"robot_variant={robot_variant!r}")

    cfg.scene.num_envs = 1

    # Empty non-deploy managers (mjlab accepts empty dict and falls back to Null* manager branch)
    cfg.events = {}
    cfg.rewards = {}
    cfg.terminations = {}
    cfg.curriculum = {}
    cfg.metrics = {}
    cfg.recorders = {}

    # Trim critic obs group (uses InHandReorient state machine attrs)
    cfg.observations = {"policy": cfg.observations["policy"]}

    # Apply policy-driven (model-intrinsic) overrides BEFORE the deploy obs
    # swap so history_length lands on the cfg the swap will mutate further.
    if policy_config is not None:
        _apply_policy_config_overrides(cfg, policy_config)

    # Swap obs term funcs to deploy-side ZMQ-direct variants (must run AFTER
    # critic trim so we don't waste work on terms that won't be used).
    _apply_deploy_obs_overrides(cfg)

    # Replace command with ZMQ-driven one
    cfg.commands = {
        "reorient_command": ExternalGoalCommandTermCfg(
            resampling_time_range=(1e9, 1e9),
            debug_vis=False,
        ),
    }

    return cfg
