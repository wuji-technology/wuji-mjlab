# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""RealHandEnv: real-hand env that shares the manager pipeline with the sim env.

Inherits mjlab.envs.ManagerBasedRlEnv and overrides step()/reset() to replace
sim.step() physics integration with:
  1. wujihandpy.write_target  -- hardware IO
  2. wujihandpy.read_encoders + ZMQ cube/goal -- sensor reads
  3. scene["robot/object"].write_*_to_sim -- write into sim.data buffer
  4. sim.forward() -- FK only, no integration
  5. observation_manager.compute() -- identical to sim, byte for byte

obs/action pipeline **runs as-is** with zero code difference from the training
side. sim2real drift can only come from sensor precision / calibration error,
not from obs computation divergence.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

from .hand_driver import HandDriverBase, MockHandDriver

log = logging.getLogger(__name__)


class RealHandEnv(ManagerBasedRlEnv):
    """ManagerBasedRlEnv subclass for real-hand sim2real deployment."""

    MAX_CONSECUTIVE_IO_FAILURES = 3
    """Trigger safe_stop after N consecutive ZMQ/IO failures."""

    def __init__(
        self,
        cfg: ManagerBasedRlEnvCfg,
        hand_driver: Optional[HandDriverBase] = None,
    ):
        """
        Args:
            cfg: deploy-side ManagerBasedRlEnvCfg (from make_real_hand_env_cfg).
            hand_driver: HandDriverBase subclass. Defaults to MockHandDriver
                (for testing without hardware); inject a robot-specific driver
                for real-hand deployment.

        Note:
            The legacy ``hand_base_pose_w`` arg was removed: cube and goal
            observations now reference the wrist-AprilTag frame, so no
            consumer reads the hand-base pose at runtime. Robot root pose
            is baked into the mjcf at compile time from
            REORIENT_ROBOT_INIT_STATE.
        """
        # Prerequisite for the mjlab CPU path
        import warp as wp
        wp.set_device("cpu")

        super().__init__(cfg=cfg, device="cpu", render_mode=None)

        self._hand: HandDriverBase = hand_driver or MockHandDriver()

        # Control period (sim.mujoco.timestep * decimation) - 20Hz @ default
        self._ctrl_dt: float = cfg.sim.mujoco.timestep * cfg.decimation
        self._next_tick: float = time.monotonic()
        self._io_failure_streak: int = 0

        # Fast-FK buffer: mujoco CPU mj_data side-buffer for FK that bypasses
        # mujoco_warp CPU overhead (~280ms → ~0.03ms, 10000× speedup).
        # Used by _fast_forward() instead of self.sim.forward().
        self._mj_model = self.sim.mj_model
        self._mj_data = mujoco.MjData(self._mj_model)

        self._cube_zmq = None
        self._goal_zmq = None

        # Joint order sanity check
        self._validate_joint_order()

    # ────────────────── core step/reset ──────────────────

    def step(self, action: torch.Tensor):
        """Override sim step: hardware IO + sensors + sim.forward() + obs.

        7-step flow:
          0) NaN/Inf guard on action
          1) action_manager.process_action (EMA/clamp/warmup), read processed
          2) hand.write_target + sleep to next tick
          3) hand.read_encoders + ZMQ cube/goal
          4) scene.write_*_to_sim (sensor data → mj_data)
          5) external goal term set_goal
          6) sim.forward() (FK only)
          7) observation_manager.compute(update_history=True)
        """
        # ── 0) Safety: refuse NaN/Inf action ──
        if not torch.isfinite(action).all():
            self._safe_stop("Non-finite action from policy")
            raise RuntimeError("Policy emitted NaN/Inf action; refused.")

        # ── 1) Action pipeline (compute only, no ctrl write) ──
        self.action_manager.process_action(action.to(self.device))
        processed = self.action_manager.get_term("joint_pos").processed_action

        # ── 2-3) Hardware IO + sensors (with watchdog) ──
        processed_np = processed[0].cpu().numpy()
        processed_np = self._clamp_to_joint_limits(processed_np)

        sensor_data_ok = False
        joint_angles: Optional[np.ndarray] = None
        goal_quat: Optional[np.ndarray] = None
        try:
            self._hand.write_target(processed_np)
            self._next_tick += self._ctrl_dt
            sleep_remaining = self._next_tick - time.monotonic()
            if sleep_remaining > 0:
                time.sleep(sleep_remaining)

            joint_angles = self._hand.read_encoders()
            if self._goal_zmq is not None:
                goal_quat = self._goal_zmq.latest()
            else:
                goal_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity wxyz
            sensor_data_ok = True
        except (TimeoutError, IOError, ConnectionError) as e:
            self._io_failure_streak += 1
            log.warning(
                "IO failure %d/%d: %s",
                self._io_failure_streak,
                self.MAX_CONSECUTIVE_IO_FAILURES,
                e,
            )
            if self._io_failure_streak >= self.MAX_CONSECUTIVE_IO_FAILURES:
                self._safe_stop(f"IO failure streak: {e}")
                raise
            self._next_tick = time.monotonic()  # reset tick

        if not sensor_data_ok:
            # Stale-frame fallback: FK on previous sim.data + return cached obs.
            # Critical: do NOT call compute(update_history=True) to avoid
            # contaminating the H>1 buffer with fake frames.
            self.sim.forward()
            zero = torch.zeros(1, device=self.device)
            self._fast_forward()  # FK refresh even on stale fallback (fast CPU path)
            return self.obs_buf, zero, zero.bool(), zero.bool(), {"stale_frame": True}

        # ── 4) Write sensor data into sim.data via Entity API ──
        # Robot is fixed-base in mjcf — its root pose is baked at compile time
        # from REORIENT_ROBOT_INIT_STATE.
        self._io_failure_streak = 0
        env_ids = torch.tensor([0], device=self.device)

        joint_angles = self._validate_joint_vector(joint_angles, "joint_angles")

        # Hand joint angles in sim/policy order.
        self.scene["robot"].write_joint_position_to_sim(
            torch.from_numpy(joint_angles).float().unsqueeze(0).to(self.device),
            env_ids=env_ids,
        )
        # NOTE: cube pose is NOT written to scene["object"]. The deploy obs
        # term overrides (lib/real_hand_obs.py) read tag-frame cube pose
        # directly from self._cube_zmq, bypassing scene["object"]. The
        # mjworld round-trip was removed because TAG_IN_PALM constants don't
        # match the physical AprilTag location precisely.

        # ── 5) Inject goal into command term ──
        goal_term = self.command_manager.get_term("reorient_command")
        goal_term.set_goal(
            torch.from_numpy(goal_quat).float().unsqueeze(0).to(self.device)
        )

        # Tick episode counter so action_term exits warmup.
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # ── 6) FK only (refresh derived quantities like body_link_pose_w) ──
        # _fast_forward = mujoco.mj_forward + sync to wp_data (~0.03ms vs
        # sim.forward's ~280ms on CPU).
        self._fast_forward()

        # ── 7) Compute obs via training-side pipeline ──
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # ── 8) Return (no reward/termination semantics on real hand) ──
        zero = torch.zeros(1, device=self.device)
        return self.obs_buf, zero, zero.bool(), zero.bool(), {}

    def reset(self, env_ids=None):
        """Home pose + clear history. Do NOT call sim.reset() (it would zero wp_data)."""
        env_ids = (
            env_ids if env_ids is not None else torch.tensor([0], device=self.device)
        )

        # 1) Home hand & read back qpos (assumes home() blocks until complete)
        self._hand.home()
        home_qpos = self._validate_joint_vector(
            self._hand.read_encoders(),
            "home_qpos",
        )

        # 2) Write home qpos to sim.data
        # (robot root pose is fixed-base, set at compile time from init_state)
        self.scene["robot"].write_joint_position_to_sim(
            torch.from_numpy(home_qpos).float().unsqueeze(0).to(self.device),
            env_ids=env_ids,
        )
        # NOTE: cube pose is NOT written to scene["object"]. The deploy obs
        # term overrides (lib/real_hand_obs.py) read cube pose from
        # self._cube_zmq directly, bypassing scene["object"] entirely.

        # 3) Clear manager history + episode counter (guard against cross-trial state contamination)
        # episode_length_buf is used by JointPositionOffsetEMAAction.in_warmup;
        # not resetting causes the action term to skip warmup on subsequent
        # trials, producing different processed_action for same raw action.
        self.episode_length_buf[env_ids] = 0
        self.action_manager.reset(env_ids)
        self.observation_manager.reset(env_ids)
        self.command_manager.reset(env_ids)

        # Prime _processed_actions to default by running process_action(zeros)
        # at the post-reset moment. Otherwise stale _processed_actions from a
        # prior trial leaks into obs term joint_pos_target_error via
        # action_manager.get_term().processed_action read.
        self.action_manager.process_action(
            torch.zeros(
                self.num_envs,
                self.action_manager.total_action_dim,
                device=self.device,
            )
        )

        # 4) FK refresh + compute initial obs (fast CPU mj path, ~0.03ms)
        self._fast_forward()
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # 5) Reset tick + IO streak
        self._next_tick = time.monotonic()
        self._io_failure_streak = 0
        return self.obs_buf, {}

    # ────────────────── helpers ──────────────────

    def _validate_joint_order(self) -> None:
        """Assert driver encoder order == sim joint order."""
        sim_names = tuple(self.scene["robot"].joint_names)
        encoder_names = tuple(self._hand.joint_names_in_encoder_order())
        assert sim_names == encoder_names, (
            f"Joint order mismatch between MJCF and hand driver encoder order:\n"
            f"  sim: {sim_names}\n"
            f"  enc: {encoder_names}\n"
            "Likely causes: stale hand SDK/profile joint order or modified MJCF "
            "joint order. Fix the robot profile/driver mapping before deploying."
        )

    def _clamp_to_joint_limits(self, qpos: np.ndarray) -> np.ndarray:
        """Soft-joint-pos-limits clamp (matches the training JointPositionOffsetEMAAction)."""
        qpos = self._validate_joint_vector(qpos, "qpos")
        soft = self.scene["robot"].data.soft_joint_pos_limits[0].cpu().numpy()
        return np.clip(qpos, soft[:, 0], soft[:, 1])

    def _validate_joint_vector(self, value: np.ndarray, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        expected = len(self.scene["robot"].joint_names)
        if vector.shape != (expected,):
            raise ValueError(f"{name} shape {vector.shape}, expected ({expected},).")
        if not np.isfinite(vector).all():
            raise ValueError(f"{name} contains NaN/Inf.")
        return vector

    def _fast_forward(self) -> None:
        """Replace self.sim.forward() (~280ms on CPU) with mj_forward (~0.03ms).

        Strategy: read wp_data.qpos[0] → copy into mj_data.qpos → mj_forward
        → copy mj_data.xpos / xquat / site_xpos back into wp_data[0].
        obs term reads (body_link_pose_w / root_link_pose_w / site_xpos) all
        come from wp_data, so the sync makes obs_manager output identical to
        full sim.forward (verified byte-equal).

        Why mujoco_warp CPU is slow: it dispatches many kernels per forward
        call; on CPU each kernel-launch overhead dominates. Standard mujoco
        CPU does the same math without kernel-launch overhead. mujoco_warp
        is GPU-first by design.
        """
        # qpos: wp_data → mj_data (env 0 only — deploy is num_envs=1)
        qpos_view = self.sim.wp_data.qpos.numpy()
        self._mj_data.qpos[:] = qpos_view[0]
        # FK + dependent derived quantities (xpos, xquat, site_xpos, geom_xpos)
        mujoco.mj_forward(self._mj_model, self._mj_data)
        # Sync derived quantities back to wp_data so Entity property reads
        # (data.xpos / data.xquat / data.site_xpos / data.geom_xpos) see the
        # newly-computed values.
        self.sim.wp_data.xpos.numpy()[0] = self._mj_data.xpos
        self.sim.wp_data.xquat.numpy()[0] = self._mj_data.xquat
        self.sim.wp_data.site_xpos.numpy()[0] = self._mj_data.site_xpos
        if hasattr(self.sim.wp_data, "geom_xpos"):
            self.sim.wp_data.geom_xpos.numpy()[0] = self._mj_data.geom_xpos

    def _safe_stop(self, reason: str) -> None:
        log.error("SAFE STOP: %s", reason)
        try:
            self._hand.home()
        except Exception:
            log.exception("safe_stop home() failed")
