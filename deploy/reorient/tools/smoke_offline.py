# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Hardware/ZMQ/mjlab-free smoke: obs assembly + onnx head + mock driver."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from wuji_reorient_deploy.constants import load_constants
from wuji_reorient_deploy.drivers import MockHandDriver
from wuji_reorient_deploy.obs_builder import ActionPostprocessor, ObsAssembler
from wuji_reorient_deploy.onnx_policy import ONNXPolicy

_POLICY = Path(__file__).resolve().parents[1] / "policies" / "policy_hand2_right.onnx"


def main() -> None:
    # Pinned to gen2 on purpose: the offline smoke has no --gen switch.
    const = load_constants(2, "right")
    policy = ONNXPolicy(str(_POLICY), const)
    asm = ObsAssembler(const)
    post = ActionPostprocessor(
        const,
        action_scale=policy.config.get("action_scale", 0.5),
        ema_alpha=policy.config.get("ema_alpha", 0.5),
        warmup_time_s=policy.config.get("warmup_time_s", 0.4),
        ctrl_dt=policy.config.get("ctrl_dt", 0.05),
    )
    drv = MockHandDriver(const, "right")
    q = drv.read_encoders()
    obs = asm.build(
        joint_pos=q,
        target=const.default_joint_pos,
        cube_pos_tag=np.zeros(3),
        cube_quat_tag=np.array([1.0, 0.0, 0.0, 0.0]),
        goal_quat_tag=np.array([1.0, 0.0, 0.0, 0.0]),
        prev_raw_action=np.zeros(const.action_dim),
    )
    assert obs.shape == (const.obs_dim,), obs.shape
    action = policy(obs)
    target = post.process(action)
    drv.write_target(target)
    assert target.shape == (const.action_dim,), target.shape
    print(f"OK obs={obs.shape} action={action.shape} target[:3]={target[:3]}")


if __name__ == "__main__":
    main()
