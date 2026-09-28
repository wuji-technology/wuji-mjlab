# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Torch-free ONNX policy runner for the fixed-wrist pen-spin contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from lib.config_loader import CONFIG_DIR

TASK_ID = "WujiHand2_PenSpin"
OBS_DIM = 155
ACTION_DIM = 20
MLP_HIDDEN_DIMS: tuple[int, ...] = (1024, 512, 256)

# Values must match training-side SingleHandResidualEMAActionCfg in
# src/wuji_mjlab/tasks/pen_spin/mdp/actions.py (scale, alpha, residual_clip)
# and pen_spin_env_cfg.py (timestep 0.005 x decimation 4 = 0.02).
PEN_SPIN_CONFIG = Path(CONFIG_DIR) / "pen_spin_policy.json"


class ONNXPolicy:
    """Load one ONNX policy and check it against config/pen_spin_policy.json."""

    def __init__(self, path: str | Path):
        # Limit worker threads to avoid contention with the camera observer.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        import onnxruntime as ort

        self.path = Path(path).expanduser().resolve()
        if self.path.suffix != ".onnx":
            raise ValueError(f"pen-spin deploy expects an .onnx policy, got {self.path}")
        if not self.path.is_file():
            raise FileNotFoundError(f"policy not found: {self.path}")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("pen-spin policy must have exactly one input and output")
        self._input = inputs[0].name
        self._output = outputs[0].name
        input_dim = inputs[0].shape[-1]
        output_dim = outputs[0].shape[-1]
        if input_dim != OBS_DIM or output_dim != ACTION_DIM:
            raise ValueError(
                f"policy IO ({input_dim}->{output_dim}) != expected "
                f"({OBS_DIM}->{ACTION_DIM})"
            )

        self.config = self._load_config()
        self._validate_config()
        self.obs_dim = OBS_DIM

        # The first forward pays allocation and kernel selection.
        self.infer(np.zeros(OBS_DIM, dtype=np.float32))

        print(
            f"[pen_spin_policy] loaded {self.path.name}: "
            f"obs_dim={OBS_DIM} action_dim={ACTION_DIM} hidden={MLP_HIDDEN_DIMS}"
        )
        print(f"  config: {PEN_SPIN_CONFIG}")
        print(
            f"  ctrl_dt={self.ctrl_dt}  scale={self.action_scale}  "
            f"clip={self.action_residual_clip}  ema_alpha={self.action_ema_alpha}"
        )

    def _load_config(self) -> dict[str, Any]:
        if not PEN_SPIN_CONFIG.is_file():
            raise FileNotFoundError(f"pen-spin config not found: {PEN_SPIN_CONFIG}")
        with PEN_SPIN_CONFIG.open(encoding="utf-8") as stream:
            config = json.load(stream)
        if not isinstance(config, dict):
            raise ValueError(f"policy config must be a JSON object: {PEN_SPIN_CONFIG}")
        return config

    def _validate_config(self) -> None:
        expected = {
            "task": TASK_ID,
            "fixed_wrist": True,
            "obs_dim": OBS_DIM,
            "actor_hidden_dims": list(MLP_HIDDEN_DIMS),
            "action_dim": ACTION_DIM,
        }
        mismatch = {
            key: (self.config.get(key), value)
            for key, value in expected.items()
            if self.config.get(key) != value
        }
        if mismatch:
            raise ValueError(f"invalid pen-spin policy config: {mismatch}")
        if float(self.config.get("ctrl_dt", 0.0)) <= 0:
            raise ValueError("policy config ctrl_dt must be positive")
        if float(self.config["action_scale"]) <= 0:
            raise ValueError("policy config action_scale must be positive")
        if not 0.0 <= float(self.config["action_ema_alpha"]) <= 1.0:
            raise ValueError("policy config action_ema_alpha must be in [0, 1]")

    @property
    def ctrl_dt(self) -> float:
        return float(self.config["ctrl_dt"])

    @property
    def action_scale(self) -> float:
        return float(self.config["action_scale"])

    @property
    def action_residual_clip(self) -> tuple[float, float]:
        clip = self.config["action_residual_clip"]
        if not isinstance(clip, (list, tuple)) or len(clip) != 2:
            raise ValueError("policy config action_residual_clip must have two values")
        return float(clip[0]), float(clip[1])

    @property
    def action_ema_alpha(self) -> float:
        return float(self.config["action_ema_alpha"])

    def infer(self, obs: np.ndarray) -> np.ndarray:
        if obs.shape != (OBS_DIM,):
            raise ValueError(f"obs shape {obs.shape} disagrees with obs_dim={OBS_DIM}")
        vector = np.asarray(obs, dtype=np.float32).reshape(1, OBS_DIM)
        action = self.session.run([self._output], {self._input: vector})[0]
        return np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
