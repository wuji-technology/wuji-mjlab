# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji-specific extension of mjlab's on-policy runner."""

from __future__ import annotations

from mjlab.rl import MjlabOnPolicyRunner


class WujiOnPolicyRunner(MjlabOnPolicyRunner):
  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    env.unwrapped.max_common_steps = (
      train_cfg["max_iterations"] * train_cfg["num_steps_per_env"]
    )
    # mjlab serializes its logger setting as a string, while rsl-rl's plugin
    # resolver requires a class_name mapping. Rewrite it here rather than
    # duplicating mjlab's runner configuration dataclass.
    if log_dir is not None and train_cfg.get("logger") == "wandb":
      from wuji_mjlab.rl.log_writer import require_wandb_project

      train_cfg["wandb_project"] = require_wandb_project()
      train_cfg["logger"] = {
        "class_name": "wuji_mjlab.rl.log_writer:WujiWandbLogWriter",
        "project_name": train_cfg["wandb_project"],
      }
    super().__init__(env, train_cfg, log_dir, device)

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    self._final_iteration = (
      self.current_learning_iteration + num_learning_iterations - 1
    )
    super().learn(num_learning_iterations, init_at_random_ep_len)

  def save(self, path: str, infos=None) -> None:
    original = self.cfg.get("upload_model")
    self.cfg["upload_model"] = self.should_upload_artifact()
    try:
      super().save(path, infos)
    finally:
      self.cfg["upload_model"] = original

  def should_upload_artifact(self) -> bool:
    """Whether the current iteration is eligible for a W&B artifact upload.

    ``>=`` prevents an upstream counter increment before the final save from
    silently suppressing the terminal artifact. Subclasses that emit their own
    artifacts should gate ``self.logger.save_model`` with this predicate.
    """
    if not self.cfg.get("upload_model"):
      return False
    final = getattr(self, "_final_iteration", None)
    if final is None:
      return False
    return self.current_learning_iteration >= final
