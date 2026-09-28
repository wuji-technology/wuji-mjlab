# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import csv
import json
import sys

import numpy as np
import onnx
import pytest
from wuji_mjlab.tasks.reorient.tooling import eval_core
from wuji_mjlab.tasks.reorient.tooling.eval_core import ObsBuilder
from wuji_mjlab.tasks.reorient.tooling.scene_builder import build_reorient_scene


def test_hand2_obs_term_layout_and_history_backfill():
  scene = build_reorient_scene(hand="hand2", hand_side="right")
  obs = ObsBuilder(history_length=3).build(
    scene,
    prev_target=scene.default_joint_pos,
    goal_quat=np.array([1.0, 0.0, 0.0, 0.0]),
    last_action=np.zeros(20, dtype=np.float32),
  )
  joint_angles = obs[0:60]
  qpos_error = obs[60:120]
  action_history = obs[147:207]

  np.testing.assert_array_equal(action_history, np.zeros(60, dtype=obs.dtype))
  np.testing.assert_allclose(qpos_error, 0.0, atol=1e-5)
  np.testing.assert_array_equal(joint_angles[0:20], joint_angles[20:40])
  np.testing.assert_array_equal(joint_angles[20:40], joint_angles[40:60])


def test_unknown_hand_raises():
  with pytest.raises(ValueError):
    build_reorient_scene(hand="hand9")


@pytest.fixture
def eval_policy(tmp_path):
  model_dir = tmp_path / "policy"
  model_dir.mkdir()
  policy = model_dir / "policy.onnx"
  graph = onnx.helper.make_graph(
    [onnx.helper.make_node("MatMul", ["obs", "weights"], ["actions"])],
    "test_eval_policy",
    [onnx.helper.make_tensor_value_info("obs", onnx.TensorProto.FLOAT, [1, 69])],
    [onnx.helper.make_tensor_value_info("actions", onnx.TensorProto.FLOAT, [1, 20])],
    [onnx.numpy_helper.from_array(np.zeros((69, 20), dtype=np.float32), "weights")],
  )
  model = onnx.helper.make_model(
    graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=8
  )
  onnx.save_model(model, policy)
  (model_dir / "config.json").write_text(json.dumps({"sim_dt": 0.01, "ctrl_dt": 0.02}))
  return policy


@pytest.mark.parametrize(
  ("output_kind", "filename"),
  [
    ("explicit", "result.json"),
    ("nested", "result.json"),
    ("default", None),
    ("explicit", "eval_results.json"),
    ("explicit", "cube_motion_log.csv"),
  ],
)
def test_eval_artifacts_follow_output_location(
  eval_policy, tmp_path, monkeypatch, output_kind, filename
):
  argv = [
    "eval_success_rate",
    str(eval_policy),
    "--hand",
    "hand2",
    "--num-trials",
    "1",
    "--trial-timeout",
    "0.1",
    "--no-viewer",
  ]
  original_files = {}
  if output_kind == "default":
    output_dir = eval_policy.parent
  else:
    output_dir = tmp_path / "results"
    if output_kind == "explicit":
      output_dir.mkdir()
    else:
      output_dir = output_dir / "nested"
    argv += ["--json-output", str(output_dir / filename)]
    for name in ("cube_motion_log.csv", "eval_results.json"):
      path = eval_policy.parent / name
      path.write_text("previous evaluation")
      original_files[path] = path.read_bytes()
  monkeypatch.setattr(sys, "argv", argv)
  random_state = np.random.get_state()
  try:
    np.random.seed(0)
    if filename in {"eval_results.json", "cube_motion_log.csv"}:
      with pytest.raises(SystemExit) as exc_info:
        eval_core.main()
      assert exc_info.value.code == 2
      assert not tuple(output_dir.iterdir())
      return
    eval_core.main()
  finally:
    np.random.set_state(random_state)
  for path, content in original_files.items():
    assert path.read_bytes() == content
  legacy = json.loads((output_dir / "eval_results.json").read_text())
  assert legacy["completed"] == 1
  assert legacy["successes"] + legacy["drops"] + legacy["timeouts"] == 1
  with (output_dir / "cube_motion_log.csv").open() as stream:
    rows = list(csv.DictReader(stream))
  assert float(rows[0]["time"]) == pytest.approx(0.02)
  assert {int(row["trial"]) for row in rows} == {1}
  if output_kind != "default":
    structured = json.loads((output_dir / "result.json").read_text())
    assert len(structured["trials"]) == 1
    assert structured["trials"][0]["status"] in {"success", "drop", "timeout"}
