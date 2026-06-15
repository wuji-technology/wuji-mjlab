# Revo3 Hand Policy Replay Deployment Specification

## Summary

Read-only inspection found an existing active Revo3 RL deployment path, but not the safe offline policy replay stage requested here. No files were modified and no hardware commands were run.

Key sources: [revo3_policy_node.py](/home/liuxinyu/workspace/revoarm_teleoperation/revoarm_hardware/Revoarm_ws/src/brainco_capabilities/revo3_rl_deploy/revo3_rl_deploy/revo3_policy_node.py), [stage2_input_builder.py](/home/liuxinyu/workspace/revoarm_teleoperation/revoarm_hardware/Revoarm_ws/src/brainco_capabilities/revo3_rl_deploy/revo3_rl_deploy/stage2_input_builder.py), [replay.py](/home/liuxinyu/workspace/revoarm_teleoperation/revoarm_hardware/Revoarm_ws/src/brainco_capabilities/revo3_rl_deploy/revo3_rl_deploy/replay.py), [policy.yaml](/home/liuxinyu/workspace/revoarm_teleoperation/revoarm_hardware/Revoarm_ws/src/brainco_capabilities/revo3_rl_deploy/config/onnx/policy.yaml), [revo3_profile.yaml](/home/liuxinyu/workspace/revoarm_teleoperation/revoarm_hardware/Revoarm_ws/src/brainco_capabilities/revo3_rl_deploy/config/robot_profile/revo3_profile.yaml), [revo3_controllers.yaml](/home/liuxinyu/workspace/revoarm_teleoperation/revoarm_hardware/Revoarm_ws/src/brainco_drivers/revo3_driver/config/revo3_controllers.yaml), [Revo3MITCommand.msg](/home/liuxinyu/workspace/revoarm_teleoperation/revoarm_hardware/Revoarm_ws/src/brainco_controllers/revo3_mit_controller_msgs/msg/Revo3MITCommand.msg).

## Existing Deployment Structure

Relevant packages:
- `revo3_rl_deploy`: current Revo3 RL/trajectory deployment package.
- `revo3_driver`: standalone and dual Revo3 ros2_control hardware launch/config.
- `revo3_mit_controller` and `revo3_mit_controller_msgs`: MIT command controller and message.
- `revo3_description`: URDF/xacro joint names, axes, and limits.
- `manus_revo3_retarget`: active Manus-to-Revo3 command publisher, useful for topic/interface context only.
- `replay` and `lerobot_deploy`: existing action/trajectory replay tools, but they publish to controllers and are not safe offline policy replay.

Relevant current nodes/tools:
- `revo3_policy_node`: subscribes Revo3 `JointState`, builds Stage 2 ONNX inputs, runs ONNX Runtime, publishes `Revo3MITCommand`.
- `revo3_trajectory_replay` via `replay.py`: replays `.target.txt` trajectories to the MIT controller.
- `revo3_param_tuner`: publishes `/revo3_param_tuner/{kp,kd,offset}` for active replay tuning.
- `forward_quintic_command.py` and `arm_hand_cylinder_demo.py`: active command tools; demo has `--dry-run` but still constructs publishers.
- `revo3_rl_deploy.launch.py`: launches only the active policy node, assuming `revo3_driver` is already running.

Missing today:
- No `policy_manifest.yaml`, `obs_normalizer.npz`, or `golden_io.npz` was found.
- The IDE tab `MODEL_ACTION_EXTRACTION_README.md` was not present in this checkout.

## Runtime Action Path

Current active path:
1. Revo3 joint states come from `/revo3_<side>/revo3_joint_state/joint_states`, produced by `joint_state_broadcaster`.
2. Hardware state is read by `Revo3HandHardware`; SDK degrees/rpm are converted to ROS radians/rad/s.
3. `Stage2InputBuilder` constructs:
   - `obs`: `float32[1,126]`, last 3 frames × 42.
   - `proprio_hist`: `float32[1,30,42]`.
   - Per-frame schema: `[normalized_joint_pos(21), current_target_rad(21)]`.
4. `revo3_policy_node` calls ONNX Runtime.
5. Post-processing clips raw action to `[-1, 1]`, applies `target_next = clip(target_prev + action_scale * action, joint_limits)`.
6. Output is permuted from policy order to controller order, `sim2real_joint_offset` is added, and `Revo3MITCommand` is published.
7. `Revo3MITController` writes `position`, `velocity`, `effort`, `kp`, `kd` command interfaces.
8. `Revo3HandHardware.write()` converts rad to degrees, rad/s to rpm, keeps effort as mA, and calls SDK `send_mit_command`.

Replay-mode stop point:
- Offline policy replay must stop after step 5 or optional controller-order conversion.
- It must not create a ROS publisher, publish `Revo3MITCommand`, start `revo3_driver`, or call controller/hardware code.

## Revo3 Hand Interface Assumptions

DoF and units:
- 21 DoF.
- ROS/deployment joint position units are radians.
- Velocity is rad/s.
- MIT effort field is mA torque feedforward.
- Hardware SDK internally uses degrees/rpm, but deployment-facing policy replay must stay in radians.

Policy joint order, current right-hand artifact:
```text
right_index_MPR_joint, right_little_MPR_joint, right_middle_MPR_joint, right_ring_MPR_joint, right_thumb_CMP_joint,
right_index_MCP_joint, right_little_MCP_joint, right_middle_MCP_joint, right_ring_MCP_joint, right_thumb_CMR_joint,
right_index_PIP_joint, right_little_PIP_joint, right_middle_PIP_joint, right_ring_PIP_joint, right_thumb_MCP_joint,
right_index_DIP_joint, right_little_DIP_joint, right_middle_DIP_joint, right_ring_DIP_joint, right_thumb_PIP_joint,
right_thumb_DIP_joint
```

Controller/SDK order:
```text
little MPR/MCP/PIP/DIP, ring MPR/MCP/PIP/DIP, middle MPR/MCP/PIP/DIP,
index MPR/MCP/PIP/DIP, thumb MCP/PIP/DIP/CMP/CMR
```

Limits from profile/URDF:
- Non-thumb `MPR`: `[-0.2618, 0.2618]`.
- Thumb `CMP`: `[0.0, 1.9199]`.
- Thumb `CMR`: `[0.0, 2.0071]`.
- Thumb `MCP`: `[0.0, 0.8727]`.
- All other flexion joints: `[0.0, 1.4835]`.
- Positive direction is available through URDF axes, but no human-readable convention is documented; replay should record the manifest’s convention and not infer mirroring.

Command mode:
- Active deployment command is absolute MIT position target plus optional velocity/kp/kd/effort.
- Policy output is normalized delta action, not a direct hardware command.
- Important mismatch to resolve in the new artifact contract: current `policy.yaml` says scale `1/24`, while current `revo3_profile.yaml` uses `action_scale: 0.02`.

## Replay-Mode Definition

Replay mode validates:
- Artifact layout and manifest contract.
- Normalizer loading and shapes.
- Dataset-to-observation construction.
- Policy inference output shape and finite values.
- Action clipping, action scaling, target integration, joint-limit clamp.
- Golden I/O reproduction.
- Deterministic replay log generation.

Replay mode must not:
- Start ROS launch files or controller managers.
- Subscribe to live robot topics.
- Create publishers or action clients.
- Publish `Revo3MITCommand`, `Float64MultiArray`, or trajectory actions.
- Open serial/Modbus/CAN devices.
- Depend on Docker script changes.

Inputs:
- `--artifact-dir`: contains model, manifest, normalizer, golden I/O.
- `--replay-data`: offline joint-state/target frames.
- `--robot-profile`: deployment clamp/order/offset profile.
- `--hand-side`: default `right`; no implicit left/right mirroring.
- `--output`: output directory or JSONL path.
- `--strict`: fail on any warning-grade contract mismatch.

Outputs:
- Per-step replay JSONL.
- Numeric arrays in `.npz` for analysis.
- Summary JSON/YAML containing manifest metadata, hashes, counts, max errors, warnings, and pass/fail status.

Replay vs shadow vs active:
- Replay: offline data only, no ROS I/O, no hardware.
- Shadow: live robot observations, policy/log output, no command publisher.
- Active: live observations and command publication to controller after replay and shadow pass.

## Training Artifact Requirements

Training must export one directory:

```text
artifact_dir/
  policy.onnx
  policy_manifest.yaml
  obs_normalizer.npz
  golden_io.npz
  replay_dataset.npz        # optional but recommended
```

`policy.onnx`:
- ONNX Runtime compatible.
- Named inputs exactly `obs` and `proprio_hist`.
- `obs`: `float32[B,126]`.
- `proprio_hist`: `float32[B,30,42]`.
- Output `action`: `float32[B,21]`.
- Dynamic batch allowed; replay uses `B=1`.

`policy_manifest.yaml` required fields:
- `schema_version`, `robot: revo3`, `hand_side`.
- `model.path`, `model.format: onnx`, `model.sha256`.
- `policy_rate_hz`, `policy_dt_sec`, `history_len: 30`, `obs_frames: 3`, `obs_per_step: 42`.
- `inputs` and `outputs` with names, shapes, dtypes.
- `joint_order_policy`, `controller_joint_order`, `joint_limits_rad`.
- `observation_schema`: first 21 normalized joint positions, next 21 current target radians.
- `normalization`: external normalizer convention, array names, epsilon, optional clip.
- `action`: `semantics: delta_position`, `clip: [-1, 1]`, `scale`, `target_clamp: joint_limits_rad`.
- `units`: joint position radians, velocity rad/s, effort mA.
- `training_metadata`: repo commit, checkpoint id, task name, export time.

`obs_normalizer.npz`:
- `obs_mean`: shape `(126,)`.
- `obs_std`: shape `(126,)`.
- `proprio_hist_mean`: shape `(30,42)`.
- `proprio_hist_std`: shape `(30,42)`.
- `epsilon`: scalar.
- Optional `clip`: scalar or two-value range.
- If normalization is intentionally baked into the model, export identity arrays and mark that explicitly in the manifest.

`golden_io.npz`:
- `joint_pos_policy_order`: `(T,21)` or at least `(1,21)`.
- `initial_target_policy_order`: `(21,)`.
- `obs_raw`: `(T,126)`.
- `proprio_hist_raw`: `(T,30,42)`.
- `obs_normalized`: `(T,126)`.
- `proprio_hist_normalized`: `(T,30,42)`.
- `action_raw`: `(T,21)`.
- `action_clipped`: `(T,21)`.
- `target_post_clip_policy_order`: `(T,21)`.
- Optional `target_controller_order_with_offset`: `(T,21)`.

Replay dataset format:
- Preferred `.npz`: `timestamp_sec (N,)`, `joint_pos_policy_order (N,21)`, optional `target_init (21,)`, optional `expected_action (N,21)`.
- JSONL is acceptable if it carries the same fields per frame.
- Timestamps must be strictly increasing.

## Validation And Fail-Fast Checks

Required fail-fast checks:
- `obs_dim == 126`, `action_dim == 21`, `history_len == 30`, `obs_per_step == 42`.
- ONNX input/output names, dtypes, and shapes match manifest.
- Normalizer arrays exactly match required shapes and contain finite values.
- `joint_order_policy` exactly matches replay data and golden I/O.
- `controller_joint_order` is a permutation of the same 21 joints.
- Joint limits exist for every policy joint and `upper > lower`.
- All replay inputs, observations, normalizer outputs, model outputs, and post-processed actions are finite.
- Raw action shape is `(21,)`; clipped action is within `[-1,1]`.
- Integrated targets remain within limits after clamp.
- Replay timestamps are strictly increasing; fixed-rate datasets must match `policy_dt_sec` within configured tolerance.
- Manifest `action.scale` must match deployment profile unless an explicit CLI override is provided and logged.
- No-hardware guarantee: replay executable must not create ROS publishers/action clients or import driver/controller modules.

## Proposed CLI

Preferred safe command:

```bash
ros2 run revo3_rl_deploy revo3_policy_replay \
  --artifact-dir /path/to/revo3_policy_export \
  --replay-data /path/to/replay_dataset.npz \
  --robot-profile /path/to/revo3_profile.yaml \
  --hand-side right \
  --output /tmp/revo3_policy_replay \
  --strict
```

Plain Python entry should do the same:

```bash
python3 -m revo3_rl_deploy.policy_replay_cli \
  --artifact-dir /path/to/revo3_policy_export \
  --replay-data /path/to/replay_dataset.npz \
  --robot-profile /path/to/revo3_profile.yaml \
  --output /tmp/revo3_policy_replay \
  --strict
```

CLI must not expose `--command-topic`, `--joint-state-topic`, or any hardware/control option.

## Output Log Format

Write `replay_steps.jsonl`, one frame per line:
- `frame_index`
- `timestamp_sec`
- `manifest_id`, `model_sha256`
- `joint_pos_policy_order`
- `target_prev_policy_order`
- `obs_raw`
- `obs_normalized`
- `proprio_hist_raw`
- `proprio_hist_normalized`
- `action_raw`
- `action_clipped`
- `target_post_clip_policy_order`
- optional `target_controller_order_with_offset`
- `warnings`

Write `replay_arrays.npz` with the same numeric arrays for plotting and comparison.

Write `summary.yaml`:
- artifact paths and hashes
- manifest metadata
- frame count and duration
- max abs golden errors for obs/action/target
- clipping counts
- validation status
- warnings
