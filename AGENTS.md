# AGENTS.md

## 0. Purpose

This file provides working rules for AI coding agents in this repository.

Treat this repository as a real robotics/RL training codebase. The code may affect policy training, simulation rollout, artifact export, and downstream real-robot deployment. Do not treat it as a scratchpad.

The main rule is:

> Preserve the actual trained policy interface. Do not fake deployment compatibility.

---

## 1. Project Context

This repository is used for dexterous-hand RL training, simulation rollout, policy evaluation, policy export, artifact generation, and simulation replay.

Current high-priority workflow:

```text
training/simulation repo
  -> export real policy artifact
  -> policy_manifest.yaml
  -> golden_io.npz
  -> replay_dataset.npz
  -> deployment repo offline replay
  -> deployment shadow mode
  -> deployment active mode
```

The current Revo3 right-hand policy uses the actual interface:

```text
raw obs[216] -> policy.onnx with internal empirical normalizer -> actions[21]
```

Important:

* The ONNX policy expects raw 216-D input.
* The empirical normalizer is inside the ONNX graph.
* Deployment must not externally normalize before ONNX.
* `obs_normalizer.npz` is for audit and validation, not runtime preprocessing.
* The older deployment expectation `obs[126] + proprio_hist[30,42]` is incompatible with this policy.
* This mismatch must be recorded explicitly in the manifest, not hidden.

---

## 2. Protected Files

The following files are protected. Do not delete, rename, overwrite, or clean them unless the user explicitly asks:

```text
AGENTS.md
AGENTS.override.md
MODEL_ACTION_EXTRACTION_README.md
policy_manifest.yaml
README.md
```

Before running cleanup commands such as:

```bash
git clean
rm
find -delete
git restore
git reset --hard
```

check whether protected files would be affected.

If `AGENTS.md` appears untracked or modified, ask before touching it.

---

## 3. Core Agent Rules

Always follow these rules:

* Do not modify files before understanding the relevant code path.
* For non-trivial work, first inspect the code and propose a plan.
* Prefer low-intrusion changes.
* Prefer adding small dedicated tools over editing core training/play/environment files.
* Do not refactor unrelated code.
* Do not change training behavior unless explicitly requested.
* Do not change observation semantics.
* Do not change action semantics.
* Do not change reward logic.
* Do not change reset logic.
* Do not change termination logic.
* Do not change domain randomization logic.
* Do not change policy architecture unless explicitly requested.
* Do not fake compatibility with deployment specs.
* Export the actual interface used by the trained checkpoint.
* If the actual interface mismatches a deployment spec, report the mismatch clearly.
* Do not add ROS2, CAN, hardware, controller, driver, or deployment dependencies to this training repository.
* Do not import the deployment repository.
* Do not create temporary debug scripts in the repository.
* Do not leave files named like `tmp_*.py`, `debug_*.py`, `quick_test_*.py`, `run_once.py`, or similar.
* Do not commit generated artifacts, videos, caches, logs, checkpoints, or large `.npz/.onnx/.pt` files unless explicitly requested.

---

## 4. Required Work Process

For any feature, export tool, replay tool, or integration change, use this process:

```text
read-only audit
  -> low-intrusion plan
  -> implementation after approval
  -> local validation
  -> diff review
  -> risk and rollback report
```

Before implementation, report:

* current entry point
* relevant files
* current call path
* existing reusable code
* minimal files to add or modify
* files that must not be touched
* risks and assumptions

If another file must be edited outside the approved plan, stop and explain first.

---

## 5. Git Hygiene

Before editing:

```bash
git status --short
git diff --stat
```

After editing:

```bash
git status --short
git diff --stat
```

A good diff usually looks like:

```text
new small exporter or replay tool
new focused validation code
small README or manifest update
small test update
```

A suspicious diff looks like:

```text
large edits to environment files
large edits to reward/reset/randomization logic
large edits to play scripts
unrelated formatting
temporary debug scripts
generated artifacts in git
deleted AGENTS.md
deleted docs
```

Never hide unrelated dirty files.

If the working tree is dirty before starting, report it before making changes.

---

## 6. Artifact Export Rules

When exporting a Revo3 policy artifact, preserve the real model interface.

The expected artifact layout is:

```text
artifact_dir/
  policy.onnx
  policy_manifest.yaml
  obs_normalizer.npz
  golden_io.npz
  replay_dataset.npz
  README.md
```

Optional but recommended:

```text
compatibility_report.md
checksums.sha256
```

The artifact must not include training source code, temporary logs, videos, cache files, or old scratch scripts.

The machine-readable contract is:

```text
policy_manifest.yaml
```

Do not treat `MODEL_ACTION_EXTRACTION_README.md` as the artifact contract. It is a reference/specification document only.

---

## 7. Revo3 216-D Policy Contract

For the current Revo3 right-hand policy, the manifest must describe the actual schema:

```yaml
schema: revo3_play_single_input_216_v1
model_input_kind: raw_obs_with_normalizer_inside_model

model:
  input_name: obs
  input_shape: [1, 216]
  output_name: actions
  output_shape: [1, 21]

normalization:
  applied_inside_model: true
  external_preprocessing_required: false
  epsilon: 1.0e-2

compatibility:
  matches_model_action_extraction_readme: false
  old_expected_schema: "obs[126] + proprio_hist[30,42]"
  actual_schema: "single_input[216]"
```

The 216-D input slice schema should be recorded explicitly:

```text
0:63     noisy_joint_angles    3 x 21
63:126   qpos_error            3 x 21
126:135  cube_pos_in_tag       3 x 3
135:153  cube_ori_error        3 x 6
153:216  action_history        3 x 21
```

If a future checkpoint has a different interface, fail clearly. Do not silently adapt it to this schema.

---

## 8. Normalization Boundary

The current ONNX policy includes the empirical normalizer.

Correct inference path:

```text
policy_input_raw[216]
  -> policy.onnx
  -> actions[21]
```

Incorrect inference path:

```text
policy_input_raw[216]
  -> external normalization
  -> policy.onnx
  -> actions[21]
```

The incorrect path double-normalizes and is wrong.

`obs_normalizer.npz` should still be exported for audit and validation:

```text
input_mean: (216,)
input_std: (216,)
epsilon: scalar, 1e-2
input_var: optional, (216,)
count: optional scalar
```

Validation should confirm:

```text
policy_input_normalized == (policy_input_raw - input_mean) / (input_std + epsilon)
policy.onnx(policy_input_raw) == action_raw
```

Do not feed `policy_input_normalized` into ONNX unless a manifest explicitly says the model expects normalized input.

---

## 9. Golden I/O and Replay Dataset

`golden_io.npz` should contain at least:

```text
policy_input_raw: (1,216)
policy_input_normalized: (1,216)
action_raw: (1,21)
action_clipped: optional, (1,21)
target_post_clip_policy_order: optional, (1,21)
joint_pos_policy_order: optional, (1,21)
object_state: optional, (1,7)
goal_state: optional, (1,4)
joint_names_policy_order: (21,)
```

`replay_dataset.npz` should contain:

```text
frame_index: (T,)
timestamp_sec: (T,)
policy_input_raw: (T,216)
policy_input_normalized: (T,216)
action_raw: (T,21)
action_clipped: optional, (T,21)
target_post_clip_policy_order: optional, (T,21)
joint_pos_policy_order: optional, (T,21)
object_state: optional, (T,7)
goal_state: optional, (T,4)
done: (T,)
reset: (T,)
episode_id: (T,)
joint_names_policy_order: (21,)
```

All arrays must be finite unless explicitly documented otherwise.

---

## 10. Simulation Replay Rules

Simulation replay is not the same as deployment replay.

Simulation replay may read:

```text
artifact_dir/replay_dataset.npz
```

and replay saved actions in the simulator.

Rules:

* Do not change environment semantics.
* Do not change policy semantics.
* Do not change reward/reset/termination/randomization logic.
* Do not claim exact state reproduction unless object and joint state errors are actually compared.
* `action_open_loop` mode should use saved actions from `replay_dataset.npz`.
* `policy_recompute` mode may recompute actions from `policy.onnx`.
* In `policy_recompute`, feed raw 216-D input into ONNX.
* Do not externally normalize before ONNX.

If `--max-steps` exceeds dataset length, fail unless `--loop` is explicitly set.

Generated simulation replay files such as:

```text
sim_replay_log.npz
sim_replay_summary.yaml
videos/
*.mp4
```

should not be committed unless explicitly requested.

---

## 11. Deployment Handoff Rules

When packaging an artifact for deployment, include only the handoff files:

```text
policy.onnx
policy_manifest.yaml
obs_normalizer.npz
golden_io.npz
replay_dataset.npz
README.md
compatibility_report.md     optional
checksums.sha256            optional
```

The handoff must state clearly:

```text
actual schema: single_input[216]
ONNX input: raw 216-D obs
normalizer: inside ONNX
external normalization required: false
old 126+proprio schema compatibility: false
deployment replay must be manifest-driven
```

Do not include:

```text
training source code
checkpoints unless explicitly requested
videos
cache files
temporary logs
debug scripts
old failed artifacts
```

---

## 12. Pixi and Commands

Use `pixi run` for repository commands when the repository environment expects Pixi.

Examples:

```bash
pixi run python -m wuji_mjlab.tasks.reorient.scripts.export_revo3_play_artifact validate \
  --artifact-dir artifacts/revo3_play_single_input_216_model_11047
```

For long rollouts, first test with a short run:

```text
short test: 5 steps
full artifact: 1000 steps
long artifact: 2000+ steps only if requested
```

Do not run long training or long rollout jobs unless the user explicitly asks.

---

## 13. Tests and Validation

Prefer focused validation over broad experiments.

For artifact export, validate:

* manifest loads
* ONNX loads
* ONNX input/output names match manifest
* ONNX input/output shapes match manifest
* `policy.onnx(policy_input_raw)` matches `action_raw`
* normalizer arrays are finite and shape `(216,)`
* `policy_input_normalized` formula matches exported mean/std/epsilon
* replay dataset arrays are finite
* replay dataset frame count matches requested length
* old README compatibility is explicitly false

If validation fails, explain the root cause. Do not hide failures.

---

## 14. Generated Files and .gitignore

Generated files should normally stay untracked.

Common generated paths:

```text
artifacts/
logs/
runs/
videos/
*.onnx
*.npz
*.npy
*.pt
*.pth
*.mp4
sim_replay_log.npz
sim_replay_summary.yaml
```

Before committing, check:

```bash
git status --short
```

If generated artifacts appear in git status, ask whether they should be committed or ignored.

Do not delete generated artifacts unless the user asks.

---

## 15. Forbidden Behaviors

Do not do the following:

* Do not delete `AGENTS.md`.
* Do not rename `AGENTS.md`.
* Do not remove protected documentation.
* Do not modify deployment code from this training repo.
* Do not import deployment packages.
* Do not add ROS2 dependencies.
* Do not add hardware dependencies.
* Do not change reward terms.
* Do not change reset logic.
* Do not change termination logic.
* Do not change domain randomization.
* Do not change observation construction to satisfy a deployment spec.
* Do not change action semantics to satisfy a deployment spec.
* Do not fake `obs[126] + proprio_hist[30,42]` compatibility.
* Do not externally normalize inputs to an ONNX model that already contains the normalizer.
* Do not claim exact simulation reproduction without state-error checks.

---

## 16. Final Response Format After Code Changes

After completing changes, respond with:

```text
Summary:
- ...

Changed files:
- path/to/file.py: reason
- path/to/test.py: reason

Validation:
- command run
- result

Generated artifacts:
- path/to/artifact_dir
- whether tracked or untracked

Risks:
- ...

Rollback:
- ...
```

If validation was not run, write:

```text
Validation:
- Not run. Reason: ...
```

Never claim a command was run if it was not run.

---

## 17. When Unsure

When uncertain:

* stop and inspect more code
* produce a read-only report first
* ask before modifying files
* prefer small isolated tools
* preserve actual policy interfaces
* record mismatches explicitly
* do not fake compatibility
* do not touch protected files
