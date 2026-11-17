# Run right-hand checkpoints on the left Hand 2

The playback adapter loads the official left Hand 2 model and converts its
observations to the coordinate convention of a right-hand checkpoint. It supports
the tasks listed in `SUPPORTED_TASKS` in
[`left_hand_playback.py`](../src/wuji_mjlab/utils/left_hand_playback.py).
The policy weights and corresponding joint commands stay unchanged.

This is experimental simulation playback. It transfers the right-hand actuator
calibration and does not establish performance on physical left-hand hardware.
Training and deployment commands keep their existing behavior.

## Play a checkpoint

Complete the [repository installation](../README.md#installation). Download the
checkpoint bundle and prepare the assets using the
[Reorient setup guide](../src/wuji_mjlab/tasks/reorient/docs/setup.md) or the
[PenSpin setup guide](../src/wuji_mjlab/tasks/pen_spin/docs/setup.md).
PenSpin requires materialized clip files and a recipe with matching hashes.

Set `wuji_assets` to the extracted checkpoint bundle, then run:

```bash
pixi run play --task WujiHand2_Reorient_50Hz --hand left \
  --checkpoint-file "$wuji_assets/reorient/checkpoints/hand2/model.pt"

pixi run play --task WujiHand2_PenSpin --hand left \
  --checkpoint-file "$wuji_assets/pen_spin/checkpoints/03_continuous_multi_axis/model.pt" \
  env.commands.motion.recipe_path=src/wuji_mjlab/tasks/pen_spin/data/recipe/03_continuous_multi_axis.json
```

To compare direct transfer, add `--no-left-policy-adapter`. Both variants use the
same left model, mirrored starting grasp and mirrored PenSpin references. The
flag changes only the policy observation conversion. Omit `--hand left` to keep
the existing right-hand playback.

## Record a comparison

Install `ffmpeg`, then run the bounded recorder on a CUDA machine. For a headless
host, select EGL before starting Python:

```bash
export MUJOCO_GL=egl
pixi run python scripts/play/record_hand_transfer.py \
  --task WujiHand2_Reorient_50Hz --hand left \
  --checkpoint "$wuji_assets/reorient/checkpoints/hand2/model.pt" \
  --steps 1000 --seed 42 --output outputs/cube-left-adapted.mp4

pixi run python scripts/play/record_hand_transfer.py \
  --task WujiHand2_PenSpin --hand left \
  --checkpoint "$wuji_assets/pen_spin/checkpoints/03_continuous_multi_axis/model.pt" \
  --recipe src/wuji_mjlab/tasks/pen_spin/data/recipe/03_continuous_multi_axis.json \
  --steps 1000 --seed 42 --output outputs/penspin-left-adapted.mp4
```

Repeat each command with `--no-left-policy-adapter` and a different output path
to record direct transfer. The recorder refuses to overwrite existing outputs.
On a multi-GPU machine, set `CUDA_VISIBLE_DEVICES` and `MUJOCO_EGL_DEVICE_ID` for
the selected CUDA and EGL devices.

Each MP4 has JSON metadata and a compressed NPZ trajectory beside it. The JSON
records source and model revisions, checkpoint and recipe hashes, camera
settings, reset causes and per-step task metrics. The NPZ stores joint positions,
actions and object poses. PenSpin also stores the reference object poses.
The video runs at simulation speed. Its encoded duration is `frames / fps`.

PenSpin play mode disables position and orientation error terminations. Inspect
the tracking errors and object trajectory to assess a drop. Zero resets alone
does not indicate success. A short single-seed comparison is not a benchmark.

## Coordinate and model conventions

The reflection uses `S = diag(1, -1, 1)` in the palm and world frames. Positions
transform as `S p`; rotations transform as `S R S`. Reflecting both frames gives
a proper rotation. The corresponding left and right hinge axes already encode
the required direction change, so joint values keep their anatomical order.

Cube playback reflects the initial object pose, collision capsules, calibration
rig and palm-normal cage margin. The adapter converts tag-local cube position
and the six-dimensional orientation error before observation history is built.

PenSpin mirrors wrist-local object and link reference poses before computing
world-frame velocities. It converts the reference command and observed shaft
points before inference. The joint references and residual actions stay in the
original scalar convention.

The left MJCF and meshes come from the repository's pinned `wuji-description`
submodule. Existing `right_*` identifiers remain internal task aliases so the
checkpoint's joint order and selectors remain stable. These aliases do not
select right-hand geometry. The official CAD models are not exact mirrors.

The configuration function returns a copy and does not patch global functions.
It keeps the critic and task metrics in the physical left-hand frame and changes
only the actor's coordinate observations. No left-hand training or hardware
deployment support is implied by this playback adapter.
