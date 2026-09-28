# Changelog


## [Unreleased]

### Added

- `deploy/reorient/scripts/run_policy.py`: require an operator to press
  Enter after homing before the ONNX policy takes control of the real
  hand (skipped in `--mock` mode).
- sim2real setup: dimensioned hand-jig assembly drawing (`assembly.pdf`)
  shipped in the release bundle and referenced from §5.1.

### Changed

- Require explicit `WANDB_PROJECT` and `WANDB_API_KEY` for W&B logging;
  TensorBoard and local checkpoint playback/video recording remain usable
  without W&B credentials.
- Align release documentation with five PenSpin checkpoint sets, motion clips
  and 3MF production files.

- Reorient: bind fingertip slip/contact terms by finger, aggregate palm contacts,
  measure saturation against actuator force ranges, restore initial/reset action
  targets, isolate per-world command resets, and disable play-time observation injection.
- Reorient disturbance ramp: use the active runner's training budget. The default
  50 Hz run reaches full amplitude at iteration 4000 instead of 1600; both 20 Hz
  schedules remain unchanged.
- Reorient standalone viewers and dummy-policy play: set the disturbance budget
  from the selected agent configuration before stepping training-mode environments.
- Reorient Viser: accept the viewer's command-GUI callbacks and honor the cage
  debug-visualization toggle.
- Reorient eval: place companion CSV/results beside `--json-output`, creating
  its parent directory when needed. Without that option, outputs remain beside
  the ONNX model.
- sim2real setup §5.1: actual assembly-drawing specs (overall height, 10° back-tilt, M6 hole
  grid, base counterbore pattern, general tolerances).
- upgrade the simulation stack from `mjlab` 1.3.0 to 1.6.0, `mujoco`
  and `mujoco-warp` 3.8.1 to 3.11.0, and `warp-lang` 1.12.1 to
  1.17.0.
- replace the vendored rsl-rl fork with official `rsl-rl-lib==5.4.2`.
- wrap the `scripts/record_video.py` rollout in `torch.inference_mode()`,
  fixing the mjlab 1.6.0 crash.
