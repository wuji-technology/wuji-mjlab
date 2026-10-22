# Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses calendar versioning (YYYY.M.D).

## [Unreleased]

## [2026.10.22]

### Changed

- Wuji Hand 2: replace the vendored Beta 2 model with `hand2/hand2_2`
  (v2026.10.12) from the `wuji-description` submodule at
  `src/wuji_mjlab/assets/robots/wuji-description`. Fingertips move about 5 mm
  in the palm frame, the thumb distal link is about 48% lighter and most link
  inertias change; actuator gains, force limits and joint limits stay the same.
  Released Hand 2 Reorient checkpoints were trained on Beta 2. The loader keeps
  the palm as the root body with the upstream mount as its child (collision
  disabled in both tasks), and drops the wrist IMU sensors. Clone with
  `--recurse-submodules` or run `git submodule update --init`. PenSpin
  fingertip bump `force_scale` becomes 2.8 (thumb) / 1.6.
- Wuji Hand 1: load `hand/body-with-soft/mjcf/right_simplified.xml` from the
  same submodule and apply the task edits in code; the compiled model is
  unchanged.
- PenSpin actuators: derive kp/kv/forcerange from the deploy MIT gains, the
  motor torque constants, the identified 1.911 A per MIT unit and the 1.5 A
  limit. Joint damping and frictionloss come from the same identification, and
  joint armature is the vendor per-joint rotor inertia, with armature DR widened
  to 0.5-2x (log-uniform). The pen-spin deploy gains change with them: MIT kd
  is now 0.075 / 0.07 / 0.05 / 0.05 for joint1-4 (was 0.1 everywhere), set
  through the new `gains.kd_joint` / `--gains kd.<joint>=` overrides. Reorient
  actuators and deploy gains are unchanged. Existing PenSpin checkpoints were
  trained with the old actuators.

### Removed

- External documentation automation: documentation is now maintained in this
  repository. Remove docs-center preview/publish notifications, post-merge
  documentation follow-up labels and comments, Feishu notifications,
  `/docs-done` and `/docs-skip` comment callbacks, automatic merges into
  `docs-prerelease` and conflict issues, and the associated scripts and
  path-filter configuration.

## [2026.9.27]

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

## [2026.5.29]

### Added

- Released the initial public version with the `WujiHand_Reorient` task, a
  pretrained checkpoint, and the sim-to-real deployment bridge.

[Unreleased]: https://github.com/wuji-technology/wuji-mjlab/compare/v2026.10.22...HEAD
[2026.10.22]: https://github.com/wuji-technology/wuji-mjlab/releases/tag/v2026.10.22
[2026.9.27]: https://github.com/wuji-technology/wuji-mjlab/releases/tag/v2026.9.27
[2026.5.29]: https://github.com/wuji-technology/wuji-mjlab/releases/tag/v2026.5.29
