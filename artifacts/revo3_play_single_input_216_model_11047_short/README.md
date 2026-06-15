# Revo3 216-D Raw-Input Policy Artifact

Checkpoint: `logs/rsl_rl/revo3_reorient/2026-06-13_10-50-05_Revo3_Reorient_ReposeReward_FineTune_RobustDR_4gpu_from10048_1000it/model_11047.pt`

This artifact records the actual checkpoint interface: one raw `obs[1,216]` ONNX input and one `actions[1,21]` output. The empirical normalizer is inside `policy.onnx`; do not externally normalize before calling ONNX.

The saved `policy_input_normalized` arrays are audit-only and are used to verify `(raw - input_mean) / (input_std + epsilon)`.

This does not strictly match the old `MODEL_ACTION_EXTRACTION_README.md` schema of `obs[126] + proprio_hist[30,42]`; the manifest marks that compatibility as false.

## Human Rollout Command

```bash
pixi run python scripts/play/play_rsl_rl_zero_wrist_pitch.py --task Revo3RightHand_Reorient_ReposeReward_FineTune --checkpoint-file logs/rsl_rl/revo3_reorient/2026-06-13_10-50-05_Revo3_Reorient_ReposeReward_FineTune_RobustDR_4gpu_from10048_1000it/model_11047.pt --viewer native --num-envs 1 --fixed-object-init --single-goal --fixed-goal --fixed-goal-axis z --fixed-goal-deg 90.0 --no-terminations --video True --video-length 1000 --video-width 640 --video-height 480
```

## Export Command

```bash
pixi run python -m wuji_mjlab.tasks.reorient.scripts.export_revo3_play_artifact export --checkpoint logs/rsl_rl/revo3_reorient/2026-06-13_10-50-05_Revo3_Reorient_ReposeReward_FineTune_RobustDR_4gpu_from10048_1000it/model_11047.pt --output-dir artifacts/revo3_play_single_input_216_model_11047_short --task Revo3RightHand_Reorient_ReposeReward_FineTune --num-steps 5 --fixed-object-init --single-goal --fixed-goal --fixed-goal-axis z --fixed-goal-deg 90.0 --no-terminations
```

## Validation Command

```bash
pixi run python -m wuji_mjlab.tasks.reorient.scripts.export_revo3_play_artifact validate --artifact-dir artifacts/revo3_play_single_input_216_model_11047_short
```

Deployment/replay consumers should read `policy_manifest.yaml` and feed `policy_input_raw`-style 216-D observations to ONNX.
