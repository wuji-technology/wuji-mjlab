# PenSpin · Train

[中文版](train_zh.md) · [Task home](../README.md) · [Setup](setup.md) · [Train](train.md) · [Deploy](deploy.md)

## Environment

Run all commands from the **repository root**. Training requires Linux x86_64, an NVIDIA GPU and the repository's CUDA 12.8/PyTorch environment. See [project installation](../../../../../README.md) for Pixi setup.

```bash
pixi install
pixi run list-envs
```

Task ID: `WujiHand2_PenSpin`.

## Dataset

| Group ID | Motion | Dataset download |
| --- | --- | --- |
| `01_single_axis` | Single-axis rotation | [Asset bundle](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip) |
| `02_continuous_single_axis` | Continuous single-axis rotation | [Asset bundle](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip) |
| `03_continuous_multi_axis` | Continuous multi-axis rotation | [Asset bundle](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip) |
| `04_stable_grasp_rotation` | Stable grasp rotation | [Asset bundle](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip) |
| `05_multi_axis_two_rotations` | Multi-axis two rotations | [Asset bundle](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip) |

All 40 clips across five groups and their recipes are included in the [v2026.9.27 asset bundle](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip). After extraction, copy `pen_spin/data/rawdata/clips/` and `pen_spin/data/recipe/` into the matching task directories below. Group clip counts are 10, 4, 9, 10 and 7 respectively. Existing local data can be used directly. Five matched checkpoint sets are also included under `pen_spin/checkpoints/`; see [Deploy](deploy.md) for placement.

```text
src/wuji_mjlab/tasks/pen_spin/data/
├── rawdata/clips/
│   ├── 01_single_axis/
│   │   ├── index.json
│   │   ├── clip_001.npz
│   │   └── ...
│   ├── 02_continuous_single_axis/
│   ├── 03_continuous_multi_axis/
│   ├── 04_stable_grasp_rotation/
│   └── 05_multi_axis_two_rotations/
└── recipe/
    └── 03_continuous_multi_axis.json
```

Trajectories are NPZ files with a **0.02 s (50 Hz)** frame interval. Place the dataset's `clips/` directory under `data/rawdata/`, preserving each group's `index.json`, NPZ filenames and contents. Training reads the catalog and verifies SHA-256 digests.

## Select a released motion group

Use the supplied trajectory selection file. For Group 3 (continuous multi-axis rotation), set:

```bash
TASK=src/wuji_mjlab/tasks/pen_spin
GROUP=03_continuous_multi_axis
RECIPE="$TASK/data/recipe/$GROUP.json"
```

To use another released group, set `GROUP` to its ID from the table above and set `RECIPE` again. Training and playback use the supplied JSON through `env.commands.motion.recipe_path`.

## Train

```bash
pixi run train --task WujiHand2_PenSpin \
  --agent.logger tensorboard --agent.upload-model False \
  env.commands.motion.recipe_path="$RECIPE" \
  env.scene.num_envs=4096 agent.max_iterations=10000 \
  agent.run_name="PenSpin_$GROUP"
```

**Around 10,000 PPO iterations are recommended for more stable pen spinning.** Select a checkpoint based on tracking error and playback quality. Reduce `env.scene.num_envs` to `2048` or `1024` if GPU memory is insufficient.

Logs and checkpoints are saved under:

```text
logs/rsl_rl/wuji_pen_spin_hand2/<timestamp>_<run_name>/
├── model_*.pt
└── params/
    ├── agent.yaml
    └── env.yaml
```

A checkpoint is saved every 500 iterations by default. Retain the complete run directory and your recipe. For W&B logging, set both `WANDB_PROJECT` and `WANDB_API_KEY` in your environment, then replace `--agent.logger tensorboard` with `--agent.logger wandb`. Missing or blank values stop training with an error; cached login credentials do not replace them. TensorBoard requires neither variable.

## Play a pretrained or custom policy

Choose the checkpoint corresponding to the recipe. The example below uses your own training result; replace `<run>` and the filename with the actual run and checkpoint:

```bash
CHECKPOINT="logs/rsl_rl/wuji_pen_spin_hand2/<run>/model_9999.pt"

pixi run play --task WujiHand2_PenSpin \
  --checkpoint-file "$CHECKPOINT" \
  env.commands.motion.recipe_path="$RECIPE"
```

For your own training runs, point `CHECKPOINT` to a PT file under `logs/rsl_rl/...`. Use the supplied group selection file for playback. Look for complete tracking, maintained grasp and smooth motion without obvious shaking or dropping.

## Export ONNX

```bash
pixi run export-onnx "$CHECKPOINT"
```

This creates `policy.onnx` and `config.json` beside the checkpoint. Keep the companion configuration and use the exported ONNX for deployment.

Deployment uses the supplied [`pen_spin_policy.json`](../../../../../deploy/pen_spin/config/pen_spin_policy.json) at 50 Hz. Keep the exported `config.json` with the model; the runtime does not load it automatically.

Next: use [Deploy](deploy.md) to select a policy and run it on the physical hand.
