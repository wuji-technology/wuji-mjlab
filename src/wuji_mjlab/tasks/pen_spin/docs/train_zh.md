# 转笔 · 训练

[English](train.md) · [任务首页](../README_zh.md) · [硬件搭建](setup_zh.md) · [训练](train_zh.md) · [部署](deploy_zh.md)

## 环境准备

所有命令均在**仓库根目录**运行。训练需要 Linux x86_64、NVIDIA GPU，以及仓库配置的 CUDA 12.8／PyTorch 环境。安装 Pixi 的方法见[项目首页](../../../../../README_zh.md)。

```bash
pixi install
pixi run list-envs
```

任务 ID 为 `WujiHand2_PenSpin`。

## 数据集

| 数据组 ID | 动作 | 数据集下载 |
| --- | --- | --- |
| `01_single_axis` | 单轴旋转 | [合并资源包](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip) |
| `02_continuous_single_axis` | 连续单轴旋转 | [合并资源包](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip) |
| `03_continuous_multi_axis` | 连续多轴旋转 | [合并资源包](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip) |
| `04_stable_grasp_rotation` | 稳定抓握旋转 | [合并资源包](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip) |
| `05_multi_axis_two_rotations` | 多轴两次旋转 | [合并资源包](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip) |

5 组共 40 段轨迹及各组 recipe 随 [v2026.9.27 合并资源包](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip) 提供。解压后，将 `pen_spin/data/rawdata/clips/` 与 `pen_spin/data/recipe/` 分别复制到下方任务目录的对应位置。各组轨迹数依次为 10、4、9、10、7。本地已有这些数据时可直接使用。5 组配套 checkpoint 已包含在 `pen_spin/checkpoints/`，下载与放置说明见 [部署](deploy_zh.md)。

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

轨迹以 NPZ 格式提供，帧间隔为 **0.02 s（50 Hz）**。将数据包中的 `clips/` 放入 `data/rawdata/`；保持每组的 `index.json`、NPZ 文件名和内容一致。训练程序会根据索引读取轨迹并校验 SHA-256。

## 选择已发布的动作组

使用随包提供的轨迹选择文件。以第三组（连续多轴旋转）为例，在终端中设置：

```bash
TASK=src/wuji_mjlab/tasks/pen_spin
GROUP=03_continuous_multi_axis
RECIPE="$TASK/data/recipe/$GROUP.json"
```

使用其他已发布动作组时，将 `GROUP` 改为上表对应的 ID，并重新设置 `RECIPE`。训练和回放通过 `env.commands.motion.recipe_path` 读取随包提供的 JSON。

## 启动训练

```bash
pixi run train --task WujiHand2_PenSpin \
  --agent.logger tensorboard --agent.upload-model False \
  env.commands.motion.recipe_path="$RECIPE" \
  env.scene.num_envs=4096 agent.max_iterations=10000 \
  agent.run_name="PenSpin_$GROUP"
```

**建议训练到约 10,000 次 PPO 迭代，转笔效果通常会更稳定**。根据跟踪误差和回放效果选择 checkpoint。显存不足时，将 `env.scene.num_envs` 降为 `2048` 或 `1024`。

训练日志与 checkpoint 保存在：

```text
logs/rsl_rl/wuji_pen_spin_hand2/<时间>_<run_name>/
├── model_*.pt
└── params/
    ├── agent.yaml
    └── env.yaml
```

默认每 500 次迭代保存一次 checkpoint。保留完整训练目录及使用的 recipe。若使用 W&B，先显式设置 `WANDB_PROJECT` 和 `WANDB_API_KEY` 环境变量，并将 `--agent.logger tensorboard` 改为 `--agent.logger wandb`。缺失或空值会报错并停止训练，本机缓存登录凭据不能代替这些变量。TensorBoard 不需要这些变量。

## 回放预训练或自行训练的策略

选择与 recipe 对应的 checkpoint。以下以自行训练的结果为例，请将 `<run>` 和文件名替换为实际名称：

```bash
CHECKPOINT="logs/rsl_rl/wuji_pen_spin_hand2/<run>/model_9999.pt"

pixi run play --task WujiHand2_PenSpin \
  --checkpoint-file "$CHECKPOINT" \
  env.commands.motion.recipe_path="$RECIPE"
```

使用自己的训练结果时，将 `CHECKPOINT` 改为 `logs/rsl_rl/...` 下的 PT 文件。回放使用随包提供的分组选择文件，观察是否完成参考动作、保持抓握，以及是否出现明显抖动或掉笔。

## 导出 ONNX

```bash
pixi run export-onnx "$CHECKPOINT"
```

导出后，在 checkpoint 所在目录生成 `policy.onnx` 和 `config.json`。保留配套配置，真机部署使用导出的 ONNX。

部署使用仓库提供的 [`pen_spin_policy.json`](../../../../../deploy/pen_spin/config/pen_spin_policy.json)，控制频率为 50 Hz。导出的 `config.json` 随模型保留，运行时不会自动读取该文件。

下一步：按 [部署](deploy_zh.md) 导出或选取策略并部署到真机。
