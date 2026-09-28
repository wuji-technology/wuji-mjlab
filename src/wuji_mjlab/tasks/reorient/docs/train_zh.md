# 方块翻转 · 训练

[English](train.md) · [任务首页](../README_zh.md) · [硬件搭建](setup_zh.md) · [训练](train_zh.md) · [部署](deploy_zh.md)

## 环境与任务选择

先完成[环境安装](../../../../../README_zh.md#安装)。以下所有命令均在**仓库根目录**运行。

| 任务 ID | 手型 / 控制频率 |
|---|---|
| `WujiHand_Reorient` | Wuji Hand 1，右手，20 Hz |
| `WujiHand2_Reorient` | Wuji Hand 2，右手，20 Hz 基线 |
| `WujiHand2_Reorient_50Hz` | Wuji Hand 2，右手，50 Hz，推荐配置 |

## 开始训练

使用 W&B 必须显式设置 `WANDB_PROJECT` 和 `WANDB_API_KEY` 环境变量。
缺失或空值会报错并停止训练；本机缓存登录凭据不能代替这些变量：

```bash
export WANDB_PROJECT=your-wandb-project
read -rsp "W&B API key: " WANDB_API_KEY
export WANDB_API_KEY
pixi run train --task WujiHand2_Reorient_50Hz --agent.upload-model False
```

`--agent.upload-model False` 表示 checkpoint 只保存在本地。去掉该参数，最后一次迭代的 checkpoint 会作为模型文件上传到 W&B——无论是否上传，本地 `.pt` 都会照常周期性写入。完全不想用 W&B 时，改传 `--agent.logger tensorboard`，不需要上述两个环境变量。

模型检查点与 W&B 日志保存在 `logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/` 下。

将 `--task` 改为上表中的 ID 即可训练另一手型。通过点号参数覆盖任务默认值，例如：

```bash
pixi run train --task WujiHand2_Reorient_50Hz \
  --agent.logger tensorboard --agent.upload-model False \
  env.scene.num_envs=2048 agent.max_iterations=2000
```

可通过 `--config your.yaml` 复用任务与参数配置，格式见 `wuji_mjlab.utils.train_config_utils`；仓库不预置配置文件。

## 回放

回放时使用与训练一致的任务 ID。下面以 Wuji Hand 2 50 Hz 为例：

以下示例中的尖括号路径必须替换为本地真实文件；checkpoint 应保留同目录下的 `params/agent.yaml` 与 `params/env.yaml`。

```bash
pixi run play --task WujiHand2_Reorient_50Hz --checkpoint-file "<path-to-ckpt.pt>"
```

## 导出 ONNX

```bash
# 导出 policy.onnx 与配套 config.json 到 checkpoint 所在目录
pixi run python -m wuji_mjlab.tasks.reorient.scripts.export_onnx "<path-to-ckpt.pt>"
```

`--filename` 可指定导出 ONNX 的相对文件名。保留与策略一起导出的配置。

## 评测

```bash
# Wuji Hand 2 评测必须加 --hand hand2；Wuji Hand 1 使用 --hand hand1
pixi run python -m wuji_mjlab.tasks.reorient.scripts.eval_success_rate "<path-to-policy.onnx>" --hand hand2

# 无窗口评测并保存 JSON 结果
pixi run python -m wuji_mjlab.tasks.reorient.scripts.eval_success_rate "<path-to-policy.onnx>" \
  --hand hand2 --num-trials 100 --no-viewer --json-output "artifacts/eval/<run>/result.json"
```

将 `<run>` 替换为本次评测的运行名。`--json-output` 指定结构化 JSON 的路径；`cube_motion_log.csv` 和 `eval_results.json` 也写入同一父目录，目录会自动创建。不传该参数时，配套文件写在 ONNX 模型旁。不同评测运行请使用不同目录。

## 开发参考

目标朝向使用腕部 AprilTag 坐标系；相关 site 见 `mdp/observations.py` 的 `*_wrist_tag` 定义。

```bash
pixi run list-envs
pixi run python -m wuji_mjlab.tasks.reorient.scripts.view_task WujiHand2_Reorient_50Hz
```

### 目录结构

- `mdp/` — 任务项。
  - `cage.py` — 掌心相对的 抓握约束空间几何。
  - `command_visualization.py` — 目标可视化。
  - `commands.py` — SO(3) 目标命令。
  - `event_impl/` — 事件实现。
  - `_env_utils.py` — 环境 ID 辅助函数。
  - `runtime_state.py` — 逐环境状态。
  - `actions.py` / `observations.py` / `rewards.py` / `terminations.py`
    / `curriculums.py` / `metrics.py` — mjlab 任务项。
  - `events.py` — 事件导出。
  - `types.py` — 课程报告类型。
- `reorient_terms.py` — 任务项构造器。
- `reorient_env_cfg.py` — 环境配置。
- `reorient_constants.py` — 任务位姿常量。
- `config/wuji_hand/` — Wuji Hand 1 配置。
  - `env_cfgs.py` — 环境配置。
  - `rsl_rl/ppo.py` — PPO 训练器配置。
  - `__init__.py` — 任务注册。
- `config/wuji_hand2/` — Wuji Hand 2 配置。
  - `env_cfgs.py` — 环境与回放配置。
  - `hand2_constants.py` — 装置模型注入、碰撞胶囊覆盖、执行器增益及机器人/方块初始位姿常量。
  - `rsl_rl/ppo.py` — PPO 训练器配置。
  - `__init__.py` — 任务注册。
- `tooling/` — 评测与导出工具。
  - `cage_pose_check.py` — 抓握约束空间位姿检查。
  - `eval_core.py` — Python 策略评测。
  - `eval_display.py` — 评测显示。
  - `scene_builder.py` — 评测场景构建。
  - `onnx_export_core.py` — ONNX 策略导出。
  - `validate_obs_vs_sim.py` — 仿真与部署比较。
  - `cage_bounds_check.py` — 抓握约束空间边界检查。
  - `capsule_overlay_check.py` — 交互检查碰撞胶囊与碰撞网格 的实际凸包。
  - `hand2_check_scene.py` — 交互场景辅助函数。
  - `stamp_policy_metadata.py` — ONNX 策略元数据。
- `scripts/` — 命令行入口。
- `tests/` — 任务测试。
  - `integration/` — 集成测试。

### 对外接口

**环境配置**:
- `make_reorient_env_cfg()` → `ManagerBasedRlEnvCfg`（与机器人无关的基线）
- `wuji_hand_reorient_env_cfg(play: bool = False)` → Wuji Hand 硬件配置覆盖
- `wuji_hand2_reorient_env_cfg()` → Wuji Hand 2 硬件配置覆盖

**强化学习配置**:
- `wuji_hand_reorient_ppo_runner_cfg()` → RSL-RL PPO 训练器配置
- `wuji_hand2_reorient_ppo_runner_cfg()` → RSL-RL PPO 训练器配置

绕过 runner 手动创建训练模式环境时，在首次 step 前设置 `env.max_common_steps = agent_cfg.max_iterations * agent_cfg.num_steps_per_env`。标准训练入口、`view_task` 和 `play_runner` 已处理各自所需配置。

**通过程序调用评测**:
- `from wuji_mjlab.tasks.reorient.tooling.eval_core import EvalConfig, EvalResult, run_eval`
- 把填好的 `EvalConfig` 传给 `run_eval()`，读回结构化的 `EvalResult`（success_rate、drop_rate、各次试验结果等）
- `run_eval()` 返回 `EvalResult`；若要写出单独的结构化 JSON 文件，调用者需自行保存 `result.to_json()`。

**已注册的任务 ID**（见 `config/wuji_hand/__init__.py` 与
`config/wuji_hand2/__init__.py`）：
- `WujiHand_Reorient` — Wuji Hand 1，右手，20 Hz
- `WujiHand2_Reorient` — Wuji Hand 2，右手，20 Hz 基线
- `WujiHand2_Reorient_50Hz` — Wuji Hand 2，右手，50 Hz 部署配方（`decimation=2`）

**通过程序调用评测**（无需命令行，适合批量评测）：

```python
from pathlib import Path
from wuji_mjlab.tasks.reorient.tooling.eval_core import EvalConfig, run_eval

result = run_eval(
  EvalConfig(
    onnx_path=Path("logs/rsl_rl/wuji_reorient_hand2/<run>/policy.onnx"),
    hand="hand2",
    num_trials=50,
    no_viewer=True,
  )
)
print(f"success_rate = {result.success_rate:.2%}")
print(f"mean min ori error = {result.mean_min_ori_error_rad:.3f} rad")

# 使用每次试验的数据进行自定义分析
for trial in result.trials:
  if trial.status == "success":
    print(f"trial {trial.trial_idx}: t_first_succ={trial.time_to_first_success_s:.2f}s")
```

下一步：按 [部署](deploy_zh.md) 导出或选取策略并部署到真机。
