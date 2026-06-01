# Reorient Task

[English version](README.md)

由一只朝下的灵巧手在手内对方块做 SO(3) 重定向。策略接收一个目标朝向（在掌心 "tag" 坐标系下），需要原地旋转方块，使其朝向在一个 hold 窗口内匹配目标，且不能掉落。

## 目录结构

- `mdp/` — 供 mjlab manager 系统使用的运行时 task term
  - `cage.py` — 掌心相对的 AABB "cage" 几何、逃逸计数器、奖励升级与终止
  - `command_visualization.py` — 目标 pose 的 Viser/MuJoCo GUI 渲染（marker、状态 markdown、debug triad）
  - `commands.py` — `InHandReorientCommand` / `InHandReorientCommandCfg`：SO(3) 目标状态机（sample → wait → resample）
  - `event_impl/` — 大型 event helper 的内部拆分（episode reset、随机化、关节 reset、curriculum、共享 event 状态）。不属于对外 API——请通过 `mdp.events` 门面导入
  - `event_utils.py` — 跨 event 模块共享的 helper（`get_linear_progress`、`resolve_env_ids`）
  - `actions.py` / `observations.py` / `rewards.py` / `terminations.py` / `curriculums.py` / `metrics.py` — 标准 mjlab term 模块
  - `events.py` — 公开 events 门面；re-export `event_impl/` 的 API
  - `types.py` — 共享 dataclass
- `reorient_terms.py` — 由 `make_reorient_env_cfg` 使用的 `build_reorient_*` 构造器
- `reorient_env_cfg.py` — 薄装配层；暴露 `make_reorient_env_cfg()`
- `reorient_constants.py` — task 级公开常量（如 `TAG_IN_PALM_POS`、`TAG_IN_PALM_QUAT_WXYZ`）
- `config/wuji_hand/` — Wuji Hand 硬件 overlay，下面带后端专属子目录（`rsl_rl/`）
  - `env_cfgs.py` — env config + play overlay（`_PLAY_DISABLED_EVENTS`）
  - `rsl_rl/ppo.py` — `wuji_hand_reorient_ppo_runner_cfg()`（RSL-RL PPO）
  - `__init__.py` — 注册 `WujiHand_Reorient`（release）与
    `WujiHand_Reorient_Light`（低显存变体）
- `tooling/` — 可导入、无副作用的实现，被 `scripts/` 使用。这里的任何东西都可以安全地从 Python（sweep、notebook、测试）导入而不会触发 CLI
  - `eval_core.py` — `EvalConfig`、`EvalResult`、`TrialOutcome`、`run_eval()`、`ObsBuilder`、ONNX 策略 eval 循环
  - `eval_display.py` — 用于实时 eval 输出的终端渲染器（`EvalDisplay`）
  - `scene_builder.py` — MuJoCo 场景构建、quat 数学、hold 状态机、contact helper（eval 与场景可视化共用）
  - `onnx_export_core.py` — ONNX 策略导出逻辑
- `scripts/` — 薄 CLI 封装（argparse + 委托给 `tooling/`）。这些是面向用户的入口；它们只负责环境变量设置（如 `MUJOCO_GL=glfw`）和 `if __name__ == "__main__"` 胶水
- `tests/` — task 本地的行为与流水线不变量测试（快速、无 sim；`conftest.py` + `fakes.py` 提供进程内 double）

## 对外接口

**Env config**:
- `make_reorient_env_cfg()` → `ManagerBasedRlEnvCfg`（与机器人无关的基线）
- `wuji_hand_reorient_env_cfg(play: bool = False)` → Wuji Hand 硬件 overlay

**RL config**:
- `wuji_hand_reorient_ppo_runner_cfg()` → RSL-RL PPO runner cfg

**Eval (programmatic)**:
- `from wuji_mjlab.tasks.reorient.tooling.eval_core import EvalConfig, EvalResult, run_eval`
- 把填好的 `EvalConfig` 传给 `run_eval()`，读回结构化的 `EvalResult`（success_rate、drop_rate、各 trial 结果等）

**已注册的 task ID**（见 `config/wuji_hand/__init__.py`）：
- `WujiHand_Reorient` — release 配置（`num_envs=8192`、`max_iterations=5000`，
  约需 ~20 GB 显存；复现已发布权重）
- `WujiHand_Reorient_Light` — 低显存变体（`num_envs=4096`、
  `max_iterations=7500`，可在 ~12 GB 卡上跑；策略明显较弱）

## 运行

train 与 play 走顶层 `pixi` task；通过 `--task` 传入 task ID：

```bash
pixi run train --task WujiHand_Reorient            # release 配置（8192 envs × 5000 iters，~20 GB）
pixi run train --task WujiHand_Reorient_Light      # 低显存变体（4096 envs × 7500 iters）
pixi run play  --task WujiHand_Reorient
pixi run list-envs                                  # show all registered task IDs
```

本子树常用的调试/eval 入口：

```bash
# Visualize the task scene
python -m wuji_mjlab.tasks.reorient.scripts.view_task WujiHand_Reorient

# Evaluate ONNX policy success rate
# Default: windowed MuJoCo viewer
python -m wuji_mjlab.tasks.reorient.scripts.eval_success_rate <onnx_path>

# Headless + machine-readable JSON output (good for CI / sweeps)
python -m wuji_mjlab.tasks.reorient.scripts.eval_success_rate <onnx_path> \
    --num-trials 100 --no-viewer --json-output result.json

# Export trained policy to ONNX (see --help for checkpoint paths)
python -m wuji_mjlab.tasks.reorient.scripts.export_onnx <path-to-ckpt.pt>
```

**Programmatic eval**（无 CLI，适合批量评测）：

```python
from pathlib import Path
from wuji_mjlab.tasks.reorient.tooling.eval_core import EvalConfig, run_eval

result = run_eval(EvalConfig(
    onnx_path=Path("logs/wuji_reorient/exp1/policy.onnx"),
    num_trials=50,
    no_viewer=True,
))
print(f"success_rate = {result.success_rate:.2%}")
print(f"mean min ori error = {result.mean_min_ori_error_rad:.3f} rad")

# Per-trial data for custom analysis
for trial in result.trials:
    if trial.status == "success":
        print(f"trial {trial.trial_idx}: t_first_succ={trial.time_to_first_success_s:.2f}s")
```

## 架构约束

- `mdp/event_impl/` 是内部模块。外部调用方只能通过 `mdp.events` 门面或顶层 `mdp` re-export 来使用 event。
- `ReorientEventState`（位于 `mdp/event_impl/state.py`）是所有 event 侧运行时缓存的唯一持有者。请始终通过 `get_reorient_event_state(env)` 访问；不要把缓存字段直接挂到 `env` 上。
- Curriculum 函数返回 `dict[str, float]` 以匹配 mjlab manager 契约——`_get_total_training_steps` 集中在 `mdp/event_impl/curriculum.py`，因此单次 rl-cfg 变更就能 rescale 所有 schedule。
- `tooling/` 是脚本逻辑的可导入、无副作用根；`scripts/` 只负责 argparse + 环境变量设置 + `if __name__ == "__main__"`。`tooling/` 中的任何东西都可以从 Python 导入并调用（如用于 sweep、测试或 notebook），而无需走 CLI。
- `config/wuji_hand/` 把 RL 配置放在后端专属子目录（`rsl_rl/`）下，这样如果加入其他后端，目录结构能自然扩展。
- 公开 API 表面（`make_reorient_env_cfg`、`wuji_hand_reorient_env_cfg`、`mdp.*` re-export、已注册的 task ID、`reorient_constants.*`、`tooling.eval_core.EvalConfig` / `EvalResult` / `run_eval`）保持稳定——内部模块可以移动，但外部调用方使用的名称不能变。
- `mdp/__init__.py` 有意使用 `from .module import *` 形式的 re-export，因此新增 term 模块时需要在那里加上对应的 wildcard import（并在新模块中加 `__all__`）。
