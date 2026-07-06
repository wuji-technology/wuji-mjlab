# wuji-mjlab

[English version](README.md)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Release](https://img.shields.io/github/v/release/wuji-technology/wuji-mjlab)](https://github.com/wuji-technology/wuji-mjlab/releases)
[![CI](https://github.com/wuji-technology/wuji-mjlab/actions/workflows/ci.yml/badge.svg)](https://github.com/wuji-technology/wuji-mjlab/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Stars](https://img.shields.io/github/stars/wuji-technology/wuji-mjlab?style=social)](https://github.com/wuji-technology/wuji-mjlab/stargazers)

> Wuji Hand 上的立方体手内翻转：基于 mjlab（底层物理由 mujoco-warp 提供 GPU 批量化）用 PPO 训练任意 SO(3) 目标姿态的策略，并通过 sim2real 桥在真机上做闭环部署。

<p align="center">
  <img src="docs/assets/sim.gif" width="45%" alt="sim reorient demo" />
  <img src="docs/assets/real.gif" width="45%" alt="real-hand reorient demo" />
</p>

## 任务

| 机器人 | 任务 ID | 预训练 checkpoint | 演示 |
|---|---|---|---|
| Wuji Hand | `WujiHand_Reorient` | [Latest release assets](https://github.com/wuji-technology/wuji-mjlab/releases/latest) | 上方 sim + real GIFs |

从最新 release 拉取 checkpoint 和 CAD 包：

```bash
# 需要 gh CLI (https://cli.github.com)；glob 通配使这条命令在未来的
# release tag 下也无需修改。若没有 gh，请参考 docs/sim2real/setup_zh.md §3
# 的手动 fallback。
gh release download --repo wuji-technology/wuji-mjlab --pattern '*-assets.zip'
unzip wuji-mjlab-*-assets.zip
mv wuji-mjlab-*-assets release-assets
```

## 仓库结构

```text
wuji-mjlab/
├── src/
│   ├── wuji_mjlab/        # task package (tasks/reorient/, assets/, utils/, rl/)
│   └── wuji_rl_libs/      # vendored rsl-rl (5.0.1+wuji1, min_std clamp)
├── deploy/reorient/       # sim2real bridge (vision, ZMQ, hand driver)
├── scripts/               # train / play / tools entry points
├── docs/                  # architecture + sim2real setup
├── pixi.toml              # canonical install + task runner
└── pyproject.toml         # package metadata
```

## 环境要求

- Linux x86_64
- NVIDIA GPU，CUDA 12.8（支持 Blackwell sm_120 / RTX 50 系列）
- [pixi](https://pixi.sh) ≥ 0.66（CI 用的版本）—— **唯一支持的安装方式**
- 用于 sim2real 部署：Wuji Hand 硬件 + Hikrobot USB-3 相机 + Hikvision MVS SDK + 3D 打印的 ArUco 标定 cube + 手腕 AprilTag——详见 [`docs/sim2real/setup_zh.md`](docs/sim2real/setup_zh.md)

> ⚠️ **注意**：本仓库**仅支持 pixi**。`conda + pip install -e .` 未经测试且不受支持。

## 安装

```bash
# 1. install pixi (one-time)
curl -fsSL https://pixi.sh/install.sh | bash

# 2. clone + resolve environment
git clone https://github.com/wuji-technology/wuji-mjlab
cd wuji-mjlab
pixi install
```

这一步会生成训练/评测用的 `default` 环境；如需 sim2real 桥，再执行 `pixi install -e deploy` 安装可选的 `deploy` 环境。

验证环境：`pixi run list-envs`（列出已注册任务，并验证 mjlab + tyro 栈可以正常 import）。

<details>
<summary><b>依赖下载缓慢的解决方案</b></summary>

如果 `pixi install` 拉取 PyTorch wheel 长时间卡住，按需将 `pixi.toml` 的 `[pypi-options]` 段替换为镜像源：

```toml
[pypi-options]
index-url = "https://mirrors.aliyun.com/pypi/simple/"
find-links = [{ url = "https://mirrors.aliyun.com/pytorch-wheels/cu128" }]
index-strategy = "unsafe-best-match"
```

⚠️ 此为本地修改，**不要 commit**——CI 的 lockfile 校验会失败。

</details>

## 训练

```bash
pixi run train --task WujiHand_Reorient --agent.upload-model False
```

`--agent.upload-model False` 表示 checkpoint 只保存在本地。去掉该参数（并设置 `WANDB_API_KEY`），最后一次迭代的 checkpoint 会作为 model artifact 上传到 W&B——无论是否上传，本地 `.pt` 都会在每个 `save_interval` 触发点写入。

> **若 `pixi run train` 出现 OOM**，切换到低显存变体：
>
> ```bash
> pixi run train --task WujiHand_Reorient_Light
> ```
>
> `WujiHand_Reorient` 用 `num_envs=8192, max_iterations=5000` 训练，近似复现 release 权重，约需 ~20 GB 显存。`WujiHand_Reorient_Light` 用 `num_envs=4096, max_iterations=7500`，能舒服跑在 ~12 GB 卡上，但策略明显较差——偶尔 drop cube、在较难的目标姿态下卡手。

Checkpoints 与 W&B 日志保存在 `logs/rsl_rl/<run_name>/` 下。任务 MDP、reward shaping，以及按解剖结构分两组的接触参数域随机化（手掌 + 拇指柔顺区 vs 食指到小指）详见下方架构章节。

## 回放与评测

```bash
# Interactive viewer with a trained checkpoint
pixi run play --task WujiHand_Reorient --checkpoint-file <path-to-ckpt.pt>

# Success-rate eval over N trials (consumes ONNX)
pixi run python -m wuji_mjlab.tasks.reorient.scripts.eval_success_rate <path-to-policy.onnx>

# Export PPO checkpoint → ONNX (sidecar JSON with action_scale / ema_alpha / ctrl_dt)
pixi run python -m wuji_mjlab.tasks.reorient.scripts.export_onnx <path-to-ckpt.pt>
```

其他开发工具：

```bash
pixi run list-envs                                                              # list registered tasks
pixi run python -m wuji_mjlab.tasks.reorient.scripts.view_task WujiHand_Reorient  # view task with a dummy policy
```

## Sim-to-real

<p align="center">
  <img src="docs/assets/deploy.gif" width="80%" alt="sim2real 部署装置：相机俯拍 Wuji Hand + 治具，右侧 MuJoCo mirror viewer 同步" />
</p>

部署桥在真机 Wuji Hand 上跑导出的 ONNX 策略：视觉模块用 USB 相机追踪带 ArUco 标签的立方体（以手腕 AprilTag 定义世界坐标系），把位姿通过 ZMQ 发布出去；`play_real` 订阅位姿、跑 ONNX 推理，把动作命令闭环下发到手部驱动。

> **部署无需自己训。** 从 [Releases](https://github.com/wuji-technology/wuji-mjlab/releases) 下载预训练好的 `policy.onnx` + `policy_config.json`，把 `policy.onnx` 路径作为下面命令的 `--ckpt` 传入即可。上面 demo GIF 就是这个 release 策略跑出来的。

```bash
# 完成 docs/sim2real/setup_zh.md 的硬件设置后:
pixi run -e deploy home                              # reset hand to home pose
pixi run -e deploy vision                            # launch cube observer (OpenCV preview)
pixi run -e deploy play-real --ckpt <path-to.onnx>   # closed-loop control + mirror viewer
```

- **软件流水线与配置**：[`deploy/reorient/README_zh.md`](deploy/reorient/README_zh.md)
- **硬件搭建、3D 打印立方体、相机安装、标定**：[`docs/sim2real/setup_zh.md`](docs/sim2real/setup_zh.md)

## 架构

三层架构：本仓库（tasks + deploy）→ [mjlab](https://github.com/mujocolab/mjlab) → [MuJoCo](https://mujoco.org) + [mujoco-warp](https://github.com/google-deepmind/mujoco_warp)。PPO 后端是内嵌（vendored）的 [rsl-rl](https://github.com/leggedrobotics/rsl_rl)，源码在 `src/wuji_rl_libs/rsl_rl/`。

<details>
<summary>深入展开 — 三层栈图、MDP 规格、域随机化、新增任务流程</summary>

```
  +--------------------------------------------------------+
  | wuji-mjlab (this repo)                                 |
  |  +----------------------+  +-------------------------+ |
  |  | tasks/reorient/      |  | deploy/reorient/        | |
  |  |   - env cfg + MDP    |  |   - real-hand env       | |
  |  |   - 2-group DR       |  |   - vision pipeline     | |
  |  |   - eval + export    |  |   - closed-loop control | |
  |  +----------------------+  +-------------------------+ |
  |  +----------------------+                              |
  |  | utils/               |  <- shared building blocks   |
  |  +----------------------+                              |
  |  +----------------------+                              |
  |  | rl/                  |  <- thin RL backend adapter  |
  |  +----------------------+                              |
  |                                                        |
  |  src/wuji_rl_libs/rsl_rl/ <- vendored PPO backend      |
  +--------------------------------------------------------+
              |                            |
              v                            v
  +-----------------------+  +---------------------------+
  | mjlab (pip / pixi)    |  | torch + onnxruntime       |
  | + mujoco-warp         |  | (training + inference)    |
  | + mujoco              |  |                           |
  +-----------------------+  +---------------------------+
```

### Reorient 任务 (`src/wuji_mjlab/tasks/reorient/`)

基于 Wuji Hand 的完整 SO(3) 手内重定向任务。文件分工：

| 文件 | 作用 |
|---|---|
| `reorient_env_cfg.py` | 顶层 `ManagerBasedRlEnvCfg` 工厂 |
| `reorient_terms.py` | 所有 event / termination / reward / DR terms（**任务设计**位于此处，而非机器人 binding 中） |
| `reorient_constants.py` | 初始位姿常量（palm-up R_y(-90°)、cube 位于手掌上方） |
| `config/wuji_hand/` | 机器人 binding 层：将任务设计精简地接入 Wuji Hand（20 自由度灵巧手） |
| `mdp/` | 重定向任务特有的观测、命令、动作 |
| `tooling/` | 评测入口 + ONNX 导出 |

任务设计（MDP terms、reward shaping、按解剖结构拆分的接触参数 DR 组）位于 `reorient_terms.py`。详见 [`src/wuji_mjlab/tasks/reorient/README_zh.md`](src/wuji_mjlab/tasks/reorient/README_zh.md)（架构约束）和 [`deploy/reorient/README_zh.md`](deploy/reorient/README_zh.md)（sim2real 桥）—— `RealHandEnv` 原封不动地复用 sim 中的 observation + action managers，无并行流水线。

### 新增任务

1. 在 `src/wuji_mjlab/tasks/<your_task>/` 下创建 env cfg 工厂。
2. 将所有 MDP 设计（events、rewards、terminations）放入 `<your_task>_terms.py`。`config/<robot>/` 下的机器人专属配置层应仅为精简的 binding。
3. 在 `config/<robot>/__init__.py` 中通过 `register_mjlab_task()` 注册。
4. 提交前先跑一遍 `pixi run train --task <your_task_id>` 冒烟（标准训练入口为 `scripts/train/train_rsl_rl.py`，通过 `train` pixi 任务暴露）。

</details>

## 开发

克隆仓库后，安装 pre-commit hooks：

```bash
pixi run pre-commit install
```

之后每次 `git commit` 都会跑 ruff、codespell 以及 [`.pre-commit-config.yaml`](.pre-commit-config.yaml) 中定义的 YAML/TOML / 大文件检查 —— 在本地拦截，不用等 CI 报红。手动全量跑：`pixi run pre-commit run --all-files`。

> ⚠️ 不要在 pixi 环境里用 `pip install` —— pip 装的依赖不会被 `pixi.toml` / `pixi.lock` 追踪，重新解析时会消失。要加依赖请改 `pixi.toml` 然后 `pixi install`。

## 相关项目

- [wujihandpy](https://github.com/wuji-technology/wujihandpy) — Wuji Hand SDK（C++ 内核 + Python 绑定）
- [wuji-retargeting](https://github.com/wuji-technology/wuji-retargeting) — 手部姿态重定向（Vision Pro / 数据手套 / 视频 → 机器人关节）
- [wujihandros2](https://github.com/wuji-technology/wujihandros2) — Wuji Hand 的 ROS 2 驱动
- [docs.wuji.tech](https://docs.wuji.tech) — Wuji 官方文档中心

## 致谢

本项目依赖以下开源工作：

- [mjlab](https://github.com/mujocolab/mjlab) — manager-based RL 框架
- [mujoco-warp](https://github.com/google-deepmind/mujoco_warp) — MuJoCo 物理仿真的 GPU 批量化实现
- [MuJoCo](https://mujoco.org/) — 底层物理引擎
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) — PPO 实现（已内嵌至 `src/wuji_rl_libs/`）
- [pupil-apriltags](https://github.com/pupil-labs/apriltags) — 部署视觉模块使用的 AprilTag 检测器

## 贡献者

- [Jielin Wu](https://github.com/AIRJASON50)
- [Shenzhe Yao](https://github.com/LeopoldYao)
- [Han Yang](https://github.com/yanghan-a)
- [Xiangrui Jiang](https://github.com/XiangruiJiang)
- [Wentao Zhang](https://github.com/zhangwt20011015)
- [Li Chengmeng](https://github.com/AsahelLee)
- [Xiaohan Liu](https://github.com/Infas12)
- [Guanqi He](https://github.com/GuanqiHe)

## 引用

如果本项目对你有帮助，欢迎引用：

```bibtex
@software{wuji2026mjlab,
  title={Wuji-MJLab: RL Training for Wuji Hand Dexterous Manipulation},
  author={{Wuji Technology}},
  year={2026},
  url={https://github.com/wuji-technology/wuji-mjlab}
}
```

## 许可协议

Apache 2.0。详见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE) 中关于第三方组件的署名信息。
