# 转笔 · 部署

[English](deploy.md) · [任务首页](../README_zh.md) · [硬件搭建](setup_zh.md) · [训练](train_zh.md) · [部署](deploy_zh.md)

所有命令均在**仓库根目录**运行。硬件与打印资源见 [硬件搭建](setup_zh.md)，训练与导出见 [训练](train_zh.md)。

**启动顺序：先开相机观测，确认笔的位姿有效，再开策略。`pen-policy` 默认同时打开实时镜像窗口。** 具体命令见[启动相机与策略](#启动相机与策略)。

## 策略模型

| 分组 | 动作 |
| --- | --- |
| `01_single_axis` | 单轴旋转 |
| `02_continuous_single_axis` | 连续单轴旋转 |
| `03_continuous_multi_axis` | 连续多轴旋转 |
| `04_stable_grasp_rotation` | 稳定抓握旋转 |
| `05_multi_axis_two_rotations` | 多轴两次旋转 |

[v2026.9.27 合并资源包](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip) 提供转笔硬件、轨迹数据及 5 组配套模型，位于包内 `pen_spin/checkpoints/<动作组>/`。将包内 `pen_spin/checkpoints/` 复制到仓库的 `src/wuji_mjlab/tasks/pen_spin/data/checkpoints/`，保留动作组目录及下方所有配套文件；下载数据集与训练步骤见 [训练](train_zh.md)。

```text
src/wuji_mjlab/tasks/pen_spin/data/checkpoints/<group>/
├── model.pt
├── policy.onnx
└── config.json
```

发布包每组提供 `model.pt`、`policy.onnx` 和 `config.json`，请成套保留。部署使用 `policy.onnx` 与同组参考轨迹。自行训练时，另外保留原始训练目录中的 `params/`，用于复现训练配置。

部署使用仓库提供的 [`pen_spin_policy.json`](../../../../../deploy/pen_spin/config/pen_spin_policy.json)，控制频率为 50 Hz。导出的 `config.json` 随模型保留，运行时不会自动读取该文件。

## 硬件与环境

准备 Wuji Hand 2、250 mm 标签笔、腕部 AprilTag、Hikrobot 相机和厂商 MVS SDK。标签笔对应 178 mm 裸露笔杆、14.9 mm 杆直径。部署环境独立安装：

```bash
pixi install -e pen-spin-deploy
```

核对[相机标定](../../../../../deploy/pen_spin/config/camera.yaml)、[笔标签布局](../../../../../deploy/pen_spin/config/pen_tags.json)和[控制配置](../../../../../deploy/pen_spin/config/control.yaml)。相机内参应对应实际设备；MVS SDK 安装和装置检查见[硬件搭建说明](setup_zh.md)。

## 相机位置与观测质量

**转笔效果非常依赖相机观测**。部署前先找好相机位置，使笔位姿观测稳定、连续，尽量减少遮挡和运动模糊。装配实物图和相机观测参考图见[转笔硬件文档](setup_zh.md#5-相机摆位与光照)。

1. 将笔的两端标签、完整运动范围和腕部基准标签放入视野，避免手指、支架长期遮挡；调整焦距、曝光和补光，让运动中的标签仍然清晰。
2. 尽量采用能看到多个标签面的斜视角。出现 `1 faces / COPLANAR` 时表示当前为单面共面观测，可进一步调整角度以改善位姿稳定性。
3. 确认检测线框与实物贴合，笔轴、头尾和位置一致；检查整个动作过程中是否出现跳变、翻转或持续丢帧。
4. 固定世界坐标后保持相机和手座不动；移动后按预览窗口的 `w` 重新采样腕部基准，并再次检查对齐。

## 启动相机与策略

默认部署只需 **两个终端**，都从仓库根目录运行。`pen-policy` 会同时启动**真机策略控制和实时 MuJoCo 镜像窗口**，显示实际关节、相机观测到的笔和参考动作。这里的镜像窗口用于观察真机，并不意味着只在仿真中运行。

### 1. 先启动相机观测，并保持运行

```bash
# 终端 1：先运行，部署期间不要关闭
pixi run -e pen-spin-deploy pen-observer --preview
```

等待腕部标签采样完成、世界坐标固定，并确认笔的线框位置正确、位姿持续更新，再启动策略。仅打开相机窗口不代表观测已经就绪。相机未就绪时，策略会在连接手之前退出；应先恢复相机观测，再重新运行策略命令。

### 2. 再启动策略与实时镜像

先准备同组的 ONNX 和 NPZ；下面以第三组为例，路径必须指向已有文件。

```bash
# 终端 2：真机控制 + 实时镜像，默认打开窗口
pixi run -e pen-spin-deploy pen-policy \
  --policy src/wuji_mjlab/tasks/pen_spin/data/checkpoints/03_continuous_multi_axis/policy.onnx \
  --motion src/wuji_mjlab/tasks/pen_spin/data/rawdata/clips/03_continuous_multi_axis/clip_001.npz
```

**默认不用另开 `pen-sim`，也不用改成 `python -m runtime --sim`。** `pen-policy` 已默认启用镜像；如相机已运行，只执行第二步。同一只手一次只运行一个策略进程，切换动作组前先退出旧策略，相机进程可以保持运行。

部署其他组时，同时更换 ONNX 和 NPZ 路径。`--motion` 指向随包提供的一段 `.npz` 轨迹；多手场景通过 `--hand-sn "<序列号>"` 指定设备。

`pen-sim` 是可选的独立只读观察器，可用于启动策略前检查观测；它不是默认部署流程的必需进程。

**控制进程会驱动真机，启动前清空手指运动范围**。程序缓慢到达起始手型后进入 `FROZEN`：

| 操作 | 效果 |
|---|---|
| 第一次 Enter | 检查笔的起始位姿，进入 `HOLD`，保持第一帧 |
| 第二次 Enter | 进入 `POLICY`，开始沿参考轨迹执行 |
| `r` | 复位并张开手 |
| Ctrl+C | 退出进程；不能替代硬件急停 |

`--hold-last-frame` 可让动作结束后保持最后一帧。更多运行参数见下文。

## 策略约定与运行选项

直接使用 release 提供的 `.npz` 参考轨迹，并选择同一动作组的策略。运行时的观测与动作配置见 [`pen_spin_policy.json`](../../../../../deploy/pen_spin/config/pen_spin_policy.json)。

开始前会拒绝“笔未被看见、位姿过旧、位置偏差超过 50 mm、笔轴偏差超过
60 度或头尾放反”的情况。

`--hold-last-frame` 让参考停在最后一帧（参考速度为零），直到按 `r` 或 Ctrl+C。
`--open-loop` 使用轨迹中的关节状态作为策略推理输入。
`--pen-axis-offset-m d` 让实时笔观测沿笔轴向 A_top 平移 `d`。

## 笔标签布局

`deploy/pen_spin/config/pen_tags.json` 已包含当前 250 mm 笔（露出笔杆 178 mm）的完整布局，直接使用即可。以下命令检查布局能否加载，并输出标签数量和笔总长：

```bash
pixi run -e pen-spin-deploy python - <<'PY'
from lib.pen_pose import PenTagLayout
layout = PenTagLayout("deploy/pen_spin/config/pen_tags.json")
print(f"tags={len(layout)}, length_mm={layout.total_length * 1000:.1f}")
PY
```

预期输出 `tags=18, length_mm=250.0`。实物尺寸及贴图必须与布局一致。
