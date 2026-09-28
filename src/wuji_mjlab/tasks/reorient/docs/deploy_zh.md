# 方块翻转 · 部署

[English](deploy.md) · [任务首页](../README_zh.md) · [硬件搭建](setup_zh.md) · [训练](train_zh.md) · [部署](deploy_zh.md)

所有命令均在**仓库根目录**运行。硬件与打印资源见 [硬件搭建](setup_zh.md)，训练与导出见 [训练](train_zh.md)。

**启动顺序：先启动相机观测，确认方块位姿有效，再启动策略。可选镜像窗口由独立进程运行。** 命令见[启动相机与策略](#启动相机与策略)。

## 策略模型下载

| 策略 | 下载链接 | 包内位置 |
|---|---|---|
| Wuji Hand 1 · Reorient | [合并资源包](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip) | `reorient/checkpoints/hand1/` |
| Wuji Hand 2 · Reorient | [合并资源包](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip) | `reorient/checkpoints/hand2/` |

独立的 **numpy + onnxruntime** 运行时，用于方块翻转策略——
部署时不导入 torch/mjlab。同一份部署目录树通过 `--gen {1,2}`
**同时**支持Wuji Hand 1 和 Wuji Hand 2：

| | gen1（`wuji_hand`） | gen2（`wuji_hand2`） |
|---|---|---|
| 驱动 | `wujihandpy` | `wuji_sdk` |
| 手别 | 仅右手 | 仅右手 |

策略权重与源码分开提供，资源包入口： [`wuji-mjlab` release `v2026.9.27`](https://github.com/wuji-technology/wuji-mjlab/releases/tag/v2026.9.27)。

**跑 `--gen 1` 需要已发布的 gen1 策略——它未随本仓提交，见下文「策略」一节。**

## 架构：3 个 ZMQ 进程

```
终端 1（观测）：     cube_world_observer.py  --preview
                          相机 -> AprilTag/ArUco PnP -> 方块位姿
                          发布到 ZMQ:5555（默认值）

终端 2（控制）：  run_policy.py --gen N --hand H
                          方块位姿（默认 5555）-> obs[207] -> ONNX -> action[20]
                          -> 插值关节目标 -> 手部驱动
                          发布目标四元数到 ZMQ:5556（默认值），关节到
                          ZMQ:5557（默认值）

终端 3（可选）：   render_viewer.py --gen N --hand H
                          订阅默认端口 5555（方块）、5556（目标）、
                          5557（关节）
                          显示 MuJoCo 镜像场景；仅负责可视化，
                          不运行策略或控制硬件
```

实际端口统一取自 `config/control.yaml` 的 `zmq` 段，三个进程都跟随。

终端 3 是可选的，纯粹是一个可视化镜像；闭环由 终端 1 + 2
构成（相机 -> 策略 -> 手）。

## 安装

单一 pixi 环境 `reorient-deploy`（覆盖Wuji Hand 1 和 Wuji Hand 2）：

```bash
pixi install -e reorient-deploy
```

有一个依赖对主机有版本下限，另一个根本不在 PyPI 上：

- **`wuji_sdk` 需要主机上 glibc ≥ 2.34**。
- **Hikvision MVS 相机 SDK**（`MvImport`）是厂商安装包，不在 PyPI 上。
  请从 https://www.hikrobotics.com 安装（默认路径 `/opt/MVS`；如安装在
  别处，用 `MVS_PYTHON_PATH` 环境变量覆盖）。它只在
  `cube_world_observer.py` / `camera_calibrate.py`（相机路径）时需要——
  `run_policy.py --mock` 和 `smoke_offline.py` 不需要它。

## 启动相机与策略

策略/手部脚本接收 `--gen {1,2}`（默认 2）
和 `--hand right`。相机观测程序（`vision`）与代际无关（无 `--gen`）。

先下载并解压[合并资源包](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip)。每代手使用各自目录中的 `policy.onnx` 与 `config.json`，保持两者在同一目录；不要混用。已按硬件搭建文档解压时可跳过下载。

```bash
gh release download v2026.9.27 --repo wuji-technology/wuji-mjlab --pattern 'wuji-mjlab-v2026.9.27-assets.zip'
unzip wuji-mjlab-v2026.9.27-assets.zip
```

### 1. 启动相机观测

```bash
pixi run -e reorient-deploy vision

# 使用自定义方块标签配置
pixi run -e reorient-deploy vision -- --cube path/to/cube_tags.json

```

### 2. 启动策略控制

```bash
pixi run -e reorient-deploy play-real -- --gen 2 --hand right \
  --ckpt wuji-mjlab-v2026.9.27-assets/reorient/checkpoints/hand2/policy.onnx
# 或者，无硬件：
pixi run -e reorient-deploy play-real -- --gen 2 --hand right --mock \
  --ckpt wuji-mjlab-v2026.9.27-assets/reorient/checkpoints/hand2/policy.onnx

```

### 3. 可选镜像窗口

```bash
pixi run -e reorient-deploy render -- --gen 2 --hand right

```

### 其他工具

```bash
# 让手平滑回到默认抓握姿态
pixi run -e reorient-deploy home -- --gen 2 --hand right

# 预检：依赖、SDK、策略 IO、（可选）手部连通性
# （--ckpt 必填——它校验你即将部署的那个 ONNX）
pixi run -e reorient-deploy check -- --gen 2 --hand right --ckpt wuji-mjlab-v2026.9.27-assets/reorient/checkpoints/hand2/policy.onnx
```

Wuji Hand 1 使用 `--connect` 时，除非显式加上 `--allow-energize`，否则会拒绝连接；
默认不会给电机上电。警告：加上 `--allow-energize` 后，检查期间电机会通电。

`tools/` 下的其他工具：`camera_calibrate.py` / `view_release_cube.py`
（相机内参标定和方块打印网格预览；两者都需要显示器，
`camera_calibrate.py` 另外还需要 MVS SDK 和一台已连接的相机）。

## 策略（不纳入 git 追踪）

`*.onnx` 权重和它们的 `*.onnx.config.json` 配套配置（ctrl_dt、action_scale、
ema_alpha、...）都**不纳入追踪**（整个仓库范围内被 gitignore）——导出器会把
配置写在 `.onnx` 旁边，因此它始终随权重一起走，无需在仓库里留存。checkpoint 导出方法见[导出 ONNX](train_zh.md#导出-onnx)。

`run_policy.py` 将默认 checkpoint 路径
解析为 `policies/policy_hand{gen}_{hand}.onnx`，并从它旁边读取 配套配置
（`<name>.onnx.config.json`，或同目录下的普通 `config.json`）。所以：要么把
导出的 `.onnx` + 配置放进 `deploy/reorient/policies/`，要么用 `--ckpt`
指向任意一个同时含这两者的目录——例如某次训练 运行的日志目录：

```bash
pixi run -e reorient-deploy play-real -- --gen 2 --hand right \
    --ckpt "logs/rsl_rl/wuji_reorient_hand2/<run>/policy_hand2_right.onnx"
```

合并资源包已包含 Wuji Hand 1 模型及身份元数据，无需再次写入身份。保持相机观测运行，选择 Wuji Hand 1 模型：

```bash
pixi run -e reorient-deploy play-real -- --gen 1 --hand right \
    --ckpt wuji-mjlab-v2026.9.27-assets/reorient/checkpoints/hand1/policy.onnx
```

硬件安装与标定见[硬件搭建](setup_zh.md)；Wuji Hand 1 端到端检查见下文[Wuji Hand 1 端到端验证](#wuji-hand-1-端到端验证)。

## Wuji Hand 1 端到端验证

完成硬件搭建和标定后，依次检查以下步骤。某一步失败时，先修复对应问题再继续。

### 1. 手部回到初始姿态

运行 `pixi run -e reorient-deploy home -- --gen 1 --hand right`，手部会平滑移动到默认姿态。终端先显示 `[home_check] gen=1 hand=right kp=5.0 kd=0.1 effort=0.5A — homing to default pose…`，到位后显示 `[home_check] holding at default pose. Ctrl+C to disable & exit.`。

程序保持该姿态直到按 Ctrl+C。确认 20 个关节与默认姿态的偏差在 ±2° 内；若手指卡顿或遇到硬限位，先停止运行，检查连接后重新尝试。

### 2. 启动方块观测

运行 `pixi run -e reorient-deploy vision`。应出现 OpenCV 预览窗口；保持腕部标签可见，黄色 `World Sampling: N/100` 进度条完成后变为绿色 `WORLD FIXED`。静止方块上的坐标轴叠加应保持稳定。

### 3. 检查 ZMQ 位姿数据流

保持 `vision` 运行，在另一个终端确认端口 **5555** 正在发布方块位姿。该端口由 `control.yaml::zmq.cube_port` 配置：

```bash
pixi run -e reorient-deploy python - <<'EOF'
import json, zmq
sock = zmq.Context().socket(zmq.SUB)
sock.connect("tcp://localhost:5555")
sock.subscribe(b"")
sock.setsockopt(zmq.RCVTIMEO, 5000)  # 5 秒未收到数据则超时
try:
    for _ in range(3):
        msg = json.loads(sock.recv_string())
        p = msg["cube1"]["position"]
        print(f"frame={msg['frame']:5d}  pos=({p['x']:+.3f},{p['y']:+.3f},{p['z']:+.3f})")
except zmq.Again:
    print("no pose received in 5s — is `vision` running and publishing on port 5555?")
EOF
```

应看到三个新的帧编号；方块静止时，输出的位置应稳定。

### 4. 检查方块位姿可视化

保持 `vision` 运行：

```bash
pixi run -e reorient-deploy python deploy/reorient/scripts/render_viewer.py --gen 1 --hand right
```

程序打开 MuJoCo 镜像窗口，订阅 ZMQ:5555 的方块位姿。如果 `play-real` 也在运行，还会订阅 ZMQ:5557 的实时关节状态；否则手保持默认姿态。移动实物方块，检查渲染出的方块是否同步移动。

重点检查以下问题：

- **坐标轴不一致**：绕某一轴旋转实物，渲染方块应绕同一轴旋转。镜像运动或相差 90° 通常表示 `cube_tags.json::face_rotations` 或标签粘贴方向有误。
- **位置偏移**：把方块放在掌心中央，渲染方块也应位于掌心。偏差超过 2 cm 时，检查搭建文档中的[手部安装](setup_zh.md#51-手部安装)与[相机内参](setup_zh.md#6-相机内参标定)。
- **延迟或抖动**：对照搭建文档的[位姿观测排查](setup_zh.md#7-位姿观测排查)检查滤波参数。

按 Ctrl+C 或关闭窗口退出。

### 5. 运行闭环策略

资源包内的 Wuji Hand 1 模型已经包含身份元数据，直接使用配套文件运行：

```bash
pixi run -e reorient-deploy play-real -- --gen 1 --hand right \
    --ckpt wuji-mjlab-v2026.9.27-assets/reorient/checkpoints/hand1/policy.onnx
```

程序加载 ONNX 策略，打印配套配置，并通过 `driver.home()` 让手回到初始姿态。连接真机且未使用 `--mock` 时，在提示处按回车后才开始策略控制。

镜像窗口由[方块位姿可视化检查](#4-检查方块位姿可视化)中的 `render_viewer.py` 独立运行，`play-real` 本身不打开观察窗口。控制期间每 5 秒打印一次 `[run_policy] cube↔goal err=…° cube_pos_tag=… cube_msgs=… reached=N timedout=N`。驱动能够返回温度时，每 10 秒还会打印 `[run_policy] temp: …`。

若策略一开始就出现明显偏差，按[故障排查](#故障排查)检查。

### 故障排查

| 现象 | 可能原因 | 处理方法 |
|---|---|---|
| 相机无法打开 | 未安装 MVS SDK，或未配置 `MVS_PYTHON_PATH` | 按搭建文档的 [SDK 安装说明](setup_zh.md#22-hikrobot-mvs-sdk)重新检查安装和导入 |
| 始终检测不到腕部标签 | 光照、标签系列、ID 或尺寸不正确 | 确认 AprilTag36h11、ID 0、检测尺寸 50.4 mm，并改善光照 |
| `World Sampling` 采样无法完成 | 腕部标签过小或模糊 | 调整位置，使标签宽度至少约 80 像素，并重新对焦 |
| 频繁丢失方块 | 重投影误差超过阈值 | 重新[标定内参](setup_zh.md#6-相机内参标定)，并按[方块位姿可视化检查](#4-检查方块位姿可视化)检查标签面映射 |
| 策略第一步就明显偏离 | 标签方向与配置不一致 | 修正 `face_rotations` 或相应标签的粘贴方向 |
| 运行时手部抖动 | 控制参数不匹配 | 分别检查 ONNX 配套配置的 `ctrl_dt` 和 `control.yaml::control.servo_hz`。Wuji Hand 1 的低通截止频率由 `hand1.py` 的 `lowpass_cutoff` 定义，默认 3.0 Hz，无 YAML 或命令行设置；Wuji Hand 2 使用 `wuji_sdk` 的 MIT 阻抗控制，没有该低通设置。 |
| 镜像窗口中的手保持默认姿态，显示 `joints=NO` | 没有进程发布关节状态 | 启动 `play-real`，检查 `control.yaml::zmq.joint_port`，默认端口为 5557 |
