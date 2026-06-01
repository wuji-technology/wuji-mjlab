# Reorient Real-Hand Deploy

[English version](README.md)

`WujiHand_Reorient` 的 sim2real 桥：基于 ArUco（cube 面）+ AprilTag（手腕）的方块追踪 + 物理 Wuji Hand 上的闭环策略控制。

## 流水线

```
+----------------+        +------------------+        +------------------+
| Camera + tags  | -ZMQ-> | cube observer    | -ZMQ-> | play_real        |
| (aruco+apriltag)|       | (cube_world_     |        | (policy + control |
|                |        |  observer.py)    |        |  + viewer)        |
+----------------+        +------------------+        +------------------+
                                                              |
                                                              v
                                                       +--------------+
                                                       | WujiHandDriver
                                                       | (real hardware)
                                                       +--------------+
```

## 快速开始

```bash
# 1. Install deploy environment (adds opencv, apriltag, zmq, hand driver)
pixi install -e deploy

# 2. Calibrate your physical cube and camera
#    运行时 cube 几何由 deploy/reorient/config/cube_tags.json 加载
#    （默认 54 mm 基线）。如需使用自定义 cube_tags.json 文件，可向
#    vision 任务传入 `--cube <path>`（`pixi run -e deploy vision -- --cube <path>`）。
#    （deploy/reorient/config/cube_calibration.yaml 只是供人阅读的参考装置
#    记录，运行时**不会**被读取。）
#    See deploy/reorient/tools/camera_calibrate.py to calibrate the camera.

# 3. Home the hand
pixi run -e deploy home

# 4. Launch the vision pipeline (one terminal)
pixi run -e deploy vision

# 5. Sanity-check the calibration: opens a mirror viewer where the rendered
#    cube tracks the physical cube while the hand sits at home. Use to spot
#    axis swaps or position offsets before running the policy.
pixi run -e deploy python deploy/reorient/tools/calib_check.py

# 6. Run sim2real control with mirror viewer (another terminal)
pixi run -e deploy play-real --ckpt <path-to-policy.onnx>
```

> **没有训好的策略？** 在
> [Releases](https://github.com/wuji-technology/wuji-mjlab/releases)
> 下载预训练的 `policy.onnx` + `policy_config.json` 即可直接 deploy，
> 不需要先训练。

## 目录结构

| 路径 | 作用 |
|---|---|
| `config/` | 相机、方块标定、控制循环的 YAML 配置 |
| `lib/` | 真机 env 子类、ZMQ 桥、基于 ArUco/AprilTag 的 observer、hand driver |
| `scripts/` | 入口脚本（`play_real.py`、`cube_world_observer.py` 等） |
| `tools/` | 相机标定 + cube 位姿 sanity check + release cube viewer |

## 架构

`RealHandEnv`（`lib/real_hand_env.py`）继承自 mjlab 的 `ManagerBasedRlEnv`，因此训练时的同一套 observation/action manager 可直接复用——不存在重复的 obs/action 流水线。真机硬件 ↔ sim 之间的桥接发生在 env 的 `step()` 内部：动作下发到 `WujiHandDriver`（真实电机），观测来自关节状态（通过 driver）和方块 pose（通过 ZMQ 从 `cube_world_observer.py` 接收）的组合。

ONNX 策略通过 `lib/onnx_policy.py` 加载，它会读取导出时记录的 sidecar JSON 以获取控制模式参数（action_scale、ema_alpha、ctrl_dt、history_len）。这样部署推理与产出该 ONNX 的 sim 策略保持完全一致。

## 许可协议

Apache 2.0。详见仓库根目录的 LICENSE。
