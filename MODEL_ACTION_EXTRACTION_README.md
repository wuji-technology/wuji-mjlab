# Revo3 模型动作提取规范

本文档用于指导从模型或仿真环境中导出 Revo3 手部动作，使导出的 `.target.txt`
可以被 `replay.py` 直接回放到 `joint_forward_mit_controller`。

重点：

- `replay.py` 不加载 `policy.onnx`，也不做在线推理。
- `replay.py` 只读取已经导出的 `.target.txt` 轨迹，再结合
  `revo3_profile.yaml` 转成 `Revo3MITCommand`。
- 推荐导出仿真空间的绝对目标关节角 `cur_targets`，并用 `--mode absolute`
  回放。

## 1. 数据流

```text
模型 / 仿真环境
  -> 导出 .target.txt
  -> replay.py 读取轨迹和 revo3_profile.yaml
  -> 重排到 controller_joint_order
  -> clip 到 joint_limits
  -> 加 sim2real_joint_offset
  -> 发布 Revo3MITCommand
```

`replay.py` 发布的目标位置为：

```text
real_command = clipped_sim_target + sim2real_joint_offset
```

其中 `sim2real_joint_offset` 和 `joint_limits` 来自
`config/robot_profile/revo3_profile.yaml`。

## 2. 必需文件

### 2.1 轨迹文件

模型动作提取脚本需要输出一个 `.target.txt` 文件，例如：

```text
config/ep00_cylinder_ppo_0516_192310.target.txt
```

如果运行 `replay.py` 时不显式传入轨迹路径，默认读取：

```text
config/ep00_cylinder_ppo_0516_192310.target.txt
```

### 2.2 Robot Profile

`replay.py` 默认读取：

```text
config/robot_profile/revo3_profile.yaml
```

该文件提供：

- `controller_joint_order`
- `joint_limits`
- `sim2real_joint_offset`
- `action_scale`
- `command_topic`

### 2.3 Policy Metadata

`config/onnx/policy.yaml` 可作为模型导出元信息参考，例如关节顺序、策略频率、
动作语义等。

注意：`replay.py` 当前主版本不读取 `policy.yaml`，它只读取 `.target.txt`
和 `revo3_profile.yaml`。

## 3. .target.txt 格式

`.target.txt` 是纯文本文件，包含两部分：

1. 以 `#` 开头的元信息 header。
2. 每一帧的 `target=[...]` 数据。

### 3.1 必需 Header

建议每个 `.target.txt` 至少包含以下字段：

| 字段 | 示例 | 说明 |
|------|------|------|
| `policy_dt_sec` | `# policy_dt_sec=0.050000` | 策略基础周期，20 Hz 时为 0.05 秒 |
| `policy_hz` | `# policy_hz=20.000000` | 策略基础频率 |
| `action_semantics` | `# action_semantics=delta` | 模型原始 action 的语义 |
| `action_formula` | `# action_formula=target=prev_target+(1/24)*raw_action then clamp(joint_limits)` | 训练/仿真中 action 到 target 的公式 |
| `joint_order` | `# joint_order=0:right_index_MPR_joint, ...` | 轨迹中 21 个数值的关节顺序 |

`joint_order` 的每一项必须是 `序号:关节名` 格式，并且必须包含 21 个互不重复的
关节名。`replay.py` 会根据这个顺序把轨迹重排到 `controller_joint_order`。

### 3.2 推荐 Header

为了便于追溯模型版本和导出含义，建议同时写入：

| 字段 | 示例 | 说明 |
|------|------|------|
| `task` | `# task=cylinder` | 任务名称 |
| `algo` | `# algo=PPO` | 训练算法 |
| `checkpoint` | `# checkpoint=/path/to/best.pth` | checkpoint 来源 |
| `raw_action_definition` | `# raw_action_definition=policy clamped delta output mu (pre delta-integration, [-1,1])` | 原始 action 的定义 |
| `target_definition` | `# target_definition=cur_targets (delta-accumulated + joint-limit clamped, used in PD formula)` | target 的定义 |
| `jointpos_definition` | `# jointpos_definition=hand.data.joint_pos (absolute joint angles, rad)` | 关节状态定义 |
| `init_joint_pos` | `# init_joint_pos=right_index_MPR_joint=-0.235620, ...` | 仿真起始关节角 |

### 3.3 帧数据格式

每一帧必须包含 21 个浮点数：

```text
frame=000 t= 0.000s reward=+2.294080 done=0 target=[-0.209690, +0.193953, ...]
frame=001 t= 0.050s reward=+2.892635 done=0 target=[-0.232908, +0.184322, ...]
```

解析要求：

- 行首必须包含 `frame=<整数>`。
- 必须包含 `t=<秒数>s`。
- 必须包含 `target=[...]`。
- `target` 内必须正好有 21 个浮点数。
- 浮点数可以用逗号、空格或逗号加空格分隔。
- `t` 必须严格递增。
- 相邻帧时间间隔必须一致；如果 header 中有 `policy_dt_sec`，则每帧间隔应与它一致。
- `replay.py` 当前允许的最大时间间隔误差为 `1e-4` 秒。

20 Hz 策略的标准时间戳通常是：

```text
0.000, 0.050, 0.100, 0.150, ...
```

## 4. 推荐导出语义

### 4.1 推荐：导出绝对目标关节角

推荐导出训练/仿真中已经积分、限幅后的目标关节角：

```text
target = cur_targets
```

这种文件应使用：

```bash
ros2 run revo3_rl_deploy replay.py \
  <your_trajectory.target.txt> \
  --hand-side right \
  --mode absolute
```

在 `absolute` 模式下，`replay.py` 将每帧 `target=[...]` 视为仿真空间绝对关节角，
然后执行：

```text
target_scaled = first_frame + trajectory_scale * (target - first_frame)
target_clipped = clip(target_scaled, joint_limits)
real_command = target_clipped + sim2real_joint_offset
```

### 4.2 可选：导出原始 Delta Action

如果导出的是模型原始 action，而不是已经积分后的 `cur_targets`，则只能用：

```bash
ros2 run revo3_rl_deploy replay.py \
  <your_actions.target.txt> \
  --hand-side right \
  --mode delta
```

在 `delta` 模式下，`target=[...]` 会被解释为 raw action，并按以下公式积分：

```text
action_clipped = clip(action, -1.0, 1.0)
target_next = target_prev + trajectory_scale * action_scale * action_clipped
target_next = clip(target_next, joint_limits)
real_command = target_next + sim2real_joint_offset
```

`action_scale` 来自 `revo3_profile.yaml`。如果模型训练时使用的是 `1/24`，
但 profile 中配置为其他值，回放动作幅度会不同。

## 5. 当前已验证回放参数

当前用于圆柱抓取效果较好的 replay 参数为：

```bash
ros2 run revo3_rl_deploy replay.py \
  --hand-side right \
  --mode absolute \
  --rate-scale 0.5 \
  --trajectory-scale 0.3 \
  --kp 1.0 \
  --kd 0.5
```

含义：

- `--mode absolute`：轨迹中的 `target` 是仿真空间绝对目标关节角。
- `--rate-scale 0.5`：以原轨迹 0.5 倍速度播放；20 Hz 轨迹实际约 10 Hz 回放。
- `--trajectory-scale 0.3`：围绕第一帧把动作幅度缩小到 30%。
- `--kp 1.0`：MIT 控制器位置增益。
- `--kd 0.5`：MIT 控制器阻尼增益。

## 6. 最小可用文件示例

```text
# task=cylinder
# algo=PPO
# checkpoint=/path/to/checkpoint.pth
# policy_dt_sec=0.050000
# policy_hz=20.000000
# action_semantics=delta
# action_formula=target=prev_target+(1/24)*raw_action then clamp(joint_limits)
# raw_action_definition=policy clamped delta output mu (pre delta-integration, [-1,1])
# target_definition=cur_targets (delta-accumulated + joint-limit clamped, used in PD formula)
# jointpos_definition=hand.data.joint_pos (absolute joint angles, rad)
# joint_order=0:right_index_MPR_joint, 1:right_little_MPR_joint, 2:right_middle_MPR_joint, 3:right_ring_MPR_joint, 4:right_thumb_CMP_joint, 5:right_index_MCP_joint, 6:right_little_MCP_joint, 7:right_middle_MCP_joint, 8:right_ring_MCP_joint, 9:right_thumb_CMR_joint, 10:right_index_PIP_joint, 11:right_little_PIP_joint, 12:right_middle_PIP_joint, 13:right_ring_PIP_joint, 14:right_thumb_MCP_joint, 15:right_index_DIP_joint, 16:right_little_DIP_joint, 17:right_middle_DIP_joint, 18:right_ring_DIP_joint, 19:right_thumb_PIP_joint, 20:right_thumb_DIP_joint

frame=000 t= 0.000s target=[-0.209690, +0.193953, +0.042161, +0.182710, +1.724058, +1.198168, +1.248383, +1.020873, +0.980713, +1.351662, +0.342860, +0.343299, +0.283374, +0.209623, +0.395100, +0.024630, +0.039816, +0.096571, +0.172623, +0.306406, +0.051092]
frame=001 t= 0.050s target=[-0.232908, +0.184322, +0.083828, +0.181677, +1.695497, +1.228145, +1.206717, +0.983606, +0.996325, +1.373837, +0.365335, +0.301633, +0.241707, +0.198014, +0.436767, +0.000000, +0.064259, +0.138238, +0.214290, +0.264739, +0.051247]
```

## 7. 提取脚本检查清单

导出新轨迹前，确认：

- 每帧 `target` 都是 21 维。
- `joint_order` 与 `target` 数值顺序完全一致。
- `joint_order` 中的关节名能在 `revo3_profile.yaml` 的 `controller_joint_order`
  和 `joint_limits` 中找到。
- 时间戳严格递增，且间隔固定。
- 若使用 `--mode absolute`，`target` 是仿真空间绝对目标关节角。
- 若使用 `--mode delta`，`target` 是 raw action，且 `action_scale` 与训练设置一致。
- 所有角度单位为 rad。
- 不要在 `.target.txt` 中写入已经加过 `sim2real_joint_offset` 的真机命令；
  offset 由 `replay.py` 统一添加。
