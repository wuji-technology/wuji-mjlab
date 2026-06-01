# Sim-to-real 部署 — Wuji Hand Reorient

[English version](setup.md)

本指南介绍在物理 Wuji Hand 上运行训练好的 `WujiHand_Reorient` ONNX 策略所需的**硬件侧**配置。关于**软件管线**（ZMQ 拓扑、observer 架构、ONNX 策略加载），参见 [`deploy/reorient/README_zh.md`](../../deploy/reorient/README_zh.md)。

完成本指南后，你将获得：
- 一个经标定的相机，以已知内参观察工作区
- 一个 3D 打印的、带 ArUco 标签的标定 cube
- 一个手腕侧 AprilTag，用于定义世界（手腕）坐标系
- 安装在治具上、位于相机视野内的 Wuji Hand
- 针对**你的**装置填好的 `camera.yaml` + `cube_tags.json`
- 可正常运行的 `pixi run -e deploy vision` + `play-real` 管线

<p align="center">
  <img src="../assets/deploy.gif" width="80%" alt="参考装置：本指南将复现上图所示的部署形态" />
</p>

## 1. 硬件清单

以下是参考装置的参考硬件清单。只要第 3 节和第 5 节的几何校验通过，等效部件（不同厂商、规格相近）均可使用。

> 视觉硬件（相机、镜头、支架）**仅作为参考配置**提供。功能层面的要求是：在完成第 6 节的内参标定后，所选光学组合能够在 reorient 可达工作区内可靠估计 cube 相对手腕 AprilTag 的位姿。任何能达到该标准的传感器 + 镜头组合都可等效替代；请把上面列出的 Hikrobot 部件视为"已知可用"的起点，而不是硬性要求。

- **工业 USB 相机**：**Hikrobot MV-CU013-A0UC**（USB-3，1280×1024，1.3 MP，彩色 Bayer GB）— 与 [`deploy/reorient/config/camera.yaml`](../../deploy/reorient/config/camera.yaml) 中编码的传感器和采集格式匹配。
  分辨率和 Bayer GB 模式匹配的等效 USB-3 工业相机（FLIR/Basler/Allied Vision 对应型号）可作为直接替代品；需相应更新 `camera.yaml`；相机 SDK 替换（替换 `MvImport`）并非易事。
  （目前 observer 中只接入了 Hikrobot 这一个传感器 SDK —— 安装方式见 §2.2。UVC 摄像头和其他厂商的工业相机需要重写 cube observer，把 `MvImport` 替换为对应厂商的 Python 绑定。）
  > ⚠️ **注意**：本说明覆盖上文的"直接替代"措辞 —— observer 硬编码 import 了
  > `MvImport.MvCameraControl_class`，因此任何非 Hikrobot 传感器都需要先在
  > [`deploy/reorient/scripts/cube_world_observer.py`](../../deploy/reorient/scripts/cube_world_observer.py)
  > 中替换为新厂商的 Python 绑定，上文列出的"直接替代"方案才真正成立。
- **FA 镜头**：**Hikrobot MVL-MF0824M-5MPE** — 8 mm 定焦，F2.4，2/3″ 像圈，C-mount，5 MP 适配。与 CU013 传感器搭配良好（较大的 2/3″ 像圈足以覆盖且无暗角）。任何 2/3″ 像圈的 C-mount 镜头，8 mm 焦距，F2.4 或更大光圈均可等效替代。
- **相机安装支架 / 三脚架**：任何刚性夹具，能将相机固定在距手掌约 350 mm 上方，且在标定（第 6 节）和 rollout 之间不发生位移。要求：1/4"-20 标准三脚架螺纹或等效 C-mount 支架；垂直行程 ≥ 400 mm；具备振动阻尼（USB 线张力下不变形）；优先选用固定高度的夹具而非伺服机械臂。
- **手腕 AprilTag 贴纸**：1 张 AprilTag36h11 ID 0，外尺寸 48 mm × 48 mm（匹配 [`deploy/reorient/scripts/cube_world_observer.py`](../../deploy/reorient/scripts/cube_world_observer.py) 中硬编码的 `WORLD_TAG_SIZE = 0.048`；该常量不通过 yaml 暴露，如需修改请直接改脚本）。打印在哑光 vinyl 或覆膜纸上以避免相机眩光；白底黑墨；安装前用卡尺校验打印外尺寸 — 任何缩放误差都会直接传播到位姿估计。打印 / 购买的具体流程以及尺寸约定见第 4 节。
  （cube 的 24 块 ArUco 贴片**不是**贴纸 — 它们通过双材料打印直接成形于随包发布的 Bambu Lab `.3mf` 文件中；参见 3.1 节。）
- **Wuji Hand 右手**。请直接联系 Wuji Technology；`wujihandpy==1.5.1` 要求与 `lib/hand_driver.py` 匹配的 Wuji Hand 固件版本。Host 端通过一根 USB 线直连（hand 内部 STM32 暴露 USB CDC 接口，vendor ID 0483）。
- **手部安装治具** — 3D 打印的 PLA 底座，用螺丝固定在铝合金蜂窝光学平板上。详细 BOM 和装配步骤见 5.1 节；CAD 文件随 release 附件提供（见 [Releases](https://github.com/wuji-technology/wuji-mjlab/releases)）。
- **标定 cube** — 3D 打印的 54 mm 棱长实心立方体，6 个面已嵌入 24 块
  ArUco 标签（与 `cube_tags.json` 匹配）。打印细节见第 3 节；CAD 文件
  随 release 附件提供（见
  [Releases](https://github.com/wuji-technology/wuji-mjlab/releases)）。
- **计算机**：Ubuntu 22.04 x86_64，NVIDIA sm_80+ GPU（Ampere+），CUDA 12.8，至少 2 个空闲 USB 接口（一个连相机，一个连 Hand）。

> 所有 Wuji 出品的部件（cube、治具）均在 Apache 2.0 协议下开源；只要 cube 棱长 = 54 mm 且标签尺寸匹配 [`deploy/reorient/config/cube_tags.json`](../../deploy/reorient/config/cube_tags.json)，商业替代品也可使用。

## 2. 软件前置依赖

### 2.1 操作系统和 GPU 驱动

- Ubuntu 22.04 LTS，x86_64。
- 与 CUDA 12.8 配套的 NVIDIA 驱动（`nvidia-smi` 应能正常报告）。
  较旧的驱动可能仅支持推理，但未经测试。
- [pixi](https://pixi.sh) ≥ 0.66（CI 用的版本），已加入 `$PATH`。

### 2.2 Hikvision MVS SDK

`tools/camera_calibrate.py` 和 `scripts/cube_world_observer.py` 会从系统级 SDK 安装路径导入 `MvImport.MvCameraControl_class`（本仓库未内置）。

**获取方式**：<https://www.hikrobotics.com> → Service & Support → Downloads → MVS Client → Linux x86_64。（如果按地区跳转到中文页面，请切换到右上角的英文版本。）

**推荐版本**：Linux 版 MVS Client **≥ 4.6.0**。参考装置运行的是 `4.6.3`（捆绑了 Machine Vision Camera SDK `4.7.1.1`）；较旧的 4.5.x 版本提供的 Python 绑定略有不同，可能导致导入失败。

**安装方法**。具体包格式因平台和 MVS 版本而异 — **请始终以 SDK 压缩包内附带的 README 为准**，因为官方命令在不同 MVS Client 小版本和不同发行版之间会有变化。常见情况：

```bash
# Ubuntu / Debian (recommended; the .deb is what hikrobotics.com offers today)
sudo apt install ./MVS-*.deb

# Or equivalently
sudo dpkg -i MVS-*.deb

# CentOS / RHEL
sudo rpm -i MVS-*.rpm

# Legacy tarball (only older releases)
tar -xf MVS-*.tar.gz && cd MVS-* && sudo ./setup.sh
```

上述所有方式默认都会将文件安装到 `/opt/MVS/` 下。安装完成后，你应当能看到：

- `/opt/MVS/lib/64/libMvCameraControl.so` — 运行时共享库
- `/opt/MVS/Samples/64/Python/MvImport/` — Python 绑定
- `/opt/MVS/bin/MVS` — 用于设备发现和实时预览的 GUI

**安装后的系统调优**（要稳定支撑 `camera.yaml` 中配置的 90 FPS 采集，必须执行）：

```bash
# USB-3 cameras: install udev rules + raise USB scheduling priority
sudo /opt/MVS/bin/set_usb_priority.sh

# GigE cameras only: raise kernel socket buffer to prevent frame drops
sudo /opt/MVS/bin/set_socket_buffer_size.sh
```

**Shell 环境变量**。MVS 安装包会把 export 语句写入 `/etc/profile.d/MVS_*.sh`，但该文件**只在登录 shell 启动时**才会加载。zsh 和大多数终端启动的 bash 都是非登录 shell，因此变量会静默缺失，导入 Python 绑定时会抛出 `TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'`。请将 export 追加到你的 shell rc 文件中，使其持久生效：

```bash
# zsh
echo 'export MVCAM_COMMON_RUNENV=/opt/MVS/lib' >> ~/.zshrc
echo 'export LD_LIBRARY_PATH=/opt/MVS/lib/64:$LD_LIBRARY_PATH' >> ~/.zshrc
source ~/.zshrc

# bash
echo 'export MVCAM_COMMON_RUNENV=/opt/MVS/lib' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/MVS/lib/64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

| 变量 | 作用 |
|---|---|
| `MVCAM_COMMON_RUNENV` | Hikvision Python 绑定通过它定位 `libMvCameraControl.so`。 |
| `LD_LIBRARY_PATH` | Linux 动态链接器搜索路径；需要它来解析 MVS 共享库的传递依赖。 |

确认：

```bash
echo $MVCAM_COMMON_RUNENV          # /opt/MVS/lib
echo $LD_LIBRARY_PATH | tr ':' '\n' | grep MVS   # contains /opt/MVS/lib/64
```

**验证安装**：

```bash
# Python binding import test
python3 -c "import sys; sys.path.insert(0, '/opt/MVS/Samples/64/Python'); from MvImport.MvCameraControl_class import *; print('ok')"

# Hardware detection — your camera should appear in the left panel
/opt/MVS/bin/MVS
```

排错：

- `ModuleNotFoundError: MvImport` — SDK 路径错误；要么重新安装到 `/opt/MVS/`，要么设置 `MVS_PYTHON_PATH=/path/to/MVS/Samples/64/Python`。
- 导入 `MvCameraControl_class` 时出现 `TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'` — `MVCAM_COMMON_RUNENV` 未设置；参见上文"Shell 环境变量"小节。
- GUI 列表中没有相机 — 重新运行 `set_usb_priority.sh`，检查线缆，把你的用户加入 `plugdev`/`dialout` 组，并通过 `lsusb | grep -i hikvis`（USB-3）或 `arp -a | grep -i hikvis`（GigE）确认硬件是否可见。

### 2.3 Deploy 环境

从仓库根目录执行 `pixi install -e deploy` 会拉取（参见 [`pixi.toml`](../../pixi.toml) `[feature.deploy.pypi-dependencies]`）：`opencv-contrib-python>=4.13`（ArUco + IPPE）、`pupil-apriltags>=1.0`（手腕标签）、`pyzmq>=27.0`（cube/goal pub-sub）、`glfw>=2.10`（passive MuJoCo viewer）、`wujihandpy==1.5.1`（Wuji Hand 驱动）、`pyyaml>=6.0`。冒烟测试：

```bash
pixi run -e deploy python -c "import cv2, pupil_apriltags, zmq, wujihandpy; print(cv2.__version__)"
```

## 3. Cube 制作

最快路径是用 release 附件里打包的资产复现参考 cube. 先拉取 release zip:

```bash
# 需要 GitHub CLI (https://cli.github.com)。glob 通配让该命令在
# 未来的 release tag 下也无需修改即可继续工作。
gh release download --repo wuji-technology/wuji-mjlab --pattern '*-assets.zip'
unzip wuji-mjlab-*-assets.zip
mv wuji-mjlab-*-assets release-assets
```

这会产生一个 `release-assets/` 目录，其中包含
`hardware/cube/{cube.3mf, cube.step, cube.obj, cube.mtl, cube.png}` 以及
`checkpoints/`、`hand-jig/` 等子目录（该目录及源 zip 都在 gitignore 中）。
Cube 本身是一个棱长 54 mm 的实心立方体，每面贴有 4 × 13 mm 的 ArUco
标签（共 24 个，ID 为 0–23）。真实性能依赖于尺寸与 `cube_tags.json`
之间的偏差控制在约 0.5 mm 以内。

> 若没有 `gh` CLI，可手动浏览到
> [latest release 页面](https://github.com/wuji-technology/wuji-mjlab/releases/latest)
> 下载 `wuji-mjlab-v*-assets.zip` 附件，然后在当前目录执行同样的
> `unzip` + `mv` 命令对。

打印前若要 3D 检视 24 个标签的布局，用随仓库提供的 viewer 脚本（默认读
cwd 下的 `release-assets/hardware/cube/cube.obj`）:

```bash
pixi run python deploy/reorient/tools/view_release_cube.py
```

### 3.1 3D 打印 cube

**请使用随包发布的 Bambu Lab `.3mf`**（release 附件中的 `release-assets/hardware/cube/cube.3mf`）— 它包含 cube 几何以及双材料分配，能直接把 24 个 ArUco 4×4 标签图案打印到各面上。无需贴纸、胶水或对齐操作。

工作流：

1. 在 **Bambu Studio** 中打开该文件，或拖到已连接的 Bambu 打印机（Bambu 打印机固件可以直接对 `.3mf` 切片）。
2. 装载**两种耗材**：一种**黑色**，一种**白色**（PLA 即可）。切片软件会提示分配 AMS / 外部料盘槽位 — `.3mf` 已经声明了哪个逻辑槽是"tag"哪个是"base"，请确认实际耗材对应正确。
3. 切片并打印。默认设置（约 0.2 mm 层高，约 30% 填充率）即可；策略不依赖 cube 的内部密度。

预期几何（与 [`cube_tags.json`](../../deploy/reorient/config/cube_tags.json) 匹配，**请勿**缩放）：**cube 棱长 54 mm**，标签贴片 13 mm，标签中心沿每个面的本地 u/v 轴各偏移 18 mm。

> 多材料打印明显比单材料更容易失败 — 在调校 AMS 冲料量和换色清料的过程中，前几次打印很可能要作为废件。

标签 ID 遵循 [`deploy/reorient/config/cube_tags.json`](../../deploy/reorient/config/cube_tags.json) 中的布局：TOP 为 ID 0–3，BOTTOM 为 8–11，FRONT 为 20–23，BACK 为 16–19，LEFT 为 12–15，RIGHT 为 4–7（每面 4 个标签，每个约 13 mm）。

**单材料 fallback。** 若你没有双材料 3D 打印机，也可以基于 `cube.step`（或 `cube.obj`）加上贴纸打印来得到等效 cube：

1. 用任意单一材料（PLA/PETG 等）从 `cube.step` 打印 cube 主体。打印后用卡尺核对 54 mm 边长。
2. 把 `cube.png` 按 UV 展开尺寸 1:1 印在哑光 vinyl 或覆膜纸上。该纹理把 24 个标签布局成一张 6 面展开图；沿面边界裁剪即可得到 6 张 54 × 54 mm 贴纸。
3. 把每张贴纸贴到对应的面上。Cube 主体本身无固定朝向，但标签 ID 编码了所属面 — 贴纸时请按照上面的 `cube_tags.json` 映射（例如 0–3 号标签贴到 TOP 面）。
4. 若贴纸没法刚好落在距面中心 18 mm 处，请在 `cube_tags.json` 中重新标定 `tag_center_offset`（默认 18 mm 是按展开网格估算的）。

在 ArUco 检测层面，贴纸 cube 与双材料 cube 功能等价，但更容易掉皮和对齐偏差。如果条件允许，仍优先选用 `.3mf`。

> **关于 cube 配色的说明。** `cube.png` 和 `cube.3mf` 中 6 面的彩色底色（蓝/绿/红/紫/青/黑）纯粹是为了肉眼区分各个面 — AprilTag/ArUco 检测器只读取黑白标签 pattern，并且 observer 会施加 BGR 取最小通道的灰度变换（见 5.3 节），任何彩色背景都会被压到接近 0。如果你走贴纸路线，可以放心选用纯白底色，只要标签 pattern 本身保持白底黑字的高对比度即可。

### 3.2 标签规格

来自 `cube_tags.json`（这些参数仅用于视觉定位；只有当你打印的 cube 几何不同才需要改，与是否重新训练策略无关）：字典 **ArUco 4×4**（`cv2.aruco.DICT_4X4_50`）；`tag_size: 0.013`（13 mm）；`tag_center_offset: 0.018`（距面中心 18 mm）；`face_rotations` TOP 0° / BOTTOM 180° / FRONT/BACK/LEFT 90° / RIGHT 270°。

每个面的标签 ID（T/R/B/L = 该面内的 top/right/bottom/left 槽位，参见 `cube_tags.json::faces_config`）：

| 面     | T  | R  | B  | L  |
|---------|----|----|----|----|
| TOP     |  0 |  2 |  3 |  1 |
| BOTTOM  | 11 |  9 |  8 | 10 |
| FRONT   | 22 | 23 | 21 | 20 |
| BACK    | 18 | 19 | 17 | 16 |
| LEFT    | 14 | 15 | 13 | 12 |
| RIGHT   |  5 |  4 |  6 |  7 |

### 3.3 每个面的轴向（参考）

标签已预先嵌入 `.3mf`，无需粘贴。下表是 cube 坐标系的轴向约定，随包 `.3mf` 与 `cube_world_observer.py::_build_aruco_board` 中的 ArUco 板构造逻辑都使用该约定 — 如果你从头重新生成 `.3mf`（例如改用其他棱长或标签字典），请保持同步。

如需视觉参考，运行 `pixi run python deploy/reorient/tools/view_release_cube.py`
（见 §3 开头说明）；上方的逐面 ID 表已用文字给出等价信息。

各面的本地轴向（来自 `cube_tags.json::face_axes`）：

| 面     | 中心（cube 坐标系）  | u 轴       | v 轴        |
|---------|---------------------|-----------|------------|
| TOP     | `[ 0,  0,  1]`      | `[1,0,0]` | `[0,1,0]`  |
| BOTTOM  | `[ 0,  0, -1]`      | `[1,0,0]` | `[0,-1,0]` |
| FRONT   | `[ 0, -1,  0]`      | `[1,0,0]` | `[0,0,1]`  |
| BACK    | `[ 0,  1,  0]`      | `[-1,0,0]`| `[0,0,1]`  |
| LEFT    | `[-1,  0,  0]`      | `[0,-1,0]`| `[0,0,1]`  |
| RIGHT   | `[ 1,  0,  0]`      | `[0,1,0]` | `[0,0,1]`  |

每个面上的 4 个标签沿 u 轴 ±`tag_center_offset`（L/R）和 v 轴 ±`tag_center_offset`（T/B）分布。如果你重新生成 `.3mf`，请确保每个贴片的"上方"与该面的 v 轴对齐 — IPPE 能吸收小幅度的面内旋转，但贴片若整体旋转 90°，会让 `detect_cube_pose` 中的主面判定逻辑直接拒绝整面。

## 4. 手腕 AprilTag

Cube observer 通过一个刚性安装在手腕板上的 AprilTag36h11 标签来定义世界（手腕）坐标系（`cube_world_observer.py`："World frame defined by AprilTag ID 0"）。强制规格（硬编码）：family **AprilTag36h11**，ID **0**，棱长 **48 mm**（`WORLD_TAG_SIZE = 0.048`）。请精确打印为外尺寸 48 mm × 48 mm（AprilTag 库以此为度量单位；打印缩放误差会传播到位姿估计）。

标签平面位于手腕背面，与手掌法线垂直。Observer 在 `WORLD_FRAME_CORRECTION` 中硬编码了一个纯手性翻转，因此打印的治具必须以与参考装置相同的朝向放置标签。
![手腕标签安装位置 — AprilTag 位于组装治具顶部](images/hand-jig-side.jpg)

> **警告。** 完成世界坐标系采样后请勿移动手腕标签。Observer 在启动时会对 100 帧做平均，然后冻结世界位姿（`_finalize_world_frame`）；之后任何位移都会使策略读取的 cube-in-tag 观测失真。

### 4.1 购买或自行打印

发布版的硬件包**不**附带预先裁切好的手腕贴纸。优先推荐自行打印，因为成品贴纸供应商极少提供"单一 ID、自定义尺寸"的选项 — 市售 AprilTag 贴纸包通常按整套 ID 列表出货且尺寸固定，而本任务只需要 ID 0 且必须严格为 48 mm。如果你坚持购买成品，可以在 Amazon / AliExpress / Taobao 上搜索"AprilTag36h11 sticker 48mm"，下单前请向卖家核实其标注的"外尺寸"与下面的尺寸约定一致。

**尺寸约定。** AprilTag36h11 是一个 10 × 10 cell 的栅格（8 × 8 数据格 + 四边各一格黑色边框）。48 mm 指的是**黑色边框的外缘**，即两条对边外缘之间的距离，也就是 `pupil-apriltags` 库识别的 `tag_size`。白色**安静区**（quiet zone，宽度 ≥ 1 cell ≈ 4.8 mm）位于 48 mm 之**外**，它不计入度量尺寸，但 detector 必须依靠它来定位边缘梯度。因此一张正确打印的贴纸总体大约为 58 mm × 58 mm（中间 48 mm 黑色 + 四周各 ≥ 5 mm 白色留白）。

按 48 mm 外缘换算，每个 cell 边长 4.8 mm，在 600 dpi 下约对应 113 px，足以保证检测稳定。

**DIY 打印工作流：**

1. **获取标签图像。** 官方 PNG 资源位于 [`AprilRobotics/apriltag-imgs`](https://github.com/AprilRobotics/apriltag-imgs)；family 36h11 的 ID 0 对应文件为 [`tag36h11/tag36_11_00000.png`](https://github.com/AprilRobotics/apriltag-imgs/blob/master/tag36h11/tag36_11_00000.png)，原图仅 10 × 10 px。该仓库自带的 `tag_to_svg.py` 可直接生成任意尺寸的矢量版本（如果你的打印机驱动接受 SVG，建议优先用矢量）：

   ```bash
   git clone https://github.com/AprilRobotics/apriltag-imgs
   cd apriltag-imgs
   python3 tag_to_svg.py tag36h11/tag36_11_00000.png tag36_11_00000.svg --size=48mm
   ```

2. **必须使用最近邻插值放大，禁止反走样。** 若坚持使用 PNG 路径，需要把 10 × 10 px 的原图放大到目标打印尺寸，并**必须**使用最近邻（nearest-neighbor）插值 — 一旦启用反走样，边缘会被平滑成灰阶过渡，破坏 detector 的角点梯度。ImageMagick 命令：

   ```bash
   # 10 px → 1134 px（48 mm @ 600 dpi）；通过 -filter point 强制最近邻
   convert tag36h11/tag36_11_00000.png -filter point -resize 11340% tag_48mm_600dpi.png
   ```

   在 GIMP / Photoshop 中操作时，请在 Image → Scale 对话框中将 Interpolation 设为 "None" 或 "Nearest neighbor"。

3. **预留安静区（quiet zone）。** 把 48 mm 的黑色图案放置在白色版面正中，四周各保留 ≥ 5 mm（约 1 cell 宽度）的纯白边距。请勿沿黑边裁切到边缘 — 一旦丢掉外缘梯度，detector 会拒绝识别该标签。

4. **打印参数。** ≥ 600 dpi；介质选哑光 vinyl 或哑光覆膜纸；黑色须为纯色调料 / 颜料墨，白底；务必避免高光面材质（工业 LED 照明下反光会直接击穿检测）。

5. **卡尺校核。** 用游标卡尺测量**黑色方块的外缘**（不是整张纸的外缘），确认两个方向都落在 48.0 ± 0.3 mm 内。任何超出该公差的偏差都会以同等比例传播为位姿估计误差。

6. **粘贴。** 按上图所示的朝向贴到手腕板上。重新运行 `vision` 之前，请再次确认本节"警告"中的注意事项。

## 5. 物理装配

### 5.1 手部安装

Wuji Hand 安装于一个 3D 打印治具上，治具用螺丝固定在铝合金蜂窝光学平板上。该治具让手腕 AprilTag 暴露给相机，并在手掌上方留出约 20 cm 的空隙以容纳 cube。请将 Hand 的 USB 线从手腕后方走线，避开相机视野。

![灵巧手在治具上的侧视图 — 组装好的 Wuji Hand 安装在 3D 打印治具上，手腕 AprilTag 装在顶部](images/hand-jig-side.jpg)

**硬件清单**：

| # | 图号 / 规格 | 零件名称 | 数量 | 材料 | 表面处理 | 类型 |
|---|---|---|---|---|---|---|
| 1 | 350 × 200 × 13 mm | 铝合金蜂窝板 | 1 | AL6061-T6 | 阳极黑色 | 标准件 |
| 2 | 见 `base.3mf`（release 附件） | 3D 打印底座 | 1 | PLA | 无 | 加工件 |

另需 4× M6 内六角螺丝（长度视蜂窝板厚度而定，常用 16 mm），用于把底座固定到蜂窝板上。

**装配步骤**：
1. 在支持 PLA 的 FDM 打印机上打印 `base.3mf`（来自 release 附件）— 文件内已捆绑 Bambu Lab 切片配置。
2. 把底座放置在铝合金蜂窝板上，使手腕安装托架朝前。底座有 4 个 φ6.60 mm 通孔 + φ11 mm 沉头孔，与蜂窝板上的 M6 螺纹阵列对齐。
3. 用 4 颗 M6 内六角螺丝穿过沉头孔，将底座拧紧到蜂窝板上。
4. 装配完成后整体高度约 147 mm，并将 Wuji Hand 后倾 10°，使静止状态下手腕标签朝向相机。
5. 把 Wuji Hand 卡入托架；将 Hand 的 USB 线从手腕后方走线，避开相机视野。

### 5.2 相机安装

把相机安装到这样一个位置：在整个 reorient 过程中，cube 可达工作区和手腕 AprilTag 始终都能完整出现在预览画面里。实际操作中大致是手掌上方 30–40 cm，但具体距离并不严格 — 策略在 rollout 期间会把 cube 保持在距手掌中心 ~10 cm 范围内，所以工作区盒子很小（手掌上方约 20 × 20 cm），中途也不会脱离画面。关键是 (a) 静止时手腕标签可见，以及 (b) cube 及其可达工作区能从容地落在下面选取的 ROI 之内。

**请勿手动编辑 [`camera.yaml`](../../deploy/reorient/config/camera.yaml) 中的 `fast_roi`** — vision 程序已内置交互式选择器。相机安装好后：

```bash
pixi run -e deploy vision
```

在 OpenCV 预览窗口中，按 **`s`** 打开 ROI 选择对话框。在 **cube 的可达工作空间**上拖出一个矩形（即 observer 在每帧上裁切后再做 ArUco 检测的 ROI）。手腕 AprilTag 不必持续在框内 —— 它只在启动时的 100 帧世界坐标系采样期间需要可见（按 `w` 重采样时同理，见下方表格）。按 ENTER / SPACE 确认（`C` 取消）。`cube_world_observer.py` 会：

- 将宽度 / 高度 / 偏移量对齐到 8 的倍数（Hikvision 传感器的要求），并强制最小 64 px；
- 原子地将新值写入 `config/camera.yaml::fast_roi`（load → modify → 临时文件 → rename）；
- 实时应用，无需重启采集。

随包默认值描述的是参考装置，首次按 `s` 保存时会被覆盖：

```yaml
fast_roi:
  offset_x: 464
  offset_y: 112
  width:    616
  height:   504
```

Vision 窗口的其他快捷键：

| 按键 | 作用 |
|---|---|
| `s` | 打开 ROI 选择器（如上） |
| `w` | 重新采样世界坐标系（重新检测手腕 AprilTag，重置 cube filter） |
| `r` | 仅重置 cube filter（世界坐标系不变） |
| `q` | 退出 |

Observer 在以 headless 方式启动（不带 `--preview`）时会自动切到配置好的 `fast_roi`；预览模式保留完整 ROI 可见，便于过程中重新构图。

**何时需要重新采样世界坐标系。** 每次启动 `vision`，observer 会自动采集 100 帧手腕 AprilTag 并冻结世界位姿（`_finalize_world_frame`）。出现以下情形时按 **`w`** 重新执行：

- 你重新安装了手部治具，手腕标签发生了位移（哪怕只有 1 mm）。
- Cube 位姿估计相对于实际手上 cube 出现明显的漂移或抖动。
- 你调整了相机位置、对焦或 `fast_roi`。

按 `w` 会重新检测标签、做一次新的 100 帧平均、并重置 cube filter。**rollout 进行中请勿按 `w`** — 策略读取的 cube-in-tag 观测是相对世界坐标系给出的，坐标系突变会破坏正在进行的 episode。

### 5.3 光照

请使用漫射环境光。避免逆光 — observer 使用 BGR 取最小通道得到灰度图（白色 → 255，彩色 → 约 0），因此标签边缘过曝是检测漏失的最大单一原因。保持 CLAHE 开启（[`observer.yaml`](../../deploy/reorient/config/observer.yaml) 中的 `enable_clahe: true`）；若 CLAHE 下 cube 面看起来"噪点多"，则关闭它并仅依靠 min-channel。

## 6. 相机内参标定

[`camera.yaml`](../../deploy/reorient/config/camera.yaml) 中的 `fx`、`fy`、`cx`、`cy` 以及 5 参数 Brown-Conrady 畸变描述的是参考装置。换用其他相机时**必须**在信任 cube 位姿之前重新标定 — 5% 的焦距误差会线性传播到 cube 位置。

### 6.1 打印棋盘格

11 × 8 **内部**角点（12 × 9 格），20 mm 方格，与标定器中的 `SQUARE_SIZE = 0.020` 常量匹配（若打印更大，请相应调整常量）。务必固定在刚性平整底板上 — 弯曲会引入系统性的径向偏差。

### 6.2 运行引导式标定器

```bash
pixi run -e deploy python deploy/reorient/tools/camera_calibrate.py
```

工具会引导走完 14 个采集任务（中心 / 左 / 右 / 上 / 下区域；近 / 中 / 远距离；正视 / 倾斜姿态），当区域/尺寸/倾角进入范围、经过 5 帧稳定、Laplacian 质量分数 ≥ 60 时自动采集。

按键：`c` 强制采集，`n` 跳过，`s` 拟合（需要 ≥ 12 张采集），`q` 退出。`s` 之后工具会打印 RMS 重投影误差并写入 `deploy/reorient/config/camera_calibration.npz`。目标是 RMS < 0.5 px；> 1.0 px 表示采集时棋盘移动了或对焦失误。

### 6.3 填写 camera.yaml

标定器只写入 `camera_calibration.npz`，**不会**就地更新 `camera.yaml` — 请手动把打印出的 `K` 和 `dist` 数值复制到 `intrinsics` 和 `distortion` 块中。与随包文件做 diff，确认 9 个数值（fx, fy, cx, cy, k1, k2, p1, p2, k3）都已填入。

### 6.4 合理性校验

`pixi run -e deploy vision`（预览模式）。当手腕标签静止不动时，其位姿应稳定到亚像素级抖动。Observer 会拒绝平均重投影误差 > 6.0 px 的 PnP 拟合（`observer.yaml::pnp.reproj_threshold`）；频繁的拒绝意味着内参拟合不足 — 回到 6.2 节，多采集一些倾斜样本。

## 7. 位姿估计调优

硬件固定后，[`observer.yaml`](../../deploy/reorient/config/observer.yaml) 提供四个旋钮用于在噪声与延迟之间权衡。

### 7.1 参数

- `rotation_filter.process_noise`（默认 0.5）— SO(3) 卡尔曼过程噪声；越高越灵敏，噪声越大。
- `rotation_filter.measurement_noise`（默认 0.1）— 越低越信任 PnP。
- `position_filter.alpha`（默认 0.8）— [0, 1] 范围的低通；1.0 = 不滤波。
- `pnp.reproj_threshold`（默认 6.0 px）— 超过该阈值的拟合会被丢弃（cube 进入"lost"状态；重新捕获时 filter 重置）。
- `preprocess.enable_clahe` / `clahe_clip` / `clahe_tile` — 在高对比度光照下 CLAHE 反而引入噪声时可关闭。

### 7.2 预设

在 `observer.yaml` 中以原文嵌入：

```yaml
# Agile (fast response, more noise):
#   process_noise: 0.5
#   measurement_noise: 0.1
#   alpha: 0.8
#
# Smooth (stable, slower response):
#   process_noise: 0.01
#   measurement_noise: 2.0
#   alpha: 0.2
```

随包默认是 agile 预设（也是训练策略部署时使用的配置）。

### 7.3 排错

- Cube 静止时抖动 → 切换到 smooth 预设。
- 快速重定向时位姿滞后 → 切换到 agile 预设。
- Cube 反复跌入"lost"状态 → 把 `pnp.reproj_threshold` 提高到例如 8.0 px，**并且**重新检查第 6 节内参；如果发现轴向错，修 `cube_tags.json::face_rotations` 或者重贴对应面的 tag。

## 8. 端到端冒烟测试

至此你已有一套经过标定的装置。按顺序走完以下五个检查点；若任何一步失败，请先跳回对应章节再继续。

### 8.1 步骤 1 — 让手回零

`pixi run -e deploy home` — 3 秒平滑斜坡；20 个关节都落入 home 位置 ±2° 范围内；脚本会打印 "Within 2° — home reached"。手指抖动或硬卡停 → 拔插 Hand 的 USB 线后重试。

### 8.2 步骤 2 — 启动 cube observer

`pixi run -e deploy vision`。预期：OpenCV 预览窗口出现；当手腕标签保持在视野内时，黄色 "World Sampling: N/100" 进度条逐渐填满；累积 100 个样本平均后，标签翻转为绿色 "WORLD FIXED"；cube 静止时轴向叠加图稳定。

### 8.3 步骤 3 — 验证 ZMQ 位姿流

在第二个终端中（`vision` 已运行），确认 cube 位姿正在端口 **5555** 上发布（`control.yaml::zmq.cube_port`）：

```bash
pixi run -e deploy python - <<'EOF'
import json, zmq
sock = zmq.Context().socket(zmq.SUB)
sock.connect("tcp://localhost:5555")
sock.subscribe(b"")
for _ in range(3):
    msg = json.loads(sock.recv_string())
    p = msg["cube1"]["position"]
    print(f"frame={msg['frame']:5d}  pos=({p['x']:+.3f},{p['y']:+.3f},{p['z']:+.3f})")
EOF
```

你应当看到 3 个新的 frame 编号和稳定的位置数值。

### 8.4 步骤 4 — cube 位姿可视化检查

保持 `vision` 仍在运行，新开一个终端：

```bash
pixi run -e deploy python deploy/reorient/tools/calib_check.py
```

这个工具会打开一个 MuJoCo passive viewer 渲染数字孪生——手部 mirror 实时 encoder 读数，cube 显示在 observer 估计的位姿上。脚本启动时手回 home 一次，之后不再动；你可以自由挪动物理 cube，观察渲染 cube 是否跟随。

相比 8.3 只是验证 ZMQ 流活着，这一步还能查出：

- **轴向错配** — 让物理 cube 绕某个面轴旋转，渲染 cube 应当绕同一个轴。镜像或 90° 偏差意味着 `cube_tags.json::face_rotations` 不对，或者某个 tag 贴反了。
- **位置偏差** — 把 cube 放在手掌中心；渲染 cube 应当落在 palm geom 上。> 2 cm 偏差通常意味着手部安装（5.1 节）或相机内参（第 6 节）有问题。
- **位姿延迟或抖动** 超出 7.1 节滤波器旋钮能解释的范围。

Ctrl+C 或关闭 viewer 窗口退出。

### 8.5 步骤 5 — 运行闭环策略

`pixi run -e deploy play-real --ckpt <path-to.onnx>`。预期：ONNX 策略加载并打印 sidecar JSON；手部经由 `env.reset()` 回零；一个 passive MuJoCo "mirror" viewer 打开（显示真实关节 + 观测到的 cube + 手掌上方 10 cm 处的半透明目标 cube）；手部把 cube 朝目标姿态调整，benchmark 各次试验结果实时打印。

如果策略一上来就发散，参见 8.6 节。

### 8.6 排错矩阵

| 现象 | 可能原因 | 解决方法 |
|---|---|---|
| 相机打不开 | MVS SDK 未安装 / `MVS_PYTHON_PATH` 未设置 | 重做 2.2 节；再跑一遍导入冒烟测试 |
| 手腕 AprilTag 一直检测不到 | 光照 / 标签 family 错 / ID 错 / 尺寸错 | 确认 AprilTag36h11，ID 0，48 mm；加强光照 |
| `World Sampling` 进度条一直不满 | 手腕标签太小/模糊 | 调位置使手腕标签 ≥ 80 px 宽；重新对焦 |
| Cube observer 频繁掉 cube | 重投影误差门触发 | 重做第 6 节内参；用 8.4 节 calib_check 校验 `cube_tags.json` 面映射 |
| 策略第一步就发散 | 8.4 节查出 tag 朝向错配 | 修 `cube_tags.json::face_rotations` 或重贴对应面的 tag |
| Rollout 时手抖动 | `control_dt` 策略侧 ↔ 硬件侧不匹配 | 检查 ONNX sidecar 的 `ctrl_dt`；降低 `control.yaml::hardware.lowpass_cutoff_hz` |
| Mirror viewer 渲染冻结的位姿 | `mj_data` 模板过期 | 重启 `play-real`（viewer 在启动时通过 `_viz_mj_data` 绑定） |

---

License: Apache 2.0. 见仓库根目录 [`LICENSE`](../../LICENSE)。
