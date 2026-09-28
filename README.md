# wuji-mjlab

[Chinese version](README_zh.md)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Release](https://img.shields.io/github/v/release/wuji-technology/wuji-mjlab)](https://github.com/wuji-technology/wuji-mjlab/releases)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![GitHub Stars](https://img.shields.io/github/stars/wuji-technology/wuji-mjlab?style=social&label=Stars)](https://github.com/wuji-technology/wuji-mjlab/stargazers)

> Dexterous manipulation with Wuji Hand: mjlab and PPO for in-hand cube reorientation on both hand generations and reference-tracking pen spinning on Wuji Hand 2, with closed-loop deployment on the physical hand.

## Demos

<table width="100%">
  <tr>
    <td width="32%" valign="top"><a href="src/wuji_mjlab/tasks/reorient/docs/assets/sim_labeled.gif"><img src="src/wuji_mjlab/tasks/reorient/docs/assets/sim_labeled.gif" width="100%" align="top" alt="Wuji Hand 1 · Cube · SIM" /></a><br /><a href="src/wuji_mjlab/tasks/reorient/docs/assets/real_labeled.gif"><img src="src/wuji_mjlab/tasks/reorient/docs/assets/real_labeled.gif" width="100%" align="top" alt="Wuji Hand 1 · Cube · REAL" /></a></td>
    <td width="32%" valign="top"><a href="src/wuji_mjlab/tasks/reorient/docs/assets/sim_hand2_labeled.gif"><img src="src/wuji_mjlab/tasks/reorient/docs/assets/sim_hand2_labeled.gif" width="100%" align="top" alt="Wuji Hand 2 · Cube · SIM" /></a><br /><a href="src/wuji_mjlab/tasks/reorient/docs/assets/real_hand2_labeled.gif"><img src="src/wuji_mjlab/tasks/reorient/docs/assets/real_hand2_labeled.gif" width="100%" align="top" alt="Wuji Hand 2 · Cube · REAL" /></a></td>
    <td width="36%" valign="top"><a href="src/wuji_mjlab/tasks/pen_spin/docs/assets/clip3_labeled.gif"><img src="src/wuji_mjlab/tasks/pen_spin/docs/assets/clip3_labeled.gif" width="100%" align="top" alt="Wuji Hand 2 · PenSpin Group 3 · REAL · 1X Speed" /></a></td>
  </tr>
</table>

Left: **Wuji Hand 1 · Cube**. Middle: **Wuji Hand 2 · Cube**. Both columns show **SIM above REAL**. Right: **Wuji Hand 2 · PenSpin Group 3**. All five clips loop independently at **1X Speed**. Click a clip to enlarge it.

## Requirements

- Linux x86_64
- NVIDIA GPU, CUDA 12.8 (Blackwell sm_120 / RTX 50-series supported)
- [pixi](https://pixi.sh) ≥ 0.72 — **the only supported installer**

> ⚠️ **CAUTION**: this repo is **pixi-only**. `conda + pip install -e .` is not tested and not supported.

## Installation

```bash
# 1. Install pixi (one-time)
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"

# 2. Clone the repository and install the training environment
git clone https://github.com/wuji-technology/wuji-mjlab
cd wuji-mjlab
pixi install

# 3. Verify installation and list registered tasks
pixi run list-envs
```

`pixi install` creates the `default` environment for training and playback. Continue with a task guide below to prepare data and start training; each guide links to its deployment environment setup.

## Tasks

[v2026.9.27 release](https://github.com/wuji-technology/wuji-mjlab/releases/tag/v2026.9.27) · [Download asset bundle](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip) — includes shared hand jigs, Reorient models and cube assets, plus PenSpin hardware, 40 motion clips, five recipes and five matched checkpoint sets.

Paths below are relative to the extracted `wuji-mjlab-v2026.9.27-assets/` directory. Setup, training and deployment instructions are in this repository.

| Bundle path | Contents |
|---|---|
| `hardware/hand1/`, `hardware/hand2/` | Mounting jigs; Hand 2 is shared by Reorient and PenSpin |
| `reorient/checkpoints/`, `pen_spin/checkpoints/` | Per-hand or per-motion-group `model.pt`, `policy.onnx` and `config.json` |
| `reorient/hardware/` | Cube STEP, 3MF, OBJ, MTL and PNG files |
| `pen_spin/hardware/` | Pen STEP and 3MF files; `tags/` contains tag assets |
| `pen_spin/data/rawdata/clips/`, `pen_spin/data/recipe/` | 40 motion clips and five recipes |

| Task | Supported hand | Setup | Train | Deploy |
|---|---|---|---|---|
| [Cube Reorient](src/wuji_mjlab/tasks/reorient/README.md) | Wuji Hand 1 / Wuji Hand 2 · right | [Setup](src/wuji_mjlab/tasks/reorient/docs/setup.md) | [Train](src/wuji_mjlab/tasks/reorient/docs/train.md) | [Deploy](src/wuji_mjlab/tasks/reorient/docs/deploy.md) |
| [PenSpin](src/wuji_mjlab/tasks/pen_spin/README.md) | Wuji Hand 2 · right | [Setup](src/wuji_mjlab/tasks/pen_spin/docs/setup.md) | [Train](src/wuji_mjlab/tasks/pen_spin/docs/train.md) | [Deploy](src/wuji_mjlab/tasks/pen_spin/docs/deploy.md) |

## Contributors

- [Jielin Wu](https://github.com/AIRJASON50)
- [Shenzhe Yao](https://github.com/LeopoldYao)
- [Han Yang](https://github.com/yanghan-a)
- [Xiangrui Jiang](https://github.com/XiangruiJiang)
- [Wentao Zhang](https://github.com/zhangwt20011015)
- [Li Chengmeng](https://github.com/AsahelLee)
- [Lijie Sheng](https://github.com/2384447291)
- [Xiaohan Liu](https://github.com/Infas12)
- [Guanqi He](https://github.com/GuanqiHe)

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{wuji2026mjlab,
  title={Wuji-MJLab: RL Training for Wuji Hand Dexterous Manipulation},
  author={{Wuji Technology}},
  year={2026},
  url={https://github.com/wuji-technology/wuji-mjlab}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party attribution.
