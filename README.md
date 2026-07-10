# wuji-mjlab

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Release](https://img.shields.io/github/v/release/wuji-technology/wuji-mjlab)](https://github.com/wuji-technology/wuji-mjlab/releases)
[![CI](https://github.com/wuji-technology/wuji-mjlab/actions/workflows/ci.yml/badge.svg)](https://github.com/wuji-technology/wuji-mjlab/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Stars](https://img.shields.io/github/stars/wuji-technology/wuji-mjlab?style=social)](https://github.com/wuji-technology/wuji-mjlab/stargazers)

wuji-mjlab is an in-hand cube reorientation RL project for the Wuji Hand. It trains PPO policies in [mjlab](https://github.com/mujocolab/mjlab) that cover the full SO(3) goal space, and deploys them on the physical hand through a sim-to-real bridge. The repo ships pretrained checkpoints, deployment scripts, and hardware guides so you can reproduce the demo end to end.

**Get started with [Quick Start](#quick-start). For detailed documentation, please refer to [Wuji MJLab](https://docs.wuji.tech/docs/en/wuji-mjlab/latest/) on Wuji Docs Center.**

<p align="center">
  <img src="docs/assets/sim.gif" width="45%" alt="sim reorient demo" />
  <img src="docs/assets/real.gif" width="45%" alt="real-hand reorient demo" />
</p>

## Repository Structure

```text
wuji-mjlab/
├── src/
│   ├── wuji_mjlab/        // task package (tasks/reorient/, assets/, utils/, rl/)
│   └── wuji_rl_libs/      // vendored rsl-rl PPO backend
├── deploy/reorient/       // sim-to-real bridge (vision, ZMQ, hand driver)
├── scripts/               // train / play / tools entry points
├── docs/                  // architecture + sim-to-real setup
├── pixi.toml              // canonical install + task runner
└── pyproject.toml         // package metadata
```

## Quick Start

### Installation

Requirements: Linux x86_64, an NVIDIA GPU with CUDA 12.8, and [pixi](https://pixi.sh) ≥ 0.66.

> ⚠️ **CAUTION**: this repo is **pixi-only**. `conda + pip install -e .` is not tested and not supported.

```bash
# 1. install pixi (one-time)
curl -fsSL https://pixi.sh/install.sh | bash

# 2. clone + resolve environment
git clone https://github.com/wuji-technology/wuji-mjlab
cd wuji-mjlab
pixi install
```

### Running

Train a policy, or skip training and deploy the pretrained checkpoint from [Releases](https://github.com/wuji-technology/wuji-mjlab/releases/latest):

```bash
# Train (GPU, ~20 GB VRAM)
pixi run train --task WujiHand_Reorient --agent.upload-model False

# Replay a trained checkpoint in the interactive viewer
pixi run play --task WujiHand_Reorient --checkpoint-file <path-to-ckpt.pt>
```

Deploy on the real hand (hardware setup required, see [Sim-to-real Deployment](https://docs.wuji.tech/docs/en/wuji-mjlab/latest/sim2real/) on Wuji Docs Center):

```bash
pixi run -e deploy home                              # reset hand to home pose
pixi run -e deploy vision                            # launch cube observer (OpenCV preview)
pixi run -e deploy play-real --ckpt <path-to.onnx>   # closed-loop control + mirror viewer
```

For training variants, evaluation, ONNX export, hardware setup, and architecture details, see the [full documentation](https://docs.wuji.tech/docs/en/wuji-mjlab/latest/).

## Contributors

- [Jielin Wu](https://github.com/AIRJASON50)
- [Shenzhe Yao](https://github.com/LeopoldYao)
- [Han Yang](https://github.com/yanghan-a)
- [Xiangrui Jiang](https://github.com/XiangruiJiang)
- [Wentao Zhang](https://github.com/zhangwt20011015)
- [Li Chengmeng](https://github.com/AsahelLee)
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

## Appendix

- **Documentation**: [Wuji MJLab on Wuji Docs Center](https://docs.wuji.tech/docs/en/wuji-mjlab/latest/)
- **Related Projects**: [wujihandpy](https://github.com/wuji-technology/wujihandpy) (Wuji Hand SDK), [wuji-retargeting](https://github.com/wuji-technology/wuji-retargeting) (hand pose retargeting), [wujihandros2](https://github.com/wuji-technology/wujihandros2) (ROS2 driver for Wuji Hand)
- **References**: built on [mjlab](https://github.com/mujocolab/mjlab), [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), [MuJoCo](https://mujoco.org/), [rsl_rl](https://github.com/leggedrobotics/rsl_rl) (vendored under `src/wuji_rl_libs/`), and [pupil-apriltags](https://github.com/pupil-labs/apriltags). See [NOTICE](NOTICE) for third-party attribution
- **Contributing**: install pre-commit hooks with `pixi run pre-commit install` before committing

## Contact

For any questions, please contact [support@wuji.tech](mailto:support@wuji.tech).
