# wuji-mjlab

[中文版](README_zh.md)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Release](https://img.shields.io/github/v/release/wuji-technology/wuji-mjlab)](https://github.com/wuji-technology/wuji-mjlab/releases)
[![CI](https://github.com/wuji-technology/wuji-mjlab/actions/workflows/ci.yml/badge.svg)](https://github.com/wuji-technology/wuji-mjlab/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Stars](https://img.shields.io/github/stars/wuji-technology/wuji-mjlab?style=social)](https://github.com/wuji-technology/wuji-mjlab/stargazers)

> In-hand cube reorientation on the Wuji Hand: PPO policies trained in mjlab (GPU-batched physics via mujoco-warp), covering the full SO(3) goal space, with a sim2real bridge for closed-loop deployment on the physical hand.

<p align="center">
  <img src="docs/assets/sim.gif" width="45%" alt="sim reorient demo" />
  <img src="docs/assets/real.gif" width="45%" alt="real-hand reorient demo" />
</p>

## Tasks

| Robot | Task ID | Pretrained checkpoint | Demo |
|---|---|---|---|
| Wuji Hand | `WujiHand_Reorient` | [Latest release assets](https://github.com/wuji-technology/wuji-mjlab/releases/latest) | sim + real GIFs above |

Pull the checkpoint and CAD bundle from the latest release:

```bash
# Requires gh CLI (https://cli.github.com); the glob keeps this command
# working across future release tags. See docs/sim2real/setup.md §3 for
# the manual fallback if you don't have gh installed.
gh release download --repo wuji-technology/wuji-mjlab --pattern '*-assets.zip'
unzip wuji-mjlab-*-assets.zip
mv wuji-mjlab-*-assets release-assets
```

## Repository layout

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

## Requirements

- Linux x86_64
- NVIDIA GPU, CUDA 12.8 (Blackwell sm_120 / RTX 50-series supported)
- [pixi](https://pixi.sh) ≥ 0.66 (the version CI uses) — **the only supported installer**
- For sim2real: Wuji Hand hardware + Hikrobot USB-3 camera + Hikvision MVS SDK + 3D-printed ArUco-tagged cube + wrist AprilTag — see [`docs/sim2real/setup.md`](docs/sim2real/setup.md)

> ⚠️ **CAUTION**: this repo is **pixi-only**. `conda + pip install -e .` is not tested and not supported.

## Installation

```bash
# 1. install pixi (one-time)
curl -fsSL https://pixi.sh/install.sh | bash

# 2. clone + resolve environment
git clone https://github.com/wuji-technology/wuji-mjlab
cd wuji-mjlab
pixi install
```

This produces a `default` environment for training/eval and an optional `deploy` environment (`pixi install -e deploy`) for the sim2real bridge.

Verify the environment: `pixi run list-envs` (lists registered tasks and confirms the mjlab + tyro stack imports cleanly).

## Train

```bash
pixi run train --task WujiHand_Reorient --agent.upload-model False
```

`--agent.upload-model False` keeps checkpoints local-only. Drop it (and set `WANDB_API_KEY`) to also push the final-iteration checkpoint to W&B as a model artifact — local `.pt` files are still written on every `save_interval` boundary either way.

> **If `pixi run train` OOMs**, swap to the lower-VRAM variant:
>
> ```bash
> pixi run train --task WujiHand_Reorient_Light
> ```
>
> `WujiHand_Reorient` approximately reproduces the released checkpoint at `num_envs=8192, max_iterations=5000` and needs ~20 GB of GPU memory. `WujiHand_Reorient_Light` uses `num_envs=4096, max_iterations=7500` — fits comfortably under ~12 GB but converges to a visibly weaker policy (occasional cube drops, finger-jam behavior on harder reorientations).

Checkpoints and W&B logs land under `logs/rsl_rl/<run_name>/`. Task MDP, reward shaping, and the contact-parameter domain randomisation split into two anatomical groups (palm + thumb compliance zone vs fingers 2-5) are documented in the Architecture section below.

## Play and evaluate

```bash
# Interactive viewer with a trained checkpoint
pixi run play --task WujiHand_Reorient --checkpoint-file <path-to-ckpt.pt>

# Success-rate eval over N trials (consumes ONNX)
pixi run python -m wuji_mjlab.tasks.reorient.scripts.eval_success_rate <path-to-policy.onnx>

# Export PPO checkpoint → ONNX (sidecar JSON with action_scale / ema_alpha / ctrl_dt)
pixi run python -m wuji_mjlab.tasks.reorient.scripts.export_onnx <path-to-ckpt.pt>
```

Additional dev utilities:

```bash
pixi run list-envs                                                              # list registered tasks
pixi run python -m wuji_mjlab.tasks.reorient.scripts.view_task WujiHand_Reorient  # view task with a dummy policy
```

## Sim-to-real

<p align="center">
  <img src="docs/assets/deploy.gif" width="80%" alt="sim2real deploy rig: camera over the Wuji Hand + jig, MuJoCo mirror viewer on the right" />
</p>

The deploy bridge runs the exported ONNX policy on the real Wuji Hand. A vision module tracks an ArUco-tagged cube (anchored to a wrist AprilTag world frame) via a USB camera and publishes the pose over ZMQ; `play_real` subscribes to that pose, runs ONNX inference, and closes the loop by sending commands to the hand driver.

> **No training needed to deploy.** Download the pre-trained `policy.onnx` + `policy_config.json` from [Releases](https://github.com/wuji-technology/wuji-mjlab/releases) and pass the `policy.onnx` path as `--ckpt` below. The released policy is what produces the demo GIF above.

```bash
# After the hardware setup in docs/sim2real/setup.md is complete:
pixi run -e deploy home                              # reset hand to home pose
pixi run -e deploy vision                            # launch cube observer (OpenCV preview)
pixi run -e deploy play-real --ckpt <path-to.onnx>   # closed-loop control + mirror viewer
```

- **Software pipeline & configuration**: [`deploy/reorient/README.md`](deploy/reorient/README.md)
- **Hardware setup, 3D-printed cube, camera mounting, calibration**: [`docs/sim2real/setup.md`](docs/sim2real/setup.md)

## Architecture

Three-layer stack: this repo (tasks + deploy) → [mjlab](https://github.com/mujocolab/mjlab) → [MuJoCo](https://mujoco.org) + [mujoco-warp](https://github.com/google-deepmind/mujoco_warp). PPO via the vendored [rsl-rl](https://github.com/leggedrobotics/rsl_rl) backend under `src/wuji_rl_libs/rsl_rl/`.

<details>
<summary>Deep dive — three-layer diagram, MDP spec, domain randomisation, adding a new task</summary>

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

### Reorient task (`src/wuji_mjlab/tasks/reorient/`)

Full SO(3) in-hand reorientation with the Wuji Hand. Files:

| File | Role |
|---|---|
| `reorient_env_cfg.py` | Top-level `ManagerBasedRlEnvCfg` factory |
| `reorient_terms.py` | All event / termination / reward / DR terms (the **task design** lives here, not in robot bindings) |
| `reorient_constants.py` | Initial pose constants (palm-up R_y(-90°), cube above palm) |
| `config/wuji_hand/` | Robot-binding layer: thin wiring of the task design onto Wuji Hand (20-DoF dexterous hand) |
| `mdp/` | Observations, commands, actions specific to reorientation |
| `tooling/` | Eval entrypoints + ONNX export |

Task design (MDP terms, reward shaping, anatomically-split contact-parameter DR groups) lives in `reorient_terms.py`. See [`src/wuji_mjlab/tasks/reorient/README.md`](src/wuji_mjlab/tasks/reorient/README.md) for the architecture invariants and [`deploy/reorient/README.md`](deploy/reorient/README.md) for the sim2real bridge — `RealHandEnv` reuses the sim observation + action managers verbatim, no parallel pipelines.

### Adding a new task

1. Create `src/wuji_mjlab/tasks/<your_task>/` with an env cfg factory.
2. Put all MDP design (events, rewards, terminations) in `<your_task>_terms.py`. The robot-specific config layer in `config/<robot>/` should be a thin binding only.
3. Register via `register_mjlab_task()` in `config/<robot>/__init__.py`.
4. Add a quick `pixi run train --task <your_task_id>` smoke run before committing (canonical training entrypoint is `scripts/train/train_rsl_rl.py`, exposed via the `train` pixi task).

</details>

## Development

After cloning, install the pre-commit hooks:

```bash
pixi run pre-commit install
```

Every `git commit` then runs ruff, codespell, and the YAML/TOML/large-file checks defined in [`.pre-commit-config.yaml`](.pre-commit-config.yaml) — locally, before CI sees the change. Manual full-tree run: `pixi run pre-commit run --all-files`.

> ⚠️ Don't `pip install` packages into the pixi env — pip deps aren't tracked by `pixi.toml` / `pixi.lock` and disappear on the next resolve. Edit `pixi.toml` and run `pixi install`.

## Related Projects

- [wujihandpy](https://github.com/wuji-technology/wujihandpy) — Wuji Hand SDK (C++ core with Python bindings)
- [wuji-retargeting](https://github.com/wuji-technology/wuji-retargeting) — Hand pose retargeting (Vision Pro / glove / video → robot joints)
- [wujihandros2](https://github.com/wuji-technology/wujihandros2) — ROS 2 driver for Wuji Hand
- [docs.wuji.tech](https://docs.wuji.tech) — Official Wuji documentation portal

## Acknowledgements

This project builds on the following open-source projects:

- [mjlab](https://github.com/mujocolab/mjlab) — manager-based RL framework
- [mujoco-warp](https://github.com/google-deepmind/mujoco_warp) — GPU-batched MuJoCo physics
- [MuJoCo](https://mujoco.org/) — the underlying physics engine
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) — PPO implementation (vendored under `src/wuji_rl_libs/`)
- [pupil-apriltags](https://github.com/pupil-labs/apriltags) — AprilTag detector for the deploy vision module

## Contributors

- [Jielin Wu](https://github.com/AIRJASON50)
- [Shenzhe Yao](https://github.com/LeopoldYao)
- [Han Yang](https://github.com/yanghan-a)
- [Li Chengmeng](https://github.com/AsahelLee)

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
