# Cube Reorientation · Train

[中文版](train_zh.md) · [Task home](../README.md) · [Setup](setup.md) · [Train](train.md) · [Deploy](deploy.md)

## Environment and task selection

Complete [environment installation](../../../../../README.md#installation) first. Run all commands below from the **repository root**.

| Task ID | Hand / control rate |
|---|---|
| `WujiHand_Reorient` | Wuji Hand 1, right, 20 Hz |
| `WujiHand2_Reorient` | Wuji Hand 2, right, 20 Hz baseline |
| `WujiHand2_Reorient_50Hz` | Wuji Hand 2, right, 50 Hz, recommended |

## Train

W&B logging requires explicit `WANDB_PROJECT` and `WANDB_API_KEY` environment variables.
Missing or blank values stop training with an error; cached login credentials do not replace them:

```bash
export WANDB_PROJECT=your-wandb-project
read -rsp "W&B API key: " WANDB_API_KEY
export WANDB_API_KEY
pixi run train --task WujiHand2_Reorient_50Hz --agent.upload-model False
```

`--agent.upload-model False` keeps checkpoints local-only. Drop it to also push the final-iteration checkpoint to W&B as a model artifact — local `.pt` files are still written periodically either way. Pass `--agent.logger tensorboard` instead if you don't want W&B at all.

Checkpoints and W&B logs land under `logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/`.

Change `--task` to an ID in the table above to train another hand configuration. Override defaults with dotted arguments, for example:

```bash
pixi run train --task WujiHand2_Reorient_50Hz \
  --agent.logger tensorboard --agent.upload-model False \
  env.scene.num_envs=2048 agent.max_iterations=2000
```

Use `--config your.yaml` to reuse a task and overrides; see `wuji_mjlab.utils.train_config_utils` for the schema. No presets ship in-tree.

## Playback

Use the same task ID as training. The examples below use Wuji Hand 2 at 50 Hz:

Replace angle-bracket paths below with actual local files. Keep `params/agent.yaml` and `params/env.yaml` in the checkpoint run directory.

```bash
pixi run play --task WujiHand2_Reorient_50Hz --checkpoint-file "<path-to-ckpt.pt>"
```

## Export ONNX

```bash
# Export policy.onnx and matching config.json beside the checkpoint
pixi run python -m wuji_mjlab.tasks.reorient.scripts.export_onnx "<path-to-ckpt.pt>"
```

Use `--filename` to choose a relative ONNX filename. Keep the configuration exported with the policy.

## Evaluation

```bash
# Wuji Hand 2 evaluation requires --hand hand2; use --hand hand1 for Wuji Hand 1
pixi run python -m wuji_mjlab.tasks.reorient.scripts.eval_success_rate "<path-to-policy.onnx>" --hand hand2

# Headless evaluation with JSON results
pixi run python -m wuji_mjlab.tasks.reorient.scripts.eval_success_rate "<path-to-policy.onnx>" \
  --hand hand2 --num-trials 100 --no-viewer --json-output "artifacts/eval/<run>/result.json"
```

Replace `<run>` with a name for this evaluation run. `--json-output` selects the structured JSON path; `cube_motion_log.csv` and `eval_results.json` are written in the same parent directory, which is created automatically. Without this option, companion files are written beside the ONNX model. Use separate directories for separate evaluation runs.

## Developer reference

The target orientation uses the wrist AprilTag frame; see the `*_wrist_tag` site in `mdp/observations.py`.

```bash
pixi run list-envs
pixi run python -m wuji_mjlab.tasks.reorient.scripts.view_task WujiHand2_Reorient_50Hz
```

### Layout

- `mdp/` — task terms.
  - `cage.py` — palm-relative cage geometry.
  - `command_visualization.py` — goal visualization.
  - `commands.py` — SO(3) goal commands.
  - `event_impl/` — event implementations.
  - `_env_utils.py` — environment ID helpers.
  - `runtime_state.py` — per-environment state.
  - `actions.py` / `observations.py` / `rewards.py` / `terminations.py`
    / `curriculums.py` / `metrics.py` — mjlab task terms.
  - `events.py` — event exports.
  - `types.py` — curriculum report types.
- `reorient_terms.py` — task term builders.
- `reorient_env_cfg.py` — environment configuration.
- `reorient_constants.py` — task pose constants.
- `config/wuji_hand/` — Wuji Hand 1 configuration.
  - `env_cfgs.py` — environment configuration.
  - `rsl_rl/ppo.py` — PPO runner configuration.
  - `__init__.py` — task registration.
- `config/wuji_hand2/` — Wuji Hand 2 configuration.
  - `env_cfgs.py` — environment and play configuration.
  - `hand2_constants.py` — rig injection, collision capsule overlay, actuator gains, and robot/cube initial-pose constants.
  - `rsl_rl/ppo.py` — PPO runner configuration.
  - `__init__.py` — task registration.
- `tooling/` — evaluation and export tools.
  - `cage_pose_check.py` — cage-pose check.
  - `eval_core.py` — programmatic policy evaluation.
  - `eval_display.py` — evaluation display.
  - `scene_builder.py` — evaluation scene construction.
  - `onnx_export_core.py` — ONNX policy export.
  - `validate_obs_vs_sim.py` — simulation/deployment comparison.
  - `cage_bounds_check.py` — cage boundary check.
  - `capsule_overlay_check.py` — interactive collision capsule and mesh convex-hull check.
  - `hand2_check_scene.py` — interactive scene helpers.
  - `stamp_policy_metadata.py` — ONNX policy metadata.
- `scripts/` — command-line entry points.
- `tests/` — task tests.
  - `integration/` — integration tests.

### Public entrypoints

**Env config**:
- `make_reorient_env_cfg()` → `ManagerBasedRlEnvCfg` (robot-agnostic baseline)
- `wuji_hand_reorient_env_cfg(play: bool = False)` → Wuji Hand hardware overlay
- `wuji_hand2_reorient_env_cfg()` → Wuji Hand 2 hardware overlay

**RL config**:
- `wuji_hand_reorient_ppo_runner_cfg()` → RSL-RL PPO runner cfg
- `wuji_hand2_reorient_ppo_runner_cfg()` → RSL-RL PPO runner cfg

When constructing a training-mode environment outside the runner, set `env.max_common_steps = agent_cfg.max_iterations * agent_cfg.num_steps_per_env` before the first step. The standard training entrypoint, `view_task`, and `play_runner` already handle the configuration they need.

**Eval (programmatic)**:
- `from wuji_mjlab.tasks.reorient.tooling.eval_core import EvalConfig, EvalResult, run_eval`
- Pass a populated `EvalConfig` to `run_eval()` and read back a structured
  `EvalResult` (success_rate, drop_rate, per-trial outcomes, etc.)
- `run_eval()` returns an `EvalResult`; to write a separate structured JSON file, the caller must save `result.to_json()`.

**Registered task IDs** (see `config/wuji_hand/__init__.py` and
`config/wuji_hand2/__init__.py`):
- `WujiHand_Reorient` — Wuji Hand 1, right hand, 20 Hz
- `WujiHand2_Reorient` — Wuji Hand 2, right hand, 20 Hz baseline
- `WujiHand2_Reorient_50Hz` — Wuji Hand 2, right hand, 50 Hz deployment recipe
  (`decimation=2`)

**Programmatic eval** (no CLI, useful for batch evaluation):

```python
from pathlib import Path
from wuji_mjlab.tasks.reorient.tooling.eval_core import EvalConfig, run_eval

result = run_eval(
  EvalConfig(
    onnx_path=Path("logs/rsl_rl/wuji_reorient_hand2/<run>/policy.onnx"),
    hand="hand2",
    num_trials=50,
    no_viewer=True,
  )
)
print(f"success_rate = {result.success_rate:.2%}")
print(f"mean min ori error = {result.mean_min_ori_error_rad:.3f} rad")

# Per-trial data for custom analysis
for trial in result.trials:
  if trial.status == "success":
    print(f"trial {trial.trial_idx}: t_first_succ={trial.time_to_first_success_s:.2f}s")
```

Next: use [Deploy](deploy.md) to select a policy and run it on the physical hand.
