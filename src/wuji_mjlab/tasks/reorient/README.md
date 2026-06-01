# Reorient Task

[中文版](README_zh.md)

SO(3) in-hand reorientation of a cube held by a downward-facing dexterous hand.
The policy receives a target orientation (in the palm's "tag" frame) and must
rotate the cube in-place until its orientation matches the goal within a hold
window, without dropping it.

## Layout

- `mdp/` — runtime task terms consumed by the mjlab manager system
  - `cage.py` — palm-relative AABB "cage" geometry, escape counter, reward
    escalation, and termination
  - `command_visualization.py` — Viser/MuJoCo GUI rendering for the goal pose
    (markers, status markdown, debug triads)
  - `commands.py` — `InHandReorientCommand` / `InHandReorientCommandCfg`:
    SO(3) goal state machine (sample → wait → resample)
  - `event_impl/` — internal split of large event helpers (episode reset,
    randomization, joint reset, curriculum, shared event state). Not part
    of the public API — import via the `mdp.events` facade
  - `event_utils.py` — shared helpers used across event modules
    (`get_linear_progress`, `resolve_env_ids`)
  - `actions.py` / `observations.py` / `rewards.py` / `terminations.py`
    / `curriculums.py` / `metrics.py` — standard mjlab term modules
  - `events.py` — public events facade; re-exports the `event_impl/` API
  - `types.py` — shared dataclasses
- `reorient_terms.py` — `build_reorient_*` builders consumed by
  `make_reorient_env_cfg`
- `reorient_env_cfg.py` — thin assembler; exposes `make_reorient_env_cfg()`
- `reorient_constants.py` — task-wide public constants
  (e.g. `TAG_IN_PALM_POS`, `TAG_IN_PALM_QUAT_WXYZ`)
- `config/wuji_hand/` — Wuji Hand hardware overlay with a backend-specific subdir
  (`rsl_rl/`)
  - `env_cfgs.py` — env config + play overlay (`_PLAY_DISABLED_EVENTS`)
  - `rsl_rl/ppo.py` — `wuji_hand_reorient_ppo_runner_cfg()` (RSL-RL PPO)
  - `__init__.py` — registers `WujiHand_Reorient` (release) and
    `WujiHand_Reorient_Light` (lower-VRAM variant)
- `tooling/` — importable, side-effect-free implementations consumed by
  `scripts/`. Anything here is safe to import from Python (sweeps,
  notebooks, tests) without invoking a CLI
  - `eval_core.py` — `EvalConfig`, `EvalResult`, `TrialOutcome`,
    `run_eval()`, `ObsBuilder`, ONNX policy eval loop
  - `eval_display.py` — terminal renderer (`EvalDisplay`) for live eval output
  - `scene_builder.py` — MuJoCo scene construction, quat math, hold state
    machine, contact helpers (shared between eval and scene viz)
  - `onnx_export_core.py` — ONNX policy export logic
- `scripts/` — thin CLI wrappers (argparse + delegate to `tooling/`).
  These are the user-facing entry points; they own only env-var setup
  (e.g. `MUJOCO_GL=glfw`) and the `if __name__ == "__main__"` glue
- `tests/` — task-local behavior and pipeline-invariant tests
  (fast, no-sim; `conftest.py` + `fakes.py` provide in-process doubles)

## Public entrypoints

**Env config**:
- `make_reorient_env_cfg()` → `ManagerBasedRlEnvCfg` (robot-agnostic baseline)
- `wuji_hand_reorient_env_cfg(play: bool = False)` → Wuji Hand hardware overlay

**RL config**:
- `wuji_hand_reorient_ppo_runner_cfg()` → RSL-RL PPO runner cfg

**Eval (programmatic)**:
- `from wuji_mjlab.tasks.reorient.tooling.eval_core import EvalConfig, EvalResult, run_eval`
- Pass a populated `EvalConfig` to `run_eval()` and read back a structured
  `EvalResult` (success_rate, drop_rate, per-trial outcomes, etc.)

**Registered task IDs** (see `config/wuji_hand/__init__.py`):
- `WujiHand_Reorient` — release config (`num_envs=8192`, `max_iterations=5000`,
  ~20 GB GPU memory; reproduces the released checkpoint)
- `WujiHand_Reorient_Light` — lower-VRAM variant (`num_envs=4096`,
  `max_iterations=7500`, fits ~12 GB GPUs; weaker policy)

## Running

Train and play go through the top-level `pixi` tasks; pass the task ID
via `--task`:

```bash
pixi run train --task WujiHand_Reorient            # release config (8192 envs × 5000 iters, ~20 GB)
pixi run train --task WujiHand_Reorient_Light      # lower-VRAM variant (4096 envs × 7500 iters)
pixi run play  --task WujiHand_Reorient
pixi run list-envs                                  # show all registered task IDs
```

Common debug/eval entrypoints from this subtree:

```bash
# Visualize the task scene
python -m wuji_mjlab.tasks.reorient.scripts.view_task WujiHand_Reorient

# Evaluate ONNX policy success rate
# Default: windowed MuJoCo viewer
python -m wuji_mjlab.tasks.reorient.scripts.eval_success_rate <onnx_path>

# Headless + machine-readable JSON output (good for CI / sweeps)
python -m wuji_mjlab.tasks.reorient.scripts.eval_success_rate <onnx_path> \
    --num-trials 100 --no-viewer --json-output result.json

# Export trained policy to ONNX (see --help for checkpoint paths)
python -m wuji_mjlab.tasks.reorient.scripts.export_onnx <path-to-ckpt.pt>
```

**Programmatic eval** (no CLI, useful for batch evaluation):

```python
from pathlib import Path
from wuji_mjlab.tasks.reorient.tooling.eval_core import EvalConfig, run_eval

result = run_eval(EvalConfig(
    onnx_path=Path("logs/wuji_reorient/exp1/policy.onnx"),
    num_trials=50,
    no_viewer=True,
))
print(f"success_rate = {result.success_rate:.2%}")
print(f"mean min ori error = {result.mean_min_ori_error_rad:.3f} rad")

# Per-trial data for custom analysis
for trial in result.trials:
    if trial.status == "success":
        print(f"trial {trial.trial_idx}: t_first_succ={trial.time_to_first_success_s:.2f}s")
```

## Architecture invariants

- `mdp/event_impl/` is internal. Outside callers consume events only through
  the `mdp.events` facade or the top-level `mdp` re-export.
- `ReorientEventState` (in `mdp/event_impl/state.py`) is the single owner of
  every event-side runtime cache. Always access it via
  `get_reorient_event_state(env)`; do not stash cache fields on `env` directly.
- Curriculum functions return `dict[str, float]` to match the mjlab manager
  contract — `_get_total_training_steps` is centralized in
  `mdp/event_impl/curriculum.py` so a single rl-cfg change rescales all
  schedules.
- `tooling/` is the importable, side-effect-free core of script logic;
  `scripts/` only own argparse + env-var setup + `if __name__ == "__main__"`.
  Anything in `tooling/` can be imported and called from Python
  (e.g. for sweeps, tests, or notebooks) without invoking a CLI.
- `config/wuji_hand/` keeps RL configs under a backend-specific subdir
  (`rsl_rl/`) so the layout extends naturally if another backend is added.
- Public API surface (`make_reorient_env_cfg`, `wuji_hand_reorient_env_cfg`,
  `mdp.*` re-exports, registered task IDs, `reorient_constants.*`,
  `tooling.eval_core.EvalConfig` / `EvalResult` / `run_eval`) is
  intended to remain stable — internal modules may move, but names
  that external callers consume may not.
- The `mdp/__init__.py` uses `from .module import *` re-exports on purpose,
  so adding a new term module requires adding the corresponding wildcard
  import there (and an `__all__` in the new module).
