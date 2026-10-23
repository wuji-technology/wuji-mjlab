# SPDX-License-Identifier: Apache-2.0
"""Record a bounded Hand 2 checkpoint rollout with per-step evidence."""

import argparse
import dataclasses
import hashlib
import json
import pathlib
import shutil
import subprocess

import mjlab.envs
import mjlab.rl
import mjlab.tasks.registry
import mjlab.utils.torch
import numpy as np
import torch
import wuji_mjlab.assets.robots.wuji_hand2.wuji_hand2_cfg as hand
import wuji_mjlab.tasks  # noqa: F401
import wuji_mjlab.tasks.pen_spin.mdp.metrics as pen_metrics
import wuji_mjlab.utils.left_hand_playback as transfer
import wuji_mjlab.utils.task_cfg_utils as task_cfg_utils


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=transfer.SUPPORTED_TASKS, required=True)
  parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
  parser.add_argument("--recipe", type=pathlib.Path)
  parser.add_argument("--output", type=pathlib.Path, required=True)
  parser.add_argument("--hand", choices=("right", "left"), default="left")
  parser.add_argument("--no-left-policy-adapter", action="store_true")
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  args = parser.parse_args()
  if args.steps <= 0 or min(args.width, args.height) <= 0:
    parser.error("steps and image dimensions must be positive")
  if args.width % 2 or args.height % 2:
    parser.error("H.264 image dimensions must be even")
  if args.no_left_policy_adapter and args.hand != "left":
    parser.error("--no-left-policy-adapter requires --hand left")
  if args.output.suffix != ".mp4":
    parser.error("--output must name an MP4 file")
  if not args.checkpoint.is_file():
    parser.error(f"Checkpoint not found: {args.checkpoint}")
  is_pen = args.task == "WujiHand2_PenSpin"
  if is_pen and (args.recipe is None or not args.recipe.is_file()):
    parser.error("PenSpin requires an existing --recipe")
  if not is_pen and args.recipe is not None:
    parser.error("--recipe applies only to PenSpin")
  ffmpeg = shutil.which("ffmpeg")
  if ffmpeg is None:
    parser.error("Install ffmpeg to record a video")
  outputs = [
    args.output,
    args.output.with_suffix(".json"),
    args.output.with_suffix(".npz"),
  ]
  if any(path.exists() for path in outputs):
    parser.error("Output already exists; choose a new --output path")
  args.output.parent.mkdir(parents=True, exist_ok=True)

  mjlab.utils.torch.configure_torch_backends()
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  overrides = ["env.scene.num_envs=1"]
  if args.recipe:
    overrides.append(f"env.commands.motion.recipe_path={args.recipe.resolve()}")
  env_cfg, agent_cfg = task_cfg_utils.prepare_task_cfgs(args.task, overrides, play=True)
  if args.hand == "left":
    env_cfg = transfer.configure_left_hand_playback(
      env_cfg, args.task, mirror_policy_observations=not args.no_left_policy_adapter
    )
  env_cfg.seed = args.seed
  env_cfg.viewer.width, env_cfg.viewer.height = args.width, args.height
  env_cfg.viewer.geom_group = (1, 1, 1, 0, 0, 0)
  env_cfg.viewer.site_group = (0, 0, 0, 0, 0, 0)
  if not is_pen:
    env_cfg.viewer.elevation, env_cfg.viewer.distance = -35.0, 0.44
  env = mjlab.envs.ManagerBasedRlEnv(
    cfg=env_cfg, device=args.device, render_mode="rgb_array"
  )
  env.max_common_steps = agent_cfg.max_iterations * agent_cfg.num_steps_per_env
  wrapped = mjlab.rl.RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  video = None
  traces = []
  total_goals, previous_goals = 0, 0
  try:
    runner_cls = (
      mjlab.tasks.registry.load_runner_cls(args.task) or mjlab.rl.MjlabOnPolicyRunner
    )
    runner = runner_cls(wrapped, dataclasses.asdict(agent_cfg), device=args.device)
    runner.load(
      str(args.checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)
    obs = wrapped.get_observations()
    renderer = env._offline_renderer
    # This model belongs only to the renderer. Simulation contact geometry stays intact.
    for geom_id in range(renderer._model.ngeom):
      geom = renderer._model.geom(geom_id)
      if geom.name.startswith("robot/") and renderer._model.geom_group[geom_id] in (
        2,
        3,
      ):
        renderer._model.geom_group[geom_id] = 3
    fps = round(1 / env.step_dt)
    video = subprocess.Popen(
      [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{args.width}x{args.height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
      ],
      stdin=subprocess.PIPE,
    )
    arrays = {key: [] for key in ("joint_pos", "object_pose_w", "actions")}
    if is_pen:
      arrays["reference_object_pose_wrist"] = []
    with torch.inference_mode():
      for step in range(args.steps):
        actions = policy(obs)
        if not torch.isfinite(actions).all():
          raise RuntimeError("Policy produced nonfinite actions")
        obs, reward, done, _ = wrapped.step(actions)
        arrays["actions"].append(actions[0].cpu().numpy().copy())
        arrays["joint_pos"].append(
          env.scene["robot"].data.joint_pos[0].cpu().numpy().copy()
        )
        arrays["object_pose_w"].append(
          env.scene["object"].data.root_link_pose_w[0].cpu().numpy().copy()
        )
        row = {
          "time_s": (step + 1) * env.step_dt,
          "reset": bool(done.any()),
          "reward": float(reward.item()),
          "terminations": {
            key: bool(env.termination_manager.get_term(key)[0])
            for key in env.termination_manager.active_terms
          },
        }
        if is_pen:
          motion = env.command_manager.get_term("motion")
          arrays["reference_object_pose_wrist"].append(
            motion.object_pose[0].cpu().numpy().copy()
          )
          row.update(
            position_error_cm=float(pen_metrics.object_pos_tracking_error_cm(env)[0]),
            shaft_error_deg=float(pen_metrics.object_ori_tracking_error_deg(env)[0]),
            trajectory_id=int(motion.trajectory_ids[0]),
            reference_frame=int(motion.time_steps[0]),
          )
        else:
          command = env.command_manager.get_term("reorient_command")
          goals = int(command.metrics["goal_reach_count"][0])
          total_goals += max(0, goals - previous_goals)
          previous_goals = goals
          row.update(
            targets_reached=total_goals,
            orientation_error_rad=float(command.metrics["ori_error"][0]),
          )
        traces.append(row)
        renderer.update(env.sim.data, debug_vis_callback=None)
        video.stdin.write(renderer.render().tobytes())
        if (step + 1) % (fps * 5) == 0:
          print(json.dumps(row), flush=True)
    video.stdin.close()
    if video.wait() != 0:
      raise RuntimeError("ffmpeg failed")
    np.savez_compressed(
      args.output.with_suffix(".npz"),
      **{key: np.stack(value) for key, value in arrays.items()},
    )
    repo = pathlib.Path(__file__).resolve().parents[2]

    def git(*argv):
      return subprocess.check_output(["git", "-C", str(repo), *argv], text=True).strip()

    metadata = {
      "task": args.task,
      "hand": args.hand,
      "mirror_policy_observations": args.hand == "left"
      and not args.no_left_policy_adapter,
      "seed": args.seed,
      "frames": args.steps,
      "fps": fps,
      "step_dt_s": env.step_dt,
      "video_duration_s": args.steps / fps,
      "resets": sum(row["reset"] for row in traces),
      "targets_reached": total_goals if not is_pen else None,
      "source_commit": git("rev-parse", "HEAD"),
      "source_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
      "description_commit": subprocess.check_output(
        ["git", "-C", str(hand.WUJI_DESCRIPTION_DIR), "rev-parse", "HEAD"], text=True
      ).strip(),
      "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
      "recipe_sha256": hashlib.sha256(args.recipe.read_bytes()).hexdigest()
      if args.recipe
      else None,
      "camera": dataclasses.asdict(env_cfg.viewer),
      "traces": traces,
    }
    args.output.with_suffix(".json").write_text(
      json.dumps(metadata, indent=2, default=str)
    )
    print(
      json.dumps(
        {
          "output": str(args.output),
          "resets": metadata["resets"],
          "targets_reached": metadata["targets_reached"],
        }
      ),
      flush=True,
    )
  finally:
    if video is not None and video.poll() is None:
      video.stdin.close()
      video.wait()
    wrapped.close()


if __name__ == "__main__":
  main()
