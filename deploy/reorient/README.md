# Reorient Real-Hand Deploy

[中文版](README_zh.md)

Sim2real bridge for `WujiHand_Reorient`: ArUco-based cube tracking with
an AprilTag-anchored wrist frame + closed-loop policy control on the
physical Wuji Hand.

## Pipeline

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

## Quick start

```bash
# 1. Install deploy environment (adds opencv, apriltag, zmq, hand driver)
pixi install -e deploy

# 2. Calibrate your physical cube and camera
#    Runtime cube geometry is loaded from deploy/reorient/config/cube_tags.json
#    (default 54 mm baseline). To use a custom cube_tags.json file, pass
#    `--cube <path>` to the vision task (`pixi run -e deploy vision -- --cube <path>`).
#    (deploy/reorient/config/cube_calibration.yaml is a human-readable
#    reference rig record only; it is NOT read at runtime.)
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

> **No trained policy?** Grab the pre-trained `policy.onnx` +
> `policy_config.json` from
> [Releases](https://github.com/wuji-technology/wuji-mjlab/releases)
> to deploy without training.

## Layout

| Path | Role |
|---|---|
| `config/` | YAML configs for camera, cube calibration, control loop |
| `lib/` | Real-hand env subclass, ZMQ bridge, ArUco/AprilTag observer, hand driver |
| `scripts/` | Entry points (`play_real.py`, `cube_world_observer.py`, etc.) |
| `tools/` | Camera calibration + cube-pose sanity check + release-cube viewer |

## Architecture

`RealHandEnv` (`lib/real_hand_env.py`) subclasses mjlab's `ManagerBasedRlEnv`
so the same observation/action managers as training apply verbatim — no
duplicated obs/action pipelines. The real hardware ↔ sim bridge happens
inside the env's `step()`: actions go to `WujiHandDriver` (real motors);
observations come from a mix of joint state (via the driver) and cube
pose (via ZMQ from `cube_world_observer.py`).

ONNX policy is loaded via `lib/onnx_policy.py`, which reads the sidecar
JSON for control-mode params (action_scale, ema_alpha, ctrl_dt,
history_len) that were captured at export time. This keeps deploy
inference identical to the sim policy that produced the ONNX.

## License

Apache 2.0. See repository root LICENSE.
