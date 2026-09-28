# Cube Reorientation · Deploy

[中文版](deploy_zh.md) · [Task home](../README.md) · [Setup](setup.md) · [Train](train.md) · [Deploy](deploy.md)

Run all commands from the **repository root**. Hardware and printable resources are in [Setup](setup.md); training and export are in [Train](train.md).

**Startup order: start camera observations, confirm a valid cube pose, then start the policy. The optional visual mirror runs as a separate process.** See [Start the camera and policy](#start-the-camera-and-policy) for commands.

## Checkpoint Releases

| Policy | Download | Bundle directory |
|---|---|---|
| Wuji Hand 1 · Reorient | [Asset bundle](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip) | `reorient/checkpoints/hand1/` |
| Wuji Hand 2 · Reorient | [Asset bundle](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip) | `reorient/checkpoints/hand2/` |

Standalone, **numpy + onnxruntime** runtime for the cube-reorient policy —
no torch/mjlab import at deploy time. One deploy tree supports **both**
Wuji Hand generations via `--gen {1,2}`:

| | gen1 (`wuji_hand`) | gen2 (`wuji_hand2`) |
|---|---|---|
| Driver | `wujihandpy` | `wuji_sdk` |
| Hand side | right only | right only |

Policy weights are distributed separately from the source repository in the asset bundle: [`wuji-mjlab` release `v2026.9.27`](https://github.com/wuji-technology/wuji-mjlab/releases/tag/v2026.9.27).

**`--gen 1` needs the published gen1 policy — it is not committed here; see
[Policies](#policies-not-tracked-in-git).**

## Architecture: 3 ZMQ processes

```
Terminal 1 (vision):     cube_world_observer.py  --preview
                          camera -> AprilTag/ArUco PnP -> cube pose
                          publishes on ZMQ:5555 (default)

Terminal 2 (play-real):  run_policy.py --gen N --hand H
                          cube pose (default 5555) -> obs[207] -> ONNX -> action[20]
                          -> interpolated joint targets -> hand driver
                          publishes goal quat on ZMQ:5556 (default), joints on
                          ZMQ:5557 (default)

Terminal 3 (optional):   render_viewer.py --gen N --hand H
                          subscribes to defaults 5555 (cube), 5556 (goal),
                          5557 (joints)
                          animates a passive MuJoCo scene — visualization only,
                          no policy/hardware in this process
```

The actual ports are read uniformly by all three processes from the `zmq`
section of `config/control.yaml`.

Terminal 3 is optional and purely a visual mirror; the closed loop is
terminals 1 + 2 (camera -> policy -> hand).

## Install

Single pixi environment, `reorient-deploy` (covers both generations):

```bash
pixi install -e reorient-deploy
```

One dependency has a host floor, and one is not on PyPI at all:

- **`wuji_sdk` needs glibc ≥ 2.34** on the host.
- **Hikvision MVS camera SDK** (`MvImport`) is a vendor install, not on PyPI.
  Install it from https://www.hikrobotics.com (default path `/opt/MVS`;
  override with the `MVS_PYTHON_PATH` env var if installed elsewhere). Only
  needed for `cube_world_observer.py` / `camera_calibrate.py` (the camera
  path) — `run_policy.py --mock` and `smoke_offline.py` don't need it.

## Start the camera and policy

The policy/hand scripts take `--gen {1,2}` (default 2)
and `--hand right`. The camera observer (`vision`) is gen-agnostic (no `--gen`).

Download and extract the [Asset bundle](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip). Keep each generation's `policy.onnx` and `config.json` together in its own directory; do not mix generations. Skip the download if you already extracted the bundle during setup.

```bash
gh release download v2026.9.27 --repo wuji-technology/wuji-mjlab --pattern 'wuji-mjlab-v2026.9.27-assets.zip'
unzip wuji-mjlab-v2026.9.27-assets.zip
```

### 1. Start camera observations

```bash
pixi run -e reorient-deploy vision

# use a custom cube tag configuration
pixi run -e reorient-deploy vision -- --cube path/to/cube_tags.json

```

### 2. Start policy control

```bash
pixi run -e reorient-deploy play-real -- --gen 2 --hand right \
  --ckpt wuji-mjlab-v2026.9.27-assets/reorient/checkpoints/hand2/policy.onnx
# or, no hardware:
pixi run -e reorient-deploy play-real -- --gen 2 --hand right --mock \
  --ckpt wuji-mjlab-v2026.9.27-assets/reorient/checkpoints/hand2/policy.onnx

```

### 3. Optional visual mirror

```bash
pixi run -e reorient-deploy render -- --gen 2 --hand right

```

### Additional tools

```bash
# home the hand (smooth ramp to the cage home pose)
pixi run -e reorient-deploy home -- --gen 2 --hand right

# preflight: deps, SDKs, policy IO, (optionally) hand connectivity
# (--ckpt is required — it validates the exact ONNX you'll deploy)
pixi run -e reorient-deploy check -- --gen 2 --hand right --ckpt wuji-mjlab-v2026.9.27-assets/reorient/checkpoints/hand2/policy.onnx
```

For gen1, `--connect` refuses to connect unless you explicitly add
`--allow-energize`; it does not power the motors by default. Warning: adding
`--allow-energize` powers the motors during the check.

Additional tools under `tools/`: `camera_calibrate.py` /
`view_release_cube.py` (camera intrinsics calibration and printable-cube mesh
preview; both need a display, and `camera_calibrate.py` additionally needs
the MVS SDK and a connected camera).

## Policies (not tracked in git)

Both the `*.onnx` weights and their `*.onnx.config.json` sidecar (ctrl_dt,
action_scale, ema_alpha, ...) are untracked (gitignored repo-wide) — the
exporter writes the config next to the `.onnx`, so it always travels with the
weights and never needs to live in-tree. For checkpoint export, see [Export ONNX](train.md#export-onnx).

`run_policy.py` resolves the default
checkpoint path as `policies/policy_hand{gen}_{hand}.onnx` and reads the config
sidecar from beside it (`<name>.onnx.config.json`, or a plain `config.json` in
the same dir). So either drop the exported `.onnx` + config into
`deploy/reorient/policies/`, or point `--ckpt` at any directory that already
holds both — e.g. a training run's log dir:

```bash
pixi run -e reorient-deploy play-real -- --gen 2 --hand right \
    --ckpt "logs/rsl_rl/wuji_reorient_hand2/<run>/policy_hand2_right.onnx"
```

The bundle includes the Wuji Hand 1 policy with identity metadata already embedded; no stamping step is required. Keep the camera observer running and select the Wuji Hand 1 model:

```bash
pixi run -e reorient-deploy play-real -- --gen 1 --hand right \
    --ckpt wuji-mjlab-v2026.9.27-assets/reorient/checkpoints/hand1/policy.onnx
```

For hardware assembly and calibration, see [Setup](setup.md). For the Wuji Hand 1 end-to-end checks, continue with the [smoke test below](#end-to-end-smoke-test-wuji-hand-1).

## End-to-end smoke test (Wuji Hand 1)

You now have a calibrated rig. Walk these five checkpoints in order; if
any fails, jump back to the indicated section before continuing.

### 1. Home the hand

`pixi run -e reorient-deploy home -- --gen 1 --hand right` — smooth ramp. It
prints `[home_check] gen=1 hand=right kp=5.0 kd=0.1 effort=0.5A — homing to default pose…`,
then `[home_check] holding at default pose. Ctrl+C to disable & exit.` The
script holds the default pose until Ctrl+C; visually confirm all 20 joints
are within ±2° of home. Finger stutter or hard stop → unplug and re-plug
the Hand's USB cable, then re-try.

### 2. Start the cube observer

`pixi run -e reorient-deploy vision`. Expected: OpenCV preview appears; yellow
"World Sampling: N/100" bar fills as the wrist tag stays in view; label
flips to green "WORLD FIXED" once 100 samples averaged; cube axes
overlay holds steady on a static cube.

### 3. Verify ZMQ pose stream

In a second terminal (with `vision` running), confirm cube poses are
publishing on port **5555** (`control.yaml::zmq.cube_port`):

```bash
pixi run -e reorient-deploy python - <<'EOF'
import json, zmq
sock = zmq.Context().socket(zmq.SUB)
sock.connect("tcp://localhost:5555")
sock.subscribe(b"")
sock.setsockopt(zmq.RCVTIMEO, 5000)  # fail after 5s instead of blocking forever
try:
    for _ in range(3):
        msg = json.loads(sock.recv_string())
        p = msg["cube1"]["position"]
        print(f"frame={msg['frame']:5d}  pos=({p['x']:+.3f},{p['y']:+.3f},{p['z']:+.3f})")
except zmq.Again:
    print("no pose received in 5s — is `vision` running and publishing on port 5555?")
EOF
```

You should see three fresh frame numbers and stable positions.

### 4. Visual cube-pose check

With `vision` still running:

```bash
pixi run -e reorient-deploy python deploy/reorient/scripts/render_viewer.py --gen 1 --hand right
```

Opens a MuJoCo passive viewer of the digital twin, subscribing to the
`vision` cube-pose stream (ZMQ:5555) and, if `play-real` is also
running, the live joint stream (ZMQ:5557); otherwise the hand sits at
its default pose. The cube renders at the observer's pose estimate; you
can move the physical cube freely and watch the rendered cube follow.

What this catches that the [ZMQ pose-stream check](#3-verify-zmq-pose-stream) doesn't:

- **Axis mismatches** — rotate the physical cube around one face axis
  and confirm the rendered cube rotates around the same axis. A
  mirrored or 90°-off rotation means `cube_tags.json::face_rotations`
  is wrong, or a tag was glued in the wrong orientation.
- **Position offset** — place the cube centered on the palm; the
  rendered cube should sit on the palm geom. A > 2 cm offset usually
  means [hand mounting](setup.md#51-hand-mounting) or
  [camera intrinsics](setup.md#6-camera-intrinsics-calibration) are off.
- **Pose lag or jitter** beyond what the
  [pose-estimation filter settings](setup.md#7-pose-estimation-troubleshooting) explain.

Press Ctrl+C or close the viewer to exit.

### 5. Run the closed-loop policy

The bundled Wuji Hand 1 model already contains identity metadata. Run it with its companion configuration:

```bash
pixi run -e reorient-deploy play-real -- --gen 1 --hand right \
    --ckpt wuji-mjlab-v2026.9.27-assets/reorient/checkpoints/hand1/policy.onnx
```

Expected: the ONNX policy loads and prints its sidecar config; the hand
homes via `driver.home()`. On real hardware (without `--mock`), press
ENTER at the safety prompt before the policy takes control. The optional
mirror viewer is the separate `render_viewer.py` process from the
[visual cube-pose check](#4-visual-cube-pose-check); `play-real` does not open a viewer. During control, it prints
`[run_policy] cube↔goal err=…°  cube_pos_tag=…  cube_msgs=…  reached=N timedout=N`
every 5 seconds and `[run_policy] temp: …` every 10 seconds when the
driver returns temperature data.

If the policy diverges immediately, see the [troubleshooting matrix](#troubleshooting-matrix).

### Troubleshooting matrix

| Symptom | Likely cause | Fix |
|---|---|---|
| Camera fails to open | MVS SDK not installed / `MVS_PYTHON_PATH` unset | Re-do [SDK installation](setup.md#22-hikvision-mvs-sdk); rerun the import smoke test |
| Wrist AprilTag never detected | Lighting / wrong tag family / wrong ID / wrong size | Confirm AprilTag36h11, ID 0, and tag size 50.4 mm; raise lighting |
| `World Sampling` bar never fills | Wrist tag small/blurry | Re-position so the wrist tag is ≥ 80 px wide; refocus |
| Cube observer drops cube frequently | Reprojection-error gate firing | Re-do [intrinsics calibration](setup.md#6-camera-intrinsics-calibration); verify `cube_tags.json` face mapping with the [visual cube-pose check](#4-visual-cube-pose-check) |
| Policy diverges on first step | Tag orientation mismatch found by the [visual cube-pose check](#4-visual-cube-pose-check) | Fix `cube_tags.json::face_rotations` or re-stick the offending face tag |
| Hand judders during rollout | Control-path tuning | `ctrl_dt` comes only from the ONNX sidecar; verify it and `control.yaml::control.servo_hz` independently. For gen1, the low-pass cutoff is `hand1.py`'s `lowpass_cutoff` default (3.0 Hz), with no YAML/CLI setting; gen2 uses `wuji_sdk` MIT impedance and has no such low-pass. |
| Hand stays at the default pose in `render_viewer`; status shows `joints=NO` | No process is publishing joint states on ZMQ:5557 | Start `play-real`; the configured port is `control.yaml::zmq.joint_port` |

---
