# PenSpin · Setup

[中文版](setup_zh.md) · [Task home](../README.md) · [Setup](setup.md) · [Train](train.md) · [Deploy](deploy.md)

## Hardware resources

| Resource | Location |
| --- | --- |
| Two pen bodies, end blocks and complete STEP assemblies | [Asset bundle](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip) · `pen_spin/hardware/` |
| Print projects and assembly CAD | `pen_spin/hardware/pen_heads.3mf`, `pen_shafts.3mf` and `pen_assembly.step` |
| Shared Wuji Hand 2 mounting jig | `hardware/hand2/` in the same bundle |
| Pen tag configuration | [pen_tags.json](../../../../../deploy/pen_spin/config/pen_tags.json) |
| Pen surface textures | [textures](../../../assets/objects/aruco_pen250/textures/) |
| Pen bodies and tape wrapping | [Reference photo](images/pen-body-friction-options.webp) |
| Hand, wrist tag, mount and pen | [Assembly photo](images/hand-pen-assembly.webp) |

The bundle stores shared jigs in `hardware/`, with task resources in `reorient/` and `pen_spin/`. The Hand 2 jig is shared by both tasks. See [Train](train.md) for datasets and [Deploy](deploy.md) for policies.

## 1. Rig overview

Prepare a right Wuji Hand 2, a rigid mount and baseplate, a 250 mm tagged pen, a wrist AprilTag, a Hikrobot camera and lens, a deployment computer and connecting cables. The photo shows the hand, mount, wrist tag and pen; the camera is outside this image.

<p align="center">
  <img src="images/hand-pen-assembly.webp" width="720" alt="A hand attached to a baseplate and red mount, with a wrist AprilTag and tagged pen" />
</p>

The camera must cover the pen's motion envelope and see the wrist tag while establishing the world frame. Secure the hand and mount, and route cables clear of the markers and finger motion.

## 2. Software environment

Install the separate deployment environment from the repository root:

```bash
pixi install -e pen-spin-deploy
```

The industrial camera requires the vendor MVS SDK. Refer to the [camera SDK installation guide](../../reorient/docs/setup.md#22-hikvision-mvs-sdk) for installation and environment variables, following the instructions bundled with your SDK version. Camera settings are in [`camera.yaml`](../../../../../deploy/pen_spin/config/camera.yaml); control settings are in [`control.yaml`](../../../../../deploy/pen_spin/config/control.yaml).

## 3. Tagged pen

We provide **ring-rib** and **crossed-helical-rib** pen bodies. You can also wrap tape around the gripping section to increase friction. Keep the markers at both ends uncovered.

<p align="center">
  <img src="images/pen-body-friction-options.webp" width="480" alt="Ring-rib body on the left, a tape-wrapped body in the middle, and a crossed-helical-rib body on the right" />
</p>

Use one shaft variant and the two different end blocks. The assembled pen is **250 mm** long. The bare shaft is **238 mm** long, with **178 mm** exposed after assembly. Each end block measures **18 × 18 × 36 mm**. The shaft's bounding diameter, including ribs, is approximately **16.9 mm**, distinct from the smooth core diameter used by the tracking model.

Files under `pen_spin/hardware/` in the bundle:

| File | Purpose |
|---|---|
| `assembly_ring_250mm.step` | Complete ring-rib assembly |
| `assembly_cross_250mm.step` | Complete crossed-helical-rib assembly |
| `shaft_ring_250mm.step`, `shaft_cross_250mm.step` | Alternative shafts |
| `end_block_1.step`, `end_block_2.step` | One of each end block per pen |

The `250mm` filename suffix refers to assembled length for the ring/cross STEP exports listed above. The same directory includes `pen_heads.3mf`, `pen_shafts.3mf` and `pen_assembly.step`. Use the corresponding 3MF for print geometry.

The print projects use Bambu Lab X2D profiles with a 0.4 mm nozzle, Bambu PLA Basic, 0.2 mm layers, 2 wall loops, 15% grid infill and automatic supports disabled. The filament slots are white and black; artwork uses the second slot. The head project contains one “长方柱1” and three “长方柱2” objects. Each pen uses one of each head type; adjust quantities before printing and choose one shaft variant.

### Tag patterns and placement

The pen uses the custom **`SOURCE_1X2_18`** dictionary with 18 patterns. Its IDs are not indices into a predefined ArUco dictionary; do not substitute standard markers with the same numeric IDs.

[`pen_tags.json`](../../../../../deploy/pen_spin/config/pen_tags.json) includes the pattern definitions, corners and 3D positions. The [texture directory](../../../assets/objects/aruco_pen250/textures/) contains ten patterned images and one blank image. The bundle stores tag assets under `pen_spin/hardware/tags/`: ten patterned images in `textures/`, face dimensions in `texture_faces.csv`, and marker positions in `pen_tags.json`.

The source PNGs are square textures; long side faces are physically 18 × 36 mm or 36 × 18 mm. Do not print every image as a square. Match each face's pattern, orientation and dimensions to the configuration. Use `pen_spin/hardware/tags/texture_faces.csv` for each face’s dimensions and `pen_tags.json` for marker positions.

## 4. Wrist tag and mounting orientation

Use **AprilTag36h11, ID 0, with a 50.4 mm black detection-frame edge**. This measurement excludes the outer white margin. Constants and the tag-to-wrist transform are defined in [`tag_frame.py`](../../../../../deploy/pen_spin/lib/tag_frame.py).

Use the assembly photo as a mounting reference. If you change the tag's position or orientation relative to the wrist, check and recalibrate the corresponding transform. Keep the assembly fixed after establishing the world frame.

## 5. Camera placement and lighting

Choose an oblique view that exposes multiple pen tag faces and covers the full motion envelope. Avoid persistent occlusion from fingers, the mount or cables. Adjust focus, exposure and lighting to keep moving patterns sharp.

<p align="center">
  <img src="assets/camera-placement.png" width="720" alt="Camera preview showing pen-end marker outlines, the pen axis and wrist reference tag" />
</p>

This is a reference observation view. `1 faces / COPLANAR` indicates observations from a single planar face; changing the camera angle can improve geometric constraints. Check that the outlines fit the physical pen and that the axes follow its motion.

## 6. Camera calibration and observation checks

The intrinsics and distortion in [`camera.yaml`](../../../../../deploy/pen_spin/config/camera.yaml) must match your camera, lens and capture configuration. Do not treat reference values as calibration for a different setup. Refer to [camera intrinsics calibration](../../reorient/docs/setup.md#6-camera-intrinsics-calibration) for the target and parameter workflow, then enter the results in the PenSpin configuration.

Start the observer first:

```bash
pixi run -e pen-spin-deploy pen-observer --preview
```

Keep the wrist tag visible while the world frame is sampled. Slowly move and rotate the pen, checking its axis, ends, position and outlines for persistent dropouts, jumps or flips. If you move the camera or rig, press `w` to resample and verify alignment again.

## 7. Start deployment

After checking hardware and observations, follow [Deploy](deploy.md#start-the-camera-and-policy) to start the camera observer first, then `pen-policy` once observations are ready. The policy opens its live mirror by default; no separate `pen-sim` process is required. Use a policy and reference trajectory from the same motion group. Downloads are listed in [Deploy](deploy.md) and [Train](train.md).
