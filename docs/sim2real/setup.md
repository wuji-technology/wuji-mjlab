# Sim-to-real Setup — Wuji Hand Reorient

[中文版](setup_zh.md)

This guide walks you through the **hardware-side** setup required to run
the trained `WujiHand_Reorient` ONNX policy on a physical Wuji Hand. For
the **software pipeline** (ZMQ topology, observer architecture, ONNX policy
loading), see [`deploy/reorient/README.md`](../../deploy/reorient/README.md).

By the end of this guide you'll have:
- A calibrated camera observing the workspace at known intrinsics
- A 3D-printed ArUco-tagged cube
- A wrist-mounted AprilTag defining the world (wrist) frame
- A Wuji Hand mounted on a fixture in view of the camera
- `camera.yaml` + `cube_tags.json` populated for **your** rig
- A working `pixi run -e deploy vision` + `play-real` pipeline

<p align="center">
  <img src="../assets/deploy.gif" width="80%" alt="reference rig: this guide reproduces the setup shown above" />
</p>

## 1. Bill of materials

A reference bill of materials for the reference rig. Equivalent parts
(different vendor, similar spec sheet) work as long as section 3 and
section 5 geometry checks pass.

> The vision hardware (camera, lens, bracket) is provided **only as a
> reference configuration**. The functional requirement is that, after
> the §6 intrinsics calibration, the chosen optics can reliably estimate
> the cube pose relative to the wrist AprilTag throughout the reorient
> reachable workspace. Any sensor + lens combination meeting that bar
> works equivalently; treat the listed Hikrobot parts as a known-good
> starting point, not a hard requirement.

- **Industrial USB camera**: **Hikrobot MV-CU013-A0UC** (USB-3, 1280×1024,
  1.3 MP, color Bayer GB) — matches the sensor + capture format encoded
  in [`deploy/reorient/config/camera.yaml`](../../deploy/reorient/config/camera.yaml).
  Equivalent USB-3 industrial cameras with matching resolution and Bayer
  GB pattern (FLIR/Basler/Allied Vision counterparts) work as drop-in
  replacements; update `camera.yaml` accordingly; the camera SDK swap
  (replacing `MvImport`) is non-trivial.
  (Hikrobot is currently the only sensor SDK wired into the observer —
  see §2.2 for the SDK install. UVC webcams and competitor industrial
  cameras require rewriting the cube observer to swap `MvImport` for the
  vendor's Python binding.)
  > ⚠️ **Note**: this overrides the "drop-in" wording above — the
  > observer hard-imports `MvImport.MvCameraControl_class`, so any
  > non-Hikrobot sensor needs a code change in
  > [`deploy/reorient/scripts/cube_world_observer.py`](../../deploy/reorient/scripts/cube_world_observer.py)
  > to swap in the new vendor's Python binding before the listed
  > "drop-in" alternatives are actually drop-in.
- **FA lens**: **Hikrobot MVL-MF0824M-5MPE** — 8 mm fixed focal length,
  F2.4, 2/3″ image circle, C-mount, 5 MP rated. Pairs cleanly with the
  CU013 sensor (the larger 2/3″ image circle covers it without vignetting).
  Any 2/3″ image-circle C-mount lens with 8 mm focal length and F2.4 or
  wider aperture works equivalently.
- **Camera mounting bracket / tripod**: any rigid fixture that holds the
  camera ~350 mm above the hand palm with no creep between calibration
  (section 6) and rollout. Requirements: 1/4"-20 standard tripod thread or
  equivalent C-mount bracket; vertical reach ≥ 400 mm; vibration-damped
  (no flex under USB cable tension); fixed-height clamp preferred over
  servo-actuated arms.
- **Wrist AprilTag sticker**: 1 × AprilTag36h11 ID 0 at 48 mm × 48 mm outer
  (matches the hardcoded `WORLD_TAG_SIZE = 0.048` in
  [`deploy/reorient/scripts/cube_world_observer.py`](../../deploy/reorient/scripts/cube_world_observer.py);
  this constant is not exposed via yaml, so any change must be made in the
  script). Print on matte vinyl
  or laminated paper to avoid camera glare; black ink on white background;
  validate the printed outer dimension with a caliper before mounting —
  any scaling error propagates directly into pose estimation. See §4 for
  the print/buy workflow and the exact dimension convention.
  (The 24 cube-face ArUco tiles are *not* stickers — they are baked into
  the shipped Bambu Lab `.3mf` via dual-material printing; see section 3.1.)
- **Wuji Hand right-hand**. Contact Wuji Technology directly;
  `wujihandpy==1.5.1` expects the Wuji Hand firmware revision matching
  `lib/hand_driver.py`. The host connects via a single USB cable (the hand
  exposes a USB CDC interface on STMicroelectronics vendor ID 0483).
- **Hand mounting jig** — 3D-printed PLA base bolted to an aluminum
  honeycomb breadboard. Detailed BOM and assembly in section 5.1; CAD
  shipped with the release attachment (see [Releases](https://github.com/wuji-technology/wuji-mjlab/releases)).
- **Instrumented cube** — 3D-printed 54 mm edge solid with 24 ArUco
  tags baked into the faces (matches `cube_tags.json`). Fabrication
  details in section 3; CAD shipped with the release attachment (see
  [Releases](https://github.com/wuji-technology/wuji-mjlab/releases)).
- **Computer**: Ubuntu 22.04 x86_64, NVIDIA sm_80+ GPU (Ampere+), CUDA
  12.8, at least 2 free USB ports (one for the camera, one for the Hand).

> All Wuji-fabricated parts (cube, jig) are open source under Apache 2.0;
> commercial alternatives work as long as cube edge = 54 mm and tag sizes
> match [`deploy/reorient/config/cube_tags.json`](../../deploy/reorient/config/cube_tags.json).

## 2. Software prerequisites

### 2.1 OS and GPU drivers

- Ubuntu 22.04 LTS, x86_64.
- NVIDIA driver bundled with CUDA 12.8 (`nvidia-smi` should report it).
  Older drivers may work for inference-only but are not tested.
- [pixi](https://pixi.sh) ≥ 0.66 (the version CI uses) in your `$PATH`.

### 2.2 Hikvision MVS SDK

`tools/camera_calibrate.py` and `scripts/cube_world_observer.py` import
`MvImport.MvCameraControl_class` from a system-level SDK install (not
vendored in this repo).

**Where to get it**: <https://www.hikrobotics.com> → Service & Support →
Downloads → MVS Client → Linux x86_64. (Switch to English in the top-right
if your locale lands you on the Chinese page.)

**Recommended version**: MVS Client **≥ 4.6.0** for Linux. The maintainer
rig runs `4.6.3` (bundles Machine Vision Camera SDK
`4.7.1.1`); older 4.5.x versions ship slightly different Python bindings
and may break the imports.

**Installing**. The exact package format varies by platform and MVS
release — **always defer to the README bundled inside the SDK archive**,
since official commands change across MVS Client minor versions and
across distros. Common cases:

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

All of the above place files under `/opt/MVS/` by default. After install,
you should have:

- `/opt/MVS/lib/64/libMvCameraControl.so` — runtime shared library
- `/opt/MVS/Samples/64/Python/MvImport/` — Python bindings
- `/opt/MVS/bin/MVS` — GUI for device discovery and live preview

**Post-install system tuning** (required to sustain the 90 FPS capture
configured in `camera.yaml`):

```bash
# USB-3 cameras: install udev rules + raise USB scheduling priority
sudo /opt/MVS/bin/set_usb_priority.sh

# GigE cameras only: raise kernel socket buffer to prevent frame drops
sudo /opt/MVS/bin/set_socket_buffer_size.sh
```

**Shell environment**. The MVS installer writes its exports to
`/etc/profile.d/MVS_*.sh`, but that file is only loaded by **login**
shells. zsh and most terminal-launched bash sessions are non-login, so
the variables are silently missing and the Python binding will throw
`TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'` on
import. Make the exports stick by appending them to your shell rc:

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

| Variable | Role |
|---|---|
| `MVCAM_COMMON_RUNENV` | Read by the Hikvision Python binding to locate `libMvCameraControl.so`. |
| `LD_LIBRARY_PATH` | Linux dynamic linker search path; required so the MVS shared libraries' transitive deps resolve. |

Confirm:

```bash
echo $MVCAM_COMMON_RUNENV          # /opt/MVS/lib
echo $LD_LIBRARY_PATH | tr ':' '\n' | grep MVS   # contains /opt/MVS/lib/64
```

**Verify the install**:

```bash
# Python binding import test
python3 -c "import sys; sys.path.insert(0, '/opt/MVS/Samples/64/Python'); from MvImport.MvCameraControl_class import *; print('ok')"

# Hardware detection — your camera should appear in the left panel
/opt/MVS/bin/MVS
```

Troubleshooting:

- `ModuleNotFoundError: MvImport` — SDK path is wrong; either reinstall to
  `/opt/MVS/` or set `MVS_PYTHON_PATH=/path/to/MVS/Samples/64/Python`.
- `TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'` on
  `MvCameraControl_class` import — `MVCAM_COMMON_RUNENV` is unset; see
  the "Shell environment" subsection above.
- GUI lists no camera — re-run `set_usb_priority.sh`, check cable, add your
  user to `plugdev`/`dialout` groups, and confirm hardware visibility with
  `lsusb | grep -i hikvis` (USB-3) or `arp -a | grep -i hikvis` (GigE).

### 2.3 Deploy environment

`pixi install -e deploy` from the repo root pulls in (see
[`pixi.toml`](../../pixi.toml) `[feature.deploy.pypi-dependencies]`):
`opencv-contrib-python>=4.13` (ArUco + IPPE), `pupil-apriltags>=1.0`
(wrist tag), `pyzmq>=27.0` (cube/goal pub-sub), `glfw>=2.10` (passive
MuJoCo viewer), `wujihandpy==1.5.1` (Wuji Hand driver), `pyyaml>=6.0`.
Smoke-test:

```bash
pixi run -e deploy python -c "import cv2, pupil_apriltags, zmq, wujihandpy; print(cv2.__version__)"
```

## 3. Cube fabrication

The fastest path is to reproduce the reference cube from the
release-bundled assets. Download the release zip:

```bash
# Requires the GitHub CLI (https://cli.github.com). The glob keeps this
# command working unchanged across future release tags.
gh release download --repo wuji-technology/wuji-mjlab --pattern '*-assets.zip'
unzip wuji-mjlab-*-assets.zip
mv wuji-mjlab-*-assets release-assets
```

This produces a `release-assets/` directory containing
`hardware/cube/{cube.3mf, cube.step, cube.obj, cube.mtl, cube.png}` plus
`checkpoints/`, `hand-jig/`, etc. (the directory and the source zip are
both gitignored). The cube is a 54 mm edge solid with 4 × 13 mm ArUco
tags per face (24 tags total, IDs 0–23). Real performance depends on
dimensions matching `cube_tags.json` to within ~0.5 mm.

> Without the `gh` CLI, browse to the
> [latest release page](https://github.com/wuji-technology/wuji-mjlab/releases/latest),
> download the `wuji-mjlab-v*-assets.zip` attachment manually, then run
> the same `unzip` + `mv` pair from this directory.

To verify the 24-tag layout in 3D before committing to a print, use the
shipped viewer script (defaults to `release-assets/hardware/cube/cube.obj`
in cwd):

```bash
pixi run python deploy/reorient/tools/view_release_cube.py
```

### 3.1 3D-printing the cube

**Use the shipped Bambu Lab `.3mf`** (in the release attachment
at `release-assets/hardware/cube/cube.3mf`) — it contains the cube geometry plus
the dual-material assignments that print the 24 ArUco 4×4 tag
patterns directly into the faces. No stickers, no glue, no
alignment fuss.

Workflow:

1. Open the file in **Bambu Studio**, or drag it onto a connected Bambu
   printer (Bambu's printer firmware can slice `.3mf` files in place).
2. Load **two filaments**: one **black**, one **white** (PLA is fine).
   The slicer will prompt for AMS / external-spool slot assignment — the
   `.3mf` declares which logical slot is "tag" vs "base", confirm the
   physical filaments match.
3. Slice and print. Default settings (~0.2 mm layer, ~30 % infill) work;
   nothing in the policy depends on internal cube density.

Expected geometry (matches [`cube_tags.json`](../../deploy/reorient/config/cube_tags.json),
do **not** rescale): **54 mm cube edge**, 13 mm tag tiles, tags centred
18 mm from face centre along each face's local u/v axes.

> Multi-material prints are noticeably more failure-prone than single-
> material — expect a couple of throw-away first prints while you dial
> in AMS flushing volume and inter-color purge.

Tag IDs follow the layout in
[`deploy/reorient/config/cube_tags.json`](../../deploy/reorient/config/cube_tags.json):
TOP holds IDs 0–3, BOTTOM 8–11, FRONT 20–23, BACK 16–19, LEFT 12–15,
RIGHT 4–7 (4 tags per face, each ~13 mm).

**Single-material fallback.** If you don't have a dual-material 3D
printer, you can produce a functional cube from `cube.step` (or
`cube.obj`) plus printed stickers:

1. Print the cube body in any single material (PLA/PETG/etc.) from
   `cube.step`. Verify the 54 mm edge with a caliper after print.
2. Print `cube.png` on matte vinyl or laminated paper at the exact
   UV-mapped size. The texture lays out the 24 tags in a 6-face
   unwrap; cut along the face boundaries to get six 54 × 54 mm
   decals.
3. Apply each decal to its corresponding face. The cube body has no
   intrinsic orientation, but tag IDs encode the face — match the
   `cube_tags.json` mapping shown above when placing decals (e.g.
   tags 0–3 go on TOP).
4. Recalibrate `tag_center_offset` in `cube_tags.json` if your
   decals don't sit exactly 18 mm from the face centre (the default
   18 mm assumes the laid-out grid).

Stickered cubes are functionally equivalent to dual-material cubes
at the ArUco-detection level, but more prone to peeling and
alignment errors. Prefer `.3mf` if available.

> **Note on cube colors.** The 6-color face background
> (blue/green/red/purple/teal/black) in `cube.png` and `cube.3mf` is
> purely a visual aid to distinguish faces by eye — the
> AprilTag/ArUco detector reads only the black-and-white tag patterns,
> and the observer applies a BGR min-channel grayscale (§5.3) that
> collapses any colored background to near-zero. If you stickerize,
> feel free to use a plain white background as long as the tag pattern
> itself is high-contrast black on white.

### 3.2 Tag specs

From `cube_tags.json` (these are vision-only; only change if your printed
cube uses different geometry, not because you're retraining the policy):
dictionary
**ArUco 4×4** (`cv2.aruco.DICT_4X4_50`); `tag_size: 0.013` (13 mm);
`tag_center_offset: 0.018` (18 mm from face center); `face_rotations`
TOP 0° / BOTTOM 180° / FRONT/BACK/LEFT 90° / RIGHT 270°.

Per-face tag IDs (T/R/B/L = top/right/bottom/left slots within the face,
see `cube_tags.json::faces_config`):

| Face    | T  | R  | B  | L  |
|---------|----|----|----|----|
| TOP     |  0 |  2 |  3 |  1 |
| BOTTOM  | 11 |  9 |  8 | 10 |
| FRONT   | 22 | 23 | 21 | 20 |
| BACK    | 18 | 19 | 17 | 16 |
| LEFT    | 14 | 15 | 13 | 12 |
| RIGHT   |  5 |  4 |  6 |  7 |

### 3.3 Per-face axes (reference)

Tags come pre-baked into the `.3mf`, so there is nothing to glue. The
table below is the cube-frame axis convention used both by the shipped
`.3mf` and by the ArUco board builder in
`cube_world_observer.py::_build_aruco_board` — keep it in sync if you
ever regenerate the `.3mf` from scratch (e.g. to retarget a different
edge length or tag dictionary).

For a visual reference, run `pixi run python deploy/reorient/tools/view_release_cube.py`
(see §3 introduction); the per-face ID table above gives the same
information textually.

Each face's local axes (from `cube_tags.json::face_axes`):

| Face    | center (cube frame) | u-axis    | v-axis     |
|---------|---------------------|-----------|------------|
| TOP     | `[ 0,  0,  1]`      | `[1,0,0]` | `[0,1,0]`  |
| BOTTOM  | `[ 0,  0, -1]`      | `[1,0,0]` | `[0,-1,0]` |
| FRONT   | `[ 0, -1,  0]`      | `[1,0,0]` | `[0,0,1]`  |
| BACK    | `[ 0,  1,  0]`      | `[-1,0,0]`| `[0,0,1]`  |
| LEFT    | `[-1,  0,  0]`      | `[0,-1,0]`| `[0,0,1]`  |
| RIGHT   | `[ 1,  0,  0]`      | `[0,1,0]` | `[0,0,1]`  |

The four tags per face sit at ±`tag_center_offset` along u (L/R) and v
(T/B). If you regenerate the `.3mf`, keep each tile's "up" aligned with
the face's v-axis — IPPE absorbs small in-plane rotations, but a tile
rotated 90° causes the dominant-face logic in `detect_cube_pose` to
reject that face entirely.

## 4. Wrist AprilTag

The cube observer defines the world (wrist) frame from a single
AprilTag36h11 marker rigidly mounted to the wrist plate
(`cube_world_observer.py`: "World frame defined by AprilTag ID 0").
Required specs (hardcoded): family **AprilTag36h11**, ID **0**, edge
**48 mm** (`WORLD_TAG_SIZE = 0.048`). Print at exactly 48 mm × 48 mm
outer (the AprilTag library uses this as the metric scale; printing scale
errors propagate into pose estimation).

The tag plane sits on the back of the wrist, perpendicular to the palm
normal. The observer hardcodes a pure handedness flip in
`WORLD_FRAME_CORRECTION`, so the printed jig must place the tag at the
same orientation as the reference rig.
![Wrist tag mounting — AprilTag visible at the top of the assembled jig](images/hand-jig-side.jpg)

> **Warning.** Do not move the wrist tag after world-frame sampling. The
> observer averages 100 frames on startup then freezes the world pose
> (`_finalize_world_frame`); any later shift corrupts the cube-in-tag
> observation the policy reads.

### 4.1 Print or buy

The factory-bundled hardware kit does **not** include a pre-cut wrist
sticker. DIY printing is the recommended path because vendors rarely
offer single-tag custom sizes — most off-the-shelf AprilTag sticker
packs ship a full ID range at one fixed size, and you only need ID 0 at
exactly 48 mm. Off-the-shelf options do exist (search e.g. "AprilTag36h11
sticker 48mm" on Amazon / AliExpress / Taobao); verify the seller's
stated outer dimension matches the 48 mm convention below before buying.

**Dimension convention.** AprilTag36h11 tiles have a 10 × 10 cell grid
(8 × 8 data + 1-cell solid-black border on each side). The 48 mm spec
is the **outer edge of the black border** — i.e. the distance between
the two outer black edges of opposite sides, which is also what
`pupil-apriltags` treats as `tag_size`. The white **quiet zone** (≥ 1
cell ≈ 4.8 mm) sits **outside** the 48 mm; it is not part of the
metric tag but the detector needs it to find the corner gradient. So a
correctly printed sticker is ~58 mm × 58 mm total (48 mm black + ≥ 5 mm
white margin on each side).

At 48 mm outer → 4.8 mm per cell → ~113 px per cell at 600 dpi (plenty
of margin for clean corner detection).

**DIY print workflow.**

1. **Source the tag image.** Official PNGs live at
   [`AprilRobotics/apriltag-imgs`](https://github.com/AprilRobotics/apriltag-imgs);
   ID 0 of family 36h11 is at
   [`tag36h11/tag36_11_00000.png`](https://github.com/AprilRobotics/apriltag-imgs/blob/master/tag36h11/tag36_11_00000.png).
   It is a 10 × 10 px raster. The same repo's `tag_to_svg.py` produces a
   vector version at any size (preferred if your printer driver accepts
   SVG):

   ```bash
   git clone https://github.com/AprilRobotics/apriltag-imgs
   cd apriltag-imgs
   python3 tag_to_svg.py tag36h11/tag36_11_00000.png tag36_11_00000.svg --size=48mm
   ```

2. **Scale without antialiasing.** If you stay with the PNG, upscale
   from 10 × 10 px to the target print size using **nearest-neighbor**
   interpolation — antialiased edges blur the corner gradient and
   degrade pose estimation. ImageMagick:

   ```bash
   # 10 px → 1134 px (48 mm at 600 dpi); nearest-neighbor via -filter point
   convert tag36h11/tag36_11_00000.png -filter point -resize 11340% tag_48mm_600dpi.png
   ```

   Or in GIMP / Photoshop: Image → Scale, Interpolation = "None" /
   "Nearest neighbor".

3. **Lay out with quiet zone.** Place the 48 mm tile on a white page
   with ≥ 5 mm of pure white margin on all four sides (1 cell width).
   Do not crop the white border tight to the black square — the
   detector will lose the outer gradient.

4. **Print.** ≥ 600 dpi, matte vinyl or laminated matte paper, black
   toner / pigment ink on white. Avoid glossy stock (glare under the
   industrial LED lighting kills detection).

5. **Verify with caliper.** Measure the black square's outer edge —
   not the page — and confirm 48.0 ± 0.3 mm on both axes. Anything
   outside that band scales pose-estimation errors linearly.

6. **Mount.** Apply to the wrist plate at the orientation shown in the
   image above. Re-read the §4 warning before re-running `vision`.

## 5. Physical assembly

### 5.1 Hand mounting

The Wuji Hand sits on a 3D-printed jig bolted to an aluminum honeycomb
breadboard. The jig exposes the wrist AprilTag to the camera and gives
the cube ~20 cm of clear space above the palm. Route the Hand's USB cable
behind the wrist, out of the camera's field of view.

![Hand on jig, side view — assembled Wuji Hand on the 3D-printed jig with the wrist AprilTag mounted on top](images/hand-jig-side.jpg)

**Bill of materials**:

| # | Part / spec | Component | Qty | Material | Finish | Type |
|---|---|---|---|---|---|---|
| 1 | 350 × 200 × 13 mm | Aluminum honeycomb breadboard | 1 | AL6061-T6 | Anodized black | Off-the-shelf |
| 2 | see `base.3mf` (release attachment) | 3D-printed base | 1 | PLA | — | Print |

Plus 4× M6 socket-head screws (length depending on breadboard
thickness — 16 mm typical) for fixing the base to the breadboard.

**Assembly**:
1. Print `base.3mf` (from the release attachment) on a PLA-capable
   FDM printer — Bambu Lab profile is bundled in the file.
2. Place the base on the aluminum honeycomb breadboard with the
   wrist-mount cradle facing forward. The base has four φ6.60 mm
   through-holes + φ11 mm counterbores aligned with the M6 thread
   grid on the breadboard.
3. Bolt the base down with four M6 socket-head screws through the
   counterbores into the breadboard.
4. The assembled stack is about 147 mm tall and tilts the Wuji Hand
   back by 10° so the wrist tag faces the camera at rest.
5. Strap the Wuji Hand into the cradle; route the Hand's USB cable
   behind the wrist out of camera view.

### 5.2 Camera mounting

Mount the camera so the entire cube reachable workspace and the wrist
AprilTag both stay inside the preview throughout reorientation. In
practice this is roughly 30-40 cm above the palm, but the exact
distance is not critical — the policy keeps the cube within ~10 cm of
palm centre during a rollout, so the workspace box is small (roughly
20 × 20 cm above the palm) and never leaves the frame mid-episode.
What matters is (a) the wrist tag is visible at rest, and (b) the
cube and its reachable workspace fit comfortably inside the ROI you
select below.

**Don't hand-edit `fast_roi`** in
[`camera.yaml`](../../deploy/reorient/config/camera.yaml) — the vision
program ships an interactive selector. With the camera mounted:

```bash
pixi run -e deploy vision
```

In the OpenCV preview window, press **`s`** to open the ROI selection
dialog. Drag a rectangle around the **cube's reachable workspace**
(this is the per-frame detection ROI that the observer crops to before
running ArUco). The wrist AprilTag doesn't need to stay inside the
ROI — it only needs to be visible during the 100-frame world-frame
sampling at launch (and again whenever you press `w`; see below).
Press ENTER / SPACE to confirm (`C` cancels). `cube_world_observer.py`:

- snaps width / height / offsets to multiples of 8 (Hikvision sensors
  require this) and enforces a 64 px minimum;
- writes the new values to `config/camera.yaml::fast_roi` atomically
  (load → modify → temp file → rename);
- applies them live without restarting capture.

Shipped defaults describe the reference rig and will be overwritten
on first `s`-save:

```yaml
fast_roi:
  offset_x: 464
  offset_y: 112
  width:    616
  height:   504
```

Other vision-window hotkeys:

| Key | Action |
|---|---|
| `s` | Open ROI selector (above) |
| `w` | Resample the world frame (re-detects wrist AprilTag, resets cube filters) |
| `r` | Reset cube filters only (world frame untouched) |
| `q` | Quit |

The observer also switches to the configured `fast_roi` automatically
when launched headless (no `--preview`); preview mode keeps the full ROI
visible so you can re-frame mid-session.

**When to resample the world frame.** On every `vision` launch the
observer auto-samples 100 frames of the wrist AprilTag and freezes the
world pose (`_finalize_world_frame`). Press **`w`** to redo this if:

- You re-mounted the hand jig and the wrist tag moved (even by 1 mm).
- Cube pose estimates start drifting or jittering noticeably relative
  to the visible cube on the hand.
- You changed the camera position, focus, or `fast_roi`.

Pressing `w` re-detects the tag, runs a fresh 100-frame average, and
resets cube filters. **Avoid `w` mid-rollout** — the policy reads
cube-in-tag observations relative to the world frame, so a sudden
frame shift will corrupt the running episode.

### 5.3 Lighting

Use diffuse ambient light. Avoid backlighting — the observer uses a
min-channel-from-BGR grayscale (white → 255, colored → ~0), so washed-out
tag edges are the single biggest cause of detection dropouts. Leave
CLAHE on (`enable_clahe: true` in
[`observer.yaml`](../../deploy/reorient/config/observer.yaml)); if cube
faces look "noisy" under CLAHE, disable it and rely on min-channel only.

## 6. Camera intrinsics calibration

`fx`, `fy`, `cx`, `cy` and the 5-parameter Brown-Conrady distortion in
[`camera.yaml`](../../deploy/reorient/config/camera.yaml) describe the
reference rig. For a different camera you **must** re-calibrate
before trusting any cube pose — a 5 % focal-length error propagates linearly to cube position.

### 6.1 Print the chessboard

11 × 8 **inner** corners (12 × 9 squares), 20 mm squares to match the
calibrator's `SQUARE_SIZE = 0.020` constant (adjust the constant if you
print larger). Mount on rigid flat backing — bowing introduces a
systematic radial bias.

### 6.2 Run the guided calibrator

```bash
pixi run -e deploy python deploy/reorient/tools/camera_calibrate.py
```

The tool walks through 14 capture tasks (center / left / right / top /
bottom regions; near / mid / far distance; straight / tilted attitude)
and auto-captures when region/size/tilt are in range, 5 stable frames
have passed, and the Laplacian quality score ≥ 60.

Keys: `c` force-capture, `n` skip, `s` fit (needs ≥ 12 captures), `q`
quit. After `s` the tool prints RMS reprojection error and writes
`deploy/reorient/config/camera_calibration.npz`. Aim for RMS < 0.5 px;
> 1.0 px indicates board motion or out-of-focus capture.

### 6.3 Populate camera.yaml

The calibrator writes `camera_calibration.npz` but does not update
`camera.yaml` in-place — copy the printed `K` and `dist` values into the
`intrinsics` and `distortion` blocks by hand. Diff against the shipped
file to confirm all 9 numbers (fx, fy, cx, cy, k1, k2, p1, p2, k3) are
populated.

### 6.4 Sanity check

`pixi run -e deploy vision` (preview mode). The wrist-tag pose should
hold steady to sub-pixel jitter when held still. The observer rejects
PnP fits with mean reprojection error > 6.0 px
(`observer.yaml::pnp.reproj_threshold`); frequent rejections mean
intrinsics are under-fit — return to section 6.2 and capture more tilted
samples.

## 7. Pose-estimation tuning

After hardware is fixed, [`observer.yaml`](../../deploy/reorient/config/observer.yaml)
gives you four knobs to trade noise vs. lag.

### 7.1 Parameters

- `rotation_filter.process_noise` (default 0.5) — SO(3) Kalman process
  noise; higher = more agile + noisier.
- `rotation_filter.measurement_noise` (default 0.1) — lower = trust PnP
  more.
- `position_filter.alpha` (default 0.8) — low-pass in [0, 1]; 1.0 = no
  filter.
- `pnp.reproj_threshold` (default 6.0 px) — fits above this are dropped
  (cube goes "lost"; filters reset on reacquire).
- `preprocess.enable_clahe` / `clahe_clip` / `clahe_tile` — disable
  under high-contrast lighting where CLAHE adds noise.

### 7.2 Presets

Embedded verbatim in `observer.yaml`:

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

The shipped default is the agile preset (this is the configuration the
trained policy was deployed against).

### 7.3 Troubleshooting

- Cube jitter when held still → smooth preset.
- Pose lag during fast reorientation → agile preset.
- Cube drops to "lost" repeatedly → raise `pnp.reproj_threshold` to e.g.
  8.0 px **and** re-check section 6 intrinsics; if axes look swapped,
  fix `cube_tags.json::face_rotations` or re-stick the offending tag.

## 8. End-to-end smoke test

You now have a calibrated rig. Walk these five checkpoints in order; if
any fails, jump back to the indicated section before continuing.

### 8.1 Step 1 — home the hand

`pixi run -e deploy home` — 3 s smooth ramp; 20 joints land within ±2°
of home; script prints "Within 2° — home reached". Finger stutter or
hard stop → unplug and re-plug the Hand's USB cable, then re-try.

### 8.2 Step 2 — start the cube observer

`pixi run -e deploy vision`. Expected: OpenCV preview appears; yellow
"World Sampling: N/100" bar fills as the wrist tag stays in view; label
flips to green "WORLD FIXED" once 100 samples averaged; cube axes
overlay holds steady on a static cube.

### 8.3 Step 3 — verify ZMQ pose stream

In a second terminal (with `vision` running), confirm cube poses are
publishing on port **5555** (`control.yaml::zmq.cube_port`):

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

You should see three fresh frame numbers and stable positions.

### 8.4 Step 4 — visual cube-pose check

With `vision` still running:

```bash
pixi run -e deploy python deploy/reorient/tools/calib_check.py
```

Opens a MuJoCo passive viewer of the digital twin — the hand mirrors
live encoder readings, the cube renders at the observer's pose
estimate. The hand homes once on start and then stays put; you can move
the physical cube freely and watch the rendered cube follow.

What this catches that 8.3 doesn't:

- **Axis mismatches** — rotate the physical cube around one face axis
  and confirm the rendered cube rotates around the same axis. A
  mirrored or 90°-off rotation means `cube_tags.json::face_rotations`
  is wrong, or a tag was glued in the wrong orientation.
- **Position offset** — place the cube centered on the palm; the
  rendered cube should sit on the palm geom. A > 2 cm offset usually
  means hand mounting (section 5.1) or camera intrinsics (section 6)
  are off.
- **Pose lag or jitter** beyond what the section 7.1 filter knobs
  explain.

Press Ctrl+C or close the viewer to exit.

### 8.5 Step 5 — run the closed-loop policy

`pixi run -e deploy play-real --ckpt <path-to.onnx>`. Expected: ONNX
policy loads + prints sidecar JSON; hand homes via `env.reset()`; a
passive MuJoCo "mirror" viewer opens (real joints + observed cube +
translucent goal cube 10 cm above); the hand reorients the cube toward
the goal, with benchmark trial outcomes printed inline.

If the policy diverges immediately, see section 8.6.

### 8.6 Troubleshooting matrix

| Symptom | Likely cause | Fix |
|---|---|---|
| Camera fails to open | MVS SDK not installed / `MVS_PYTHON_PATH` unset | Re-do section 2.2; rerun the import smoke test |
| Wrist AprilTag never detected | Lighting / wrong tag family / wrong ID / wrong size | Confirm AprilTag36h11, ID 0, 48 mm; raise lighting |
| `World Sampling` bar never fills | Wrist tag small/blurry | Re-position so the wrist tag is ≥ 80 px wide; refocus |
| Cube observer drops cube frequently | Reprojection-error gate firing | Re-do section 6 intrinsics; verify `cube_tags.json` face mapping by running section 8.4 |
| Policy diverges on first step | Tag orientation mismatch found by section 8.4 | Fix `cube_tags.json::face_rotations` or re-stick the offending face tag |
| Hand judders during rollout | `control_dt` mismatch policy ↔ hardware | Check ONNX sidecar `ctrl_dt`; drop `control.yaml::hardware.lowpass_cutoff_hz` |
| Mirror viewer renders frozen pose | Stale `mj_data` template | Restart `play-real` (viewer attaches via `_viz_mj_data` on launch) |

---

License: Apache 2.0. See repository root [`LICENSE`](../../LICENSE).
