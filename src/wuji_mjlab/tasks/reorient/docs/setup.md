# Cube Reorientation · Setup

[中文版](setup_zh.md) · [Task home](../README.md) · [Setup](setup.md) · [Train](train.md) · [Deploy](deploy.md)

## Hardware resources

| Resource | Download | Bundle directory |
|---|---|---|
| Cube models, print files and textures | [Asset bundle](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip) | `reorient/hardware/` |
| Wuji Hand 1 mounting jig | [Asset bundle](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip) | `hardware/hand1/` |
| Wuji Hand 2 mounting jig | [Asset bundle](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.9.27/wuji-mjlab-v2026.9.27-assets.zip) | `hardware/hand2/` |

The existing hardware walkthrough describes the Wuji Hand 1 reference rig. Checkpoint Release links are maintained in [Deploy](deploy.md).

<p align="center">
  <img src="assets/deploy.gif" width="80%" alt="Cube reorientation reference rig" />
</p>

The shared jigs are under `hardware/hand1/` and `hardware/hand2/` at the bundle root. The Hand 2 jig is shared by Reorient and PenSpin. Use the matching drawings and base for Wuji Hand 2; the detailed dimensions below describe the Wuji Hand 1 reference rig.

## 1. Bill of materials

A reference bill of materials for the reference rig. Equivalent parts
(different vendor, similar spec sheet) work as long as section 3 and
section 5 geometry checks pass.

> The vision hardware (camera, lens, bracket) is provided **only as a
> reference configuration**. The functional requirement is that, after
> the §6 intrinsics calibration, the chosen optics can reliably estimate
> the cube pose relative to the wrist AprilTag throughout the reorient
> reachable workspace.

- **Industrial USB camera**: **Hikrobot MV-CU013-A0UC** (USB-3, 1280×1024,
  1.3 MP, color Bayer GB) — matches the sensor + capture format encoded
  in [`deploy/reorient/config/camera.yaml`](../../../../../deploy/reorient/config/camera.yaml).
  > The observer hard-imports `MvImport.MvCameraControl_class`, so any
  > non-Hikrobot sensor needs a code change in
  > [`deploy/reorient/scripts/cube_world_observer.py`](../../../../../deploy/reorient/scripts/cube_world_observer.py)
  > to swap in the new vendor's Python binding. See §2.2 for the
  > Hikrobot SDK install.
- **FA lens**: **Hikrobot MVL-MF0824M-5MPE** — 8 mm fixed focal length,
  F2.4, 2/3″ image circle, C-mount, 5 MP rated.
  Any 2/3″ image-circle C-mount lens with 8 mm focal length and F2.4 or
  wider aperture works equivalently.
- **Camera mounting bracket / tripod**: any rigid fixture that holds the
  camera ~350 mm above the hand palm with no creep between calibration
  (section 6) and rollout. Requirements: 1/4"-20 standard tripod thread or
  equivalent C-mount bracket; vertical reach ≥ 400 mm; vibration-damped
  (no flex under USB cable tension); fixed-height clamp preferred over
  servo-actuated arms.
- **Wrist AprilTag sticker**: 1 × AprilTag36h11 ID 0;
  **tag size 50.4 mm** (matches the hardcoded `WORLD_TAG_SIZE = 0.0504`
  in
  [`deploy/reorient/scripts/cube_world_observer.py`](../../../../../deploy/reorient/scripts/cube_world_observer.py);
  this constant is not exposed via yaml, so any change must be made in
  the script). Print on matte vinyl or laminated paper to avoid camera
  glare; black ink on white background; validate the printed
  tag size with a caliper before mounting — any scaling error
  propagates directly into pose estimation. See §4 for the print/buy
  workflow and the exact dimension convention.
  (The 24 cube-face ArUco tiles are *not* stickers — they are baked into
  the shipped Bambu Lab `.3mf` via dual-material printing; see section 3.1.)
- **Wuji Hand right-hand**. Contact Wuji Technology directly;
  `wujihandpy==1.5.1` expects the Wuji Hand firmware revision matching
  `wuji_reorient_deploy/drivers/hand1.py`. The host connects via a single USB cable (the hand
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
> match [`deploy/reorient/config/cube_tags.json`](../../../../../deploy/reorient/config/cube_tags.json).

## 2. Software prerequisites

### 2.1 OS and GPU drivers

- Ubuntu 22.04 LTS, x86_64.
- NVIDIA driver bundled with CUDA 12.8 (`nvidia-smi` should report it).
- [pixi](https://pixi.sh) ≥ 0.72 in your `$PATH`.

### 2.2 Hikvision MVS SDK

`tools/camera_calibrate.py` and `scripts/cube_world_observer.py` import
`MvImport.MvCameraControl_class` from a system-level SDK install.

**Where to get it**: <https://www.hikrobotics.com> → Service & Support →
Downloads → MVS Client → Linux x86_64. (Switch to English in the top-right
if your locale lands you on the Chinese page.)

**Recommended version**: MVS Client **≥ 4.6.0** for Linux; older 4.5.x versions ship slightly different Python bindings and may break the imports.

**Installing**. **Always defer to the README bundled inside the SDK archive**,
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
import. Set the variables in the current terminal first:

```bash
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export LD_LIBRARY_PATH="/opt/MVS/lib/64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Add these two lines to the rc file for your shell (`~/.bashrc` or `~/.zshrc`) for future terminals. Override stale installation paths as well; `MVCAM_COMMON_RUNENV` points to `lib/`, not `lib/64/`.

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

`pixi install -e reorient-deploy` from the repo root pulls in (see
[`pixi.toml`](../../../../../pixi.toml) `[feature.reorient-deploy.pypi-dependencies]`):
`opencv-contrib-python>=4.13` (ArUco + IPPE), `pupil-apriltags>=1.0`
(wrist tag), `pyzmq>=27.0` (cube/goal pub-sub), `glfw>=2.10` (passive
MuJoCo viewer), `wujihandpy==1.5.1` (Wuji Hand driver), `pyyaml>=6.0`.
Smoke-test:

```bash
pixi run -e reorient-deploy python -c "import cv2, pupil_apriltags, zmq, wujihandpy; print(cv2.__version__)"
```

## 3. Cube fabrication

The fastest path is to reproduce the reference cube from the
release-bundled assets. Download the release zip:

```bash
# Requires the GitHub CLI (https://cli.github.com) and unzip.
# Run once in a directory without an existing wuji-mjlab-v2026.9.27-assets folder.
gh release download v2026.9.27 --repo wuji-technology/wuji-mjlab --pattern 'wuji-mjlab-v2026.9.27-assets.zip'
unzip wuji-mjlab-v2026.9.27-assets.zip
```

The cube bundle is
`reorient/hardware/{cube.3mf, cube.step, cube.obj, cube.mtl, cube.png}`, and
the jig assets are under `hardware/hand1/` and `hardware/hand2/`. The cube is a 54 mm edge solid with
4 × 13 mm ArUco tags per face (24 tags total, IDs 0–23).

Before printing, inspect the bundled cube mesh and its 24-tag layout:

```bash
pixi run -e reorient-deploy python deploy/reorient/tools/view_release_cube.py \
  --cube wuji-mjlab-v2026.9.27-assets/reorient/hardware/cube.obj
```

> Without the `gh` CLI, browse to the
> [v2026.9.27 release page](https://github.com/wuji-technology/wuji-mjlab/releases/tag/v2026.9.27),
> download the `wuji-mjlab-v2026.9.27-assets.zip` attachment manually, then run
> the same `unzip` command from this directory.

### 3.1 3D-printing the cube

**Use the shipped Bambu Lab `.3mf`** (in the release attachment
at `wuji-mjlab-v2026.9.27-assets/reorient/hardware/cube.3mf`) — it contains the cube
geometry plus the dual-material assignments that print the 24 ArUco
4×4 tag patterns directly into the faces. No stickers, no glue, no
alignment fuss.

Workflow:

1. Open the file in **Bambu Studio**, or drag it onto a connected Bambu
   printer (Bambu's printer firmware can slice `.3mf` files in place).
2. Load **two filaments**: one **black**, one **white** (PLA is fine).
   The slicer will prompt for AMS / external-spool slot assignment — the
   `.3mf` declares which logical slot is "tag" vs "base", confirm the
   physical filaments match.
3. Slice and print, then weigh the finished cube.

Expected geometry (matches [`cube_tags.json`](../../../../../deploy/reorient/config/cube_tags.json),
do **not** rescale): **54 mm cube edge**, 13 mm tag tiles, tags centred
18 mm from face centre along each face's local u/v axes.

Tag IDs follow the layout in
[`deploy/reorient/config/cube_tags.json`](../../../../../deploy/reorient/config/cube_tags.json):
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

> The printable `cube.3mf` uses the
> black/white dual-material assignment described above; a
> sticker fallback may use a plain white background as long as the
> pattern remains high-contrast black on white.

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
ever regenerate the `.3mf` from scratch (e.g. to use a different
edge length or tag dictionary).

For a visual reference, run `view_release_cube.py` as shown in section 3;
the per-face ID table above provides the same information textually.

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
(T/B). Keep each tile's "up" aligned with the face's v-axis. An in-plane
mounting error changes that tag's cube-frame corner correspondence: with
one visible tag it can yield a rotated cube pose, while inconsistent tags
on the same face can raise reprojection error and trigger the PnP gate.

## 4. Wrist AprilTag

The cube observer defines the world (wrist) frame from a single
AprilTag36h11 marker rigidly mounted to the wrist plate
(`cube_world_observer.py`: "World frame defined by AprilTag ID 0").
Required specs (hardcoded): family **AprilTag36h11**, ID **0**,
**tag size 50.4 mm** (`WORLD_TAG_SIZE = 0.0504`). The AprilTag library
uses tag size as the metric scale, so printing-scale errors propagate
into pose estimation.

The tag plane sits on the back of the wrist, perpendicular to the palm
normal. The observer hardcodes a 180° rotation about the tag X-axis in
`WORLD_FRAME_CORRECTION` (X unchanged, Y and Z negated), so the printed
jig must place the tag at the same orientation as the reference rig.
![Wrist tag mounting — AprilTag visible at the top of the assembled jig](images/hand-jig-side.jpg)

> **Warning.** Do not move the wrist tag after world-frame sampling. The
> observer averages 100 frames on startup then freezes the world pose
> (`_finalize_world_frame`); any later shift corrupts the cube-in-tag
> observation the policy reads.

### 4.1 Print or buy

You only need ID 0 with tag size 50.4 mm. Off-the-shelf options do exist (search
e.g. "AprilTag36h11 sticker 50.4mm" on Amazon / AliExpress / Taobao);
before buying, verify that the tag size is 50.4 mm.

**Dimension convention.** [AprilTag defines tag size](https://github.com/AprilRobotics/apriltag#pose-estimation)
as the edge between the white border and the black border (the distance
between the detection corners). For AprilTag36h11, the official image is
a 10 × 10-cell tile: 6 × 6 data cells, a 1-cell black frame on each side,
and a 1-cell white quiet zone on each side. Tag size is the outer edge
of the 8 × 8-cell black frame: 50.4 mm, or 6.3 mm per cell. The complete
10 × 10 tile is therefore 63.0 mm × 63.0 mm, including the one-cell
(6.3 mm) outer white quiet zone.

**DIY print workflow.**

1. **Source the tag image.** Official PNGs live at
   [`AprilRobotics/apriltag-imgs`](https://github.com/AprilRobotics/apriltag-imgs);
   ID 0 of family 36h11 is at
   [`tag36h11/tag36_11_00000.png`](https://github.com/AprilRobotics/apriltag-imgs/blob/master/tag36h11/tag36_11_00000.png).
   It is a 10 × 10 px raster. The same repo's `tag_to_svg.py` produces a
   vector version at any size (preferred if your printer driver accepts
   SVG):

   ```bash
   # Start from the wuji-mjlab repository root
   git clone https://github.com/AprilRobotics/apriltag-imgs
   cd apriltag-imgs
   pixi run --manifest-path ../pixi.toml python tag_to_svg.py tag36h11/tag36_11_00000.png tag36_11_00000.svg --size=63mm
   ```

2. **Scale without antialiasing.** If you stay with the PNG, upscale
   from 10 × 10 px to the target print size using **nearest-neighbor**
   interpolation — antialiased edges blur the corner gradient and
   degrade pose estimation. ImageMagick:

   ```bash
   # 10 px → 1488 px (63 mm at 600 dpi); nearest-neighbor via -filter point
   # Requires ImageMagick 6; use magick instead of convert for version 7
   convert tag36h11/tag36_11_00000.png -filter point -resize 14880% tag_63mm_600dpi.png
   ```

   Or in GIMP / Photoshop: Image → Scale, Interpolation = "None" /
   "Nearest neighbor".

3. **Lay out with quiet zone.** Use the full 63 mm official tile and
   preserve its 6.3 mm outer white quiet-zone cell; do not crop into
   that white cell.

4. **Print.** ≥ 600 dpi, matte vinyl or laminated matte paper, black
   toner / pigment ink on white. Avoid glossy stock (glare under the
   industrial LED lighting kills detection).

5. **Verify with caliper.** Measure the black frame's outer edge, not
   the white border's outer edge, and confirm tag size 50.4 mm on both axes.

6. **Mount.** Apply to the wrist plate at the orientation shown in the
   image above. Re-read the §4 warning before re-running `vision`.

The observer's metric scale is `WORLD_TAG_SIZE`; before world-frame
sampling, measure the mounted tag size and make sure the constant
matches the measured value.

After creating the tags, return to the repository root before continuing:

```bash
cd ..
```

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
| 1 | 350 × 200 × 13 mm | Aluminum honeycomb breadboard | 1 | AL6061-T6 (SS) | Anodized black | Off-the-shelf |
| 2 | see `base.3mf` (release attachment) | 3D-printed base | 1 | PLA | — | Print |

Plus 4× M6 socket-head screws (length depending on breadboard
thickness — 16 mm typical) for fixing the base to the breadboard.

The breadboard is a standard catalog part — a 350 × 200 × 13 mm
AL6061-T6 aluminum honeycomb plate, anodized black, drilled with an
**M6 tapped through-hole grid on a 25 mm pitch** (25 mm edge margin,
91 holes in a 13 × 7 array). Any plate matching that grid substitutes;
no custom machining is needed on the breadboard itself.

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

**Dimensioned reference.** The release attachment's
`hardware/hand1/` directory ships `assembly.pdf` (the dimensioned
assembly drawing), `assembly.step` (the full STEP solid — open it in
any CAD viewer to measure or remix) and `base.3mf` (the printable
base). If you reproduce from scratch rather than from the shipped CAD,
these are the key numbers off the drawing (all mm):

| Dimension | Value |
|---|---|
| Assembled height (base + breadboard) | **146.7 mm** |
| Hand back-tilt at rest | **10.0°** |
| Total assembly mass | ~14.3 kg (dominated by the breadboard) |
| Breadboard | 350 × 200 × 13 mm, AL6061-T6, anodized black |
| Breadboard hole grid | 91 × M6 tapped through-holes, 25 mm pitch, 13 × 7 array |
| 3D-printed base bounding box | ~90 (W) × 93 (D) × 134 (H) mm |
| Base → breadboard fixing | 4 × M6 SHCS through φ6.60 clearance holes, φ11.0 ↧ 6.8 counterbore (head recess); 2 × φ3.2 locating holes |
| Base print profile | PLA, 0.2 mm layer, 3 walls, 50 % infill, 0.4 mm nozzle (Bambu profile bundled in `base.3mf`) |

General fabrication notes from the drawing: untoleranced dimensions
follow **GB/T1804-2000-F** (e.g. ±0.10 over 18–50 mm, ±0.20 over
120–250 mm) with an Ra 3.2 unspecified surface finish; deburr/chamfer
all sharp edges; the part is RoHS 2.0 / REACH compliant. Functionally,
only the **10° back-tilt** and the **counterbore pattern** matter for
camera framing and a flush bolt-down — the rest is reproduction
convenience.

### 5.2 Camera mounting

Mount the camera so the entire cube reachable workspace and the wrist
AprilTag both stay inside the preview throughout reorientation. In
practice this is roughly 30-40 cm above the palm, but the exact
distance is not critical. What matters is (a) the wrist tag is visible
at rest, and (b) the cube and its reachable workspace fit comfortably
inside the ROI you select below.

**Don't hand-edit `fast_roi`** in
[`camera.yaml`](../../../../../deploy/reorient/config/camera.yaml) — the vision
program ships an interactive selector. With the camera mounted:

```bash
pixi run -e reorient-deploy vision
```

In the OpenCV preview window, press **`s`** to open the ROI selection
dialog. Drag a rectangle around the **cube's reachable workspace**
(this is the per-frame detection ROI that the observer crops to before
running ArUco). The wrist AprilTag doesn't need to stay inside the
ROI — it only needs to be visible during the 100-frame world-frame
sampling at launch (and again whenever you press `w`; see below).
Press ENTER / SPACE to confirm (`C` cancels).

Other vision-window hotkeys:

| Key | Action |
|---|---|
| `s` | Open ROI selector (above) |
| `w` | Resample the world frame (re-detects wrist AprilTag, resets cube filters) |
| `r` | Reset cube filters only (world frame untouched) |
| `q` | Quit |

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

Use diffuse ambient light. Avoid backlighting. Leave
CLAHE on (`enable_clahe: true` in
[`observer.yaml`](../../../../../deploy/reorient/config/observer.yaml)); if cube
faces look "noisy" under CLAHE, disable it.

## 6. Camera intrinsics calibration

`fx`, `fy`, `cx`, `cy` and the 5-parameter Brown-Conrady distortion in
[`camera.yaml`](../../../../../deploy/reorient/config/camera.yaml) describe the
reference rig. For a different camera you **must** re-calibrate
before trusting any cube pose.

### 6.1 Print the chessboard

11 × 8 **inner** corners (12 × 9 squares), 20 mm squares to match the
calibrator's `SQUARE_SIZE = 0.020` constant (adjust the constant if you
print larger). Mount on rigid flat backing — bowing introduces a
systematic radial bias.

### 6.2 Run the guided calibrator

```bash
pixi run -e reorient-deploy python deploy/reorient/tools/camera_calibrate.py
```

The tool walks through 14 capture tasks (center / left / right / top /
bottom regions; near / mid / far distance; straight / tilted attitude).

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

`pixi run -e reorient-deploy vision` (preview mode). The wrist-tag pose should
hold steady to sub-pixel jitter when held still. The observer rejects
PnP fits with mean reprojection error > 6.0 px
(`observer.yaml::pnp.reproj_threshold`); frequent rejections mean
intrinsics are under-fit — return to section 6.2 and capture more tilted
samples.

## 7. Pose-estimation troubleshooting

After hardware is fixed, [`observer.yaml`](../../../../../deploy/reorient/config/observer.yaml)
gives you four knobs to trade noise vs. lag.

- Cube jitter when held still → lower `process_noise` and `alpha` and/or
  raise `measurement_noise`.
- Pose lag during fast reorientation → raise `process_noise` and `alpha`
  and/or lower `measurement_noise`.
- Cube drops to "lost" repeatedly → raise `pnp.reproj_threshold` to e.g.
  8.0 px **and** re-check section 6 intrinsics; if axes look swapped,
  fix `cube_tags.json::face_rotations` or re-stick the offending tag.

## 8. Continue to deployment

After hardware setup and calibration, follow [Deploy](deploy.md#end-to-end-smoke-test-wuji-hand-1) to select a checkpoint and complete the end-to-end smoke test.

License: Apache 2.0. See repository root [`LICENSE`](../../../../../LICENSE).
