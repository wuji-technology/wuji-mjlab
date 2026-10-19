# Wuji MJLab Assets

- `robots/wuji_hand/`: lazy-loaded right-hand Wuji Hand 1 robot configuration.
  `wuji_hand_cfg` reads `hand/body-with-soft/mjcf/right_simplified.xml` and its
  meshes from the `robots/wuji-description` submodule and applies the task's
  edits at load time: simulation options, contact defaults, collision geom
  names, fingertip sites and palm contact excludes.
- `robots/wuji_hand2/`: lazy-loaded right-hand Wuji Hand 2 robot configuration.
  The MJCF and meshes are not copied here: `wuji_hand2_cfg` reads
  `hand2/hand2_2/body/` from the
  [`wuji-description`](https://github.com/wuji-technology/wuji-description)
  submodule at `robots/wuji-description` (MIT license, Copyright (c) 2025
  Wuji Technology). `wuji_hand2_cfg._get_spec` roots the model at the palm with
  the upstream mount as its child, drops the fixed-palm wrist IMU sensors, and
  renames the upstream body/joint names to task names with
  `WUJI_HAND2_TASK_NAMES`. `meshes/right/jig_table.STL` is the reorient
  calibration jig, which wuji-description does not ship.
- `objects/inhand_object/`: in-hand manipulation objects used by the reorient task.
