# Wuji MJLab Assets

- `robots/wuji_hand/`: Wuji Hand robot XML/mesh assets used by Wuji tasks.
- `robots/wuji_hand2/`: right-hand Wuji Hand 2 MJCF/mesh assets
  and lazy-loaded robot configuration. Sourced from the public
  [`wuji-description`](https://github.com/wuji-technology/wuji-description)
  repository's right-hand body asset directory
  under the MIT license, Copyright (c) 2025 Wuji Technology.
  `mjcf/right_mjlab.xml` uses the public model name `wujihand2-right` and keeps
  the upstream body/joint names; `wuji_hand2_cfg._get_spec`
  renames them to task names with `WUJI_HAND2_TASK_NAMES`.
- `objects/inhand_object/`: in-hand manipulation objects used by the reorient task.
