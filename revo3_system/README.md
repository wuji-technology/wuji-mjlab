# revo3_system

Self-contained URDF + mesh bundle.

```
revo3_system/
├── meshes/
│   ├── arm/{visual,collision}/
│   ├── body/{visual,collision}/        # if applicable
│   ├── connector/{visual,collision}/
│   └── hands/{visual,collision}/
└── urdf/
    └── *.urdf
```

URDFs reference meshes via RELATIVE paths (`../meshes/...`), so the
folder is fully self-contained — no ROS package resolution required.
RViz / MoveIt / urdfpy / pybullet all resolve relative
`<mesh filename>` against the URDF file's directory.
