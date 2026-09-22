# Unitree G1 asset source

The G1 MJCF and STL assets in this directory are derived from Unitree Robotics'
`unitree_mujoco` repository at commit
`c598f103acb87a5fd3de7c9037f4dab6aa7f232b`:

https://github.com/unitreerobotics/unitree_mujoco

They are redistributed under the BSD 3-Clause license in `LICENSE`. Only the
STL files referenced by `g1_29dof.xml` are included.

Local changes relative to that upstream revision:

- `g1_29dof.xml` adds the non-colliding, group-3
  `front_depth_camera_visual` shell from the training G1 XML.
- `scene_stair.xml` is the generated Stair course used by this
  project.

The sim2sim server adds the actual `front_depth` camera at runtime; the marker
is excluded from the policy depth render.
