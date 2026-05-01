# Canonical Robot Set

Auto-generated from `data/robot_manifest_research.json` by `scripts/summarize_manifest.py`. One row per canonical (deduped) robot in the research-grade training pool.

## Headline numbers

- **Canonical robots: 371**
- Vendors covered: 47
- DOF range: 1–102
- Fidelity: 130 high / 176 medium / 65 low
- Scale: 52 compact / 166 tabletop / 141 fullsize / 5 huge / 0 unit_bug / 7 unknown

## By vendor (top 15)

| Vendor | N | Sample family |
|---|---:|---|
| fanuc | 41 | crx |
| <unmatched> | 38 | bullet3 |
| nasa | 33 | r2_description |
| kinova | 27 | jaco |
| abb | 25 | abb |
| kuka | 19 | iiwa |
| unitree | 16 | a1 |
| franka | 15 | franka |
| rainbow_robotics | 14 | rb |
| open_robotics | 13 | turtlebot |
| universal_robots | 11 | ur10 |
| staubli | 10 | staubli |
| <research> | 9 | adroit |
| ufactory | 9 | lite6 |
| ghost_robotics | 8 | minitaur |

## By source

| Source | N |
|---|---:|
| urdf_files_dataset | 183 |
| mujoco_menagerie | 69 |
| robot-assets | 44 |
| bullet3 | 34 |
| robosuite | 25 |
| Gymnasium-Robotics | 16 |

## DOF distribution

| DOF | Count |
|---:|---:|
| 1 | 14 |
| 2 | 24 |
| 3 | 2 |
| 4 | 9 |
| 5 | 9 |
| 6 | 110 |
| 7 | 34 |
| 8 | 19 |
| 9 | 5 |
| 10 | 10 |
| 11 | 3 |
| 12 | 26 |
| 13 | 3 |
| 14 | 4 |
| 15 | 8 |
| 16 | 15 |
| 17 | 1 |
| 18 | 5 |
| 19 | 3 |
| 20 | 4 |
| 22 | 1 |
| 23 | 3 |
| 24 | 10 |
| 25 | 3 |
| 26 | 4 |
| 27 | 1 |
| 29 | 1 |
| 30 | 2 |
| 32 | 7 |
| 33 | 1 |
| 36 | 1 |
| 39 | 2 |
| 42 | 3 |
| 43 | 1 |
| 44 | 2 |
| 49 | 1 |
| 54 | 1 |
| 56 | 4 |
| 58 | 4 |
| 59 | 3 |
| 64 | 1 |
| 74 | 4 |
| 92 | 1 |
| 93 | 1 |
| 102 | 1 |

## Scale class distribution

Scale class buckets are derived from the link-origin AABB at zero pose (a lower bound on the robot's full extent — meshes could push it further). `unit_bug` flags suspected mm-encoded URDFs (>50 m chain), `unknown` is one-link or all-coincident fixtures where FK can't separate origins.

| Scale | Max-axis range | Count |
|---|---|---:|
| compact | <0.3 m | 52 |
| tabletop | 0.3–1.0 m | 166 |
| fullsize | 1.0–2.5 m | 141 |
| huge | >2.5 m | 5 |
| unit_bug | >50 m (mm-bug) | 0 |
| unknown | — | 7 |

## Full canonical robot list

Sorted by vendor, family, DOF.

| Vendor | Family | DOF | Links | Fidelity | Scale | Mesh MB | AABB (m) | Range (rad) | Range (m) | License | Path |
|---|---|---:|---:|---|---|---:|---|---:|---:|---|---|
| <generic> | racecar | 6 | 13 | medium | tabletop | 0.80 | 0.39×0.20×0.12 | 8.28 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/data/racecar/racecar.urdf` |
| <research> | adroit | 30 | 31 | low | tabletop | 0.37 | 0.29×0.74×0.45 | 41.05 | 0.80 | MIT (Farama Foundation, 2022) | `Gymnasium-Robotics/gymnasium_robotics/envs/assets/adroit_hand/adroit_door.xml` |
| <research> | adroit | 30 | 30 | low | tabletop | 0.37 | 0.23×0.70×0.25 | 50.53 | 0.00 | MIT (Farama Foundation, 2022) | `Gymnasium-Robotics/gymnasium_robotics/envs/assets/adroit_hand/adroit_pen.xml` |
| <research> | adroit | 33 | 31 | low | tabletop | 0.37 | 0.13×0.70×0.25 | 51.78 | 0.10 | MIT (Farama Foundation, 2022) | `Gymnasium-Robotics/gymnasium_robotics/envs/assets/adroit_hand/adroit_hammer.xml` |
| <research> | adroit | 36 | 29 | low | tabletop | 0.37 | 0.12×0.70×0.25 | 55.03 | 1.50 | MIT (Farama Foundation, 2022) | `Gymnasium-Robotics/gymnasium_robotics/envs/assets/adroit_hand/adroit_relocate.xml` |
| <research> | flybody | 102 | 68 | high | tabletop | 13.08 | 0.28×0.33×0.13 | 136.99 | 0.00 | Apache-2.0 (Menagerie: flybody) | `mujoco_menagerie/flybody/fruitfly.xml` |
| <research> | leap | 16 | 18 | medium | compact | 4.71 | 0.16×0.17×0.03 | 39.33 | 0.00 | Unknown (Menagerie: leap_hand) | `mujoco_menagerie/leap_hand/left_hand.xml` |
| <research> | low_cost_robot_arm | 6 | 8 | medium | compact | 2.36 | 0.17×0.02×0.18 | 21.50 | 0.00 | Apache-2.0 (Menagerie: low_cost_robot_arm) | `mujoco_menagerie/low_cost_robot_arm/low_cost_robot_arm.xml` |
| <research> | microtaur | 12 | 26 | high | tabletop | 20.95 | 0.28×0.56×0.06 | 0.00 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/examples/pybullet/gym/pybullet_data/quadruped/microtaur/microtaur.urdf` |
| <research> | robot_soccer_kit | 64 | 66 | medium | compact | 2.17 | 0.13×0.11×0.06 | 395.84 | 0.01 | Unknown (Menagerie: robot_soccer_kit) | `mujoco_menagerie/robot_soccer_kit/robot_soccer_kit.xml` |
|  | bullet3 | 1 | 2 | low | unknown | 0.00 | 0.00×0.00×0.00 | 0.00 | 5.93 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/data/cube_gripper_left.urdf` |
|  | bullet3 | 1 | 2 | low | unknown | 0.00 | 0.00×0.00×0.00 | 0.00 | 5.93 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/data/cube_gripper_right.urdf` |
|  | bullet3 | 1 | 2 | low | compact | 0.35 | 0.26×0.29×0.03 | 6.28 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/data/wheel.urdf` |
|  | bullet3 | 1 | 2 | low | unknown | 0.00 | 0.00×0.00×0.00 | 6.28 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/examples/pybullet/gym/pybullet_data/cube_rotate.urdf` |
|  | bullet3 | 4 | 5 | low | unknown | 0.00 | 0.00×0.00×0.00 | 6.28 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/examples/pybullet/gym/pybullet_data/cube_no_rotation.urdf` |
|  | bullet3 | 8 | 16 | low | tabletop | 0.04 | 0.44×0.52×0.86 | 7.38 | 0.38 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/data/r2d2.urdf` |
|  | gymnasium-robotics | 24 | 30 | medium | tabletop | 0.61 | 0.12×0.46×0.20 | 30.60 | 0.00 | MIT (Farama Foundation, 2022) | `Gymnasium-Robotics/gymnasium_robotics/envs/assets/hand/manipulate_block.xml` |
|  | gymnasium-robotics | 24 | 30 | medium | tabletop | 0.61 | 0.12×0.46×0.20 | 30.60 | 0.00 | MIT (Farama Foundation, 2022) | `Gymnasium-Robotics/gymnasium_robotics/envs/assets/hand/manipulate_block_touch_sensors.xml` |
|  | gymnasium-robotics | 24 | 30 | medium | tabletop | 0.61 | 0.12×0.46×0.20 | 30.60 | 0.00 | MIT (Farama Foundation, 2022) | `Gymnasium-Robotics/gymnasium_robotics/envs/assets/hand/manipulate_egg.xml` |
|  | gymnasium-robotics | 24 | 30 | medium | tabletop | 0.61 | 0.12×0.46×0.20 | 30.60 | 0.00 | MIT (Farama Foundation, 2022) | `Gymnasium-Robotics/gymnasium_robotics/envs/assets/hand/manipulate_egg_touch_sensors.xml` |
|  | gymnasium-robotics | 24 | 30 | medium | tabletop | 0.61 | 0.12×0.46×0.20 | 30.60 | 0.00 | MIT (Farama Foundation, 2022) | `Gymnasium-Robotics/gymnasium_robotics/envs/assets/hand/manipulate_pen.xml` |
|  | gymnasium-robotics | 24 | 30 | medium | tabletop | 0.61 | 0.12×0.46×0.20 | 30.60 | 0.00 | MIT (Farama Foundation, 2022) | `Gymnasium-Robotics/gymnasium_robotics/envs/assets/hand/manipulate_pen_touch_sensors.xml` |
|  | gymnasium-robotics | 24 | 28 | medium | tabletop | 0.61 | 0.12×0.46×0.15 | 30.60 | 0.00 | MIT (Farama Foundation, 2022) | `Gymnasium-Robotics/gymnasium_robotics/envs/assets/hand/reach.xml` |
|  | mujoco_menagerie | 8 | 4 | low | unknown | 0.48 | 0.00×0.00×0.00 | 18.85 | 7.30 | MIT (Menagerie: umi_gripper) | `mujoco_menagerie/umi_gripper/umi_gripper.xml` |
|  | robosuite | 1 | 5 | low | compact | 0.38 | 0.12×0.00×0.01 | 1.57 | 0.00 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/grippers/bd_gripper.xml` |
|  | robosuite | 2 | 7 | medium | compact | 2.06 | 0.00×0.05×0.12 | 0.00 | 0.06 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/grippers/rethink_gripper.xml` |
|  | robosuite | 4 | 5 | medium | tabletop | 2.55 | 0.20×0.00×0.70 | 6.28 | 0.34 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/bases/omron_mobile_base.xml` |
|  | robosuite | 11 | 21 | medium | compact | 1.85 | 0.02×0.14×0.13 | 17.43 | 0.00 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/grippers/fourier_left_hand.xml` |
|  | robosuite | 11 | 21 | medium | compact | 1.84 | 0.02×0.14×0.13 | 17.43 | 0.00 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/grippers/fourier_right_hand.xml` |
|  | robosuite | 12 | 21 | medium | compact | 3.98 | 0.04×0.12×0.14 | 17.17 | 0.00 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/grippers/inspire_left_hand.xml` |
|  | robosuite | 12 | 21 | medium | compact | 4.04 | 0.04×0.12×0.14 | 17.17 | 0.00 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/grippers/inspire_right_hand.xml` |
|  | robosuite | 32 | 46 | high | fullsize | 7.44 | 0.05×0.45×1.45 | 87.81 | 0.00 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/robots/gr1/robot.xml` |
|  | robot-assets | 49 | 50 | high | fullsize | 18.46 | 0.44×0.19×1.43 | 307.88 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/ginger_robot/gingerurdf.urdf` |
|  | urdf_files_dataset | 1 | 6 | medium | compact | 2.53 | 0.00×0.03×0.13 | 0.00 | 0.02 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/grippers_rethink_robotics/rethink_ee_description/urdf/electric_gripper/rethink_electric_gripper.urdf` |
|  | urdf_files_dataset | 2 | 11 | low | tabletop | 0.16 | 0.32×0.30×0.23 | 0.00 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/amr_robots_description/urdf/amrPioneer3DX.urdf` |
|  | urdf_files_dataset | 2 | 4 | high | tabletop | 18.87 | 0.07×0.40×0.41 | 12.57 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/amr_robots_description/urdf/amrPioneerLX.urdf` |
|  | urdf_files_dataset | 2 | 4 | high | tabletop | 18.87 | 0.07×0.40×0.41 | 12.57 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/pioneer_adept_mobile_robots/description/urdf/pioneer-lx.urdf` |
|  | urdf_files_dataset | 2 | 11 | low | tabletop | 0.16 | 0.32×0.30×0.23 | 0.00 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/pioneer_adept_mobile_robots/description/urdf/pioneer3dx.urdf` |
|  | urdf_files_dataset | 4 | 16 | low | tabletop | 0.14 | 0.38×0.39×0.27 | 0.00 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/amr_robots_description/urdf/amrPioneer3AT.urdf` |
|  | urdf_files_dataset | 4 | 16 | low | tabletop | 0.14 | 0.38×0.39×0.27 | 0.00 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/pioneer_adept_mobile_robots/description/urdf/pioneer3at.urdf` |
|  | urdf_files_dataset | 4 | 6 | high | tabletop | 18.16 | 0.29×0.00×0.49 | 17.89 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/qarm_quanser/qarm_description/urdf/QARM.urdf` |
|  | urdf_files_dataset | 4 | 5 | medium | compact | 2.95 | 0.18×0.00×0.22 | 12.57 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/robotics-toolbox/al5d_description/urdf/al5d_robot.urdf` |
|  | urdf_files_dataset | 5 | 9 | medium | compact | 4.08 | 0.29×0.04×0.20 | 15.17 | 0.03 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/open_manipulator_description/urdf/robotisOpenManipulator.urdf` |
|  | urdf_files_dataset | 5 | 9 | medium | compact | 4.08 | 0.29×0.04×0.20 | 15.17 | 0.03 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/open-manipulator_robotis/open_manipulator_description/urdf/open_manipulator_robot.urdf` |
|  | urdf_files_dataset | 6 | 7 | medium | tabletop | 4.28 | 0.20×0.00×0.31 | 28.71 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/mecademic_description/urdf/meca500r3.urdf` |
|  | urdf_files_dataset | 6 | 7 | low | tabletop | 0.41 | 0.43×0.15×0.67 | 21.99 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/robotics-toolbox/puma560_description/urdf/puma560_robot.urdf` |
|  | urdf_files_dataset | 10 | 19 | high | tabletop | 6.38 | 0.80×0.47×0.35 | 50.27 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/random/xacro_generated/mir_description/urdf/mir.urdf` |
|  | urdf_files_dataset | 24 | 31 | high | tabletop | 28.70 | 0.13×0.29×0.40 | 125.66 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/robotis_op_description/robots/robotisOP2.urdf` |
| abb | abb | 6 | 9 | medium | tabletop | 2.00 | 0.37×0.00×0.63 | 28.80 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/abb_irb120_support/urdf/abbIrb120.urdf` |
| abb | abb | 6 | 9 | medium | tabletop | 2.00 | 0.37×0.00×0.63 | 28.80 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/abb_irb120_support/urdf/abbIrb120T.urdf` |
| abb | abb | 6 | 8 | medium | tabletop | 0.96 | 0.82×0.00×0.96 | 31.64 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/abb_irb1600_support/urdf/abbIrb1600.urdf` |
| abb | abb | 6 | 10 | medium | tabletop | 2.63 | 0.57×0.00×0.90 | 36.83 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/abb/abb_crb15000_support/urdf/crb15000_5_95.urdf` |
| abb | abb | 6 | 10 | high | tabletop | 7.13 | 0.53×0.00×0.89 | 41.19 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/abb/abb_irb1200_support/urdf/irb1200_5_90.urdf` |
| abb | abb | 6 | 10 | high | tabletop | 6.23 | 0.43×0.00×0.79 | 41.28 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/abb/abb_irb1200_support/urdf/irb1200_7_70.urdf` |
| abb | abb | 6 | 10 | medium | tabletop | 2.00 | 0.37×0.00×0.63 | 28.80 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/abb/abb_irb120_support/urdf/irb120_3_58.urdf` |
| abb | abb | 6 | 10 | medium | tabletop | 2.00 | 0.37×0.00×0.63 | 28.80 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/abb/abb_irb120_support/urdf/irb120t_3_58.urdf` |
| abb | abb | 6 | 10 | medium | tabletop | 0.96 | 0.82×0.00×0.96 | 31.64 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/abb/abb_irb1600_support/urdf/irb1600_6_12.urdf` |
| abb | abb | 6 | 10 | medium | fullsize | 1.18 | 0.82×0.00×1.19 | 32.64 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/abb/abb_irb1600_support/urdf/irb1600_8_145.urdf` |
| abb | abb | 6 | 10 | high | fullsize | 13.60 | 1.03×0.00×1.26 | 31.41 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/abb/abb_irb2600_support/urdf/irb2600_12_165.urdf` |
| abb | abb | 6 | 9 | medium | fullsize | 0.59 | 1.72×0.00×1.72 | 28.27 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/abb/abb_irb4400_support/urdf/irb4400l_30_243.urdf` |
| abb | abb | 6 | 10 | medium | fullsize | 2.54 | 1.49×0.00×1.77 | 31.76 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/abb/abb_irb4600_support/urdf/irb4600_20_250.urdf` |
| abb | abb | 6 | 10 | medium | fullsize | 2.65 | 1.58×0.00×1.77 | 31.76 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/abb/abb_irb4600_support/urdf/irb4600_40_255.urdf` |
| abb | abb | 6 | 10 | medium | fullsize | 4.16 | 1.27×0.00×1.57 | 31.76 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/abb/abb_irb4600_support/urdf/irb4600_60_205.urdf` |
| abb | abb | 6 | 10 | medium | tabletop | 1.30 | 0.82×0.00×0.96 | 31.64 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/abb/abb_irb52_support/urdf/irb52_7_120.urdf` |
| abb | abb | 6 | 10 | medium | fullsize | 1.34 | 0.82×0.00×1.19 | 32.64 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/abb/abb_irb52_support/urdf/irb52_7_145.urdf` |
| abb | abb | 6 | 11 | medium | fullsize | 1.83 | 1.95×0.00×2.05 | 38.14 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/abb/abb_irb6600_support/urdf/irb6600_225_255.urdf` |
| abb | abb | 6 | 10 | medium | fullsize | 1.62 | 2.39×0.00×2.11 | 41.36 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/abb/abb_irb6650s_support/urdf/irb6650s_125_350.urdf` |
| abb | abb | 6 | 10 | medium | huge | 1.70 | 2.84×0.00×2.11 | 41.36 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/abb/abb_irb6650s_support/urdf/irb6650s_90_390.urdf` |
| abb | abb | 6 | 12 | medium | fullsize | 3.33 | 2.01×0.19×2.10 | 40.49 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/abb/abb_irb6700_support/urdf/irb6700_200_260.urdf` |
| abb | abb | 6 | 12 | medium | fullsize | 3.06 | 2.05×0.19×2.12 | 40.49 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/abb/abb_irb6700_support/urdf/irb6700_235_265.urdf` |
| abb | abb | 6 | 11 | medium | huge | 1.49 | 3.05×0.00×2.02 | 39.53 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/abb/abb_irb7600_support/urdf/irb7600_150_350.urdf` |
| abb | yumi | 16 | 23 | high | tabletop | 58.55 | 0.62×0.33×0.66 | 81.28 | 0.05 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/yumi/yumi.urdf` |
| abb | yumi | 16 | 22 | high | tabletop | 58.55 | 0.62×0.33×0.56 | 81.28 | 0.05 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/yumi_description/urdf/abbYuMi.urdf` |
| agile_x | arx | 8 | 10 | medium | tabletop | 2.59 | 0.43×0.05×0.25 | 25.64 | 0.09 | BSD-3-Clause (Menagerie: arx_l5) | `mujoco_menagerie/arx_l5/arx_l5.xml` |
| agilex | piper | 8 | 10 | high | tabletop | 8.46 | 0.47×0.00×0.22 | 23.46 | 0.07 | MIT (Menagerie: agilex_piper) | `mujoco_menagerie/agilex_piper/piper.xml` |
| agility | cassie | 20 | 26 | medium | fullsize | 1.71 | 0.41×0.27×1.01 | 50.25 | 0.00 | Unknown (Menagerie: agility_cassie) | `mujoco_menagerie/agility_cassie/cassie.xml` |
| anybotics | anymal | 12 | 14 | high | tabletop | 10.70 | 0.68×0.53×0.25 | 75.40 | 0.00 | BSD-3-Clause (Menagerie: anybotics_anymal_b) | `mujoco_menagerie/anybotics_anymal_b/anymal_b.xml` |
| anybotics | anymal | 12 | 14 | medium | tabletop | 0.93 | 0.72×0.58×0.28 | 55.11 | 0.00 | BSD-3-Clause (Menagerie: anybotics_anymal_c) | `mujoco_menagerie/anybotics_anymal_c/anymal_c.xml` |
| anybotics | anymal | 12 | 22 | high | tabletop | 34.17 | 0.88×0.53×0.57 | 75.40 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/anymal_anybotics/anymal_b_simple_description/urdf/anymal.urdf` |
| apptronik | apollo | 32 | 37 | high | fullsize | 26.03 | 0.25×0.53×1.64 | 61.08 | 0.00 | Apache-2.0 (Menagerie: apptronik_apollo) | `mujoco_menagerie/apptronik_apollo/apptronik_apollo.xml` |
| booster_robotics | booster_t1 | 23 | 25 | medium | tabletop | 3.03 | 0.06×0.72×0.95 | 60.65 | 0.00 | Apache-2.0 (Menagerie: booster_t1) | `mujoco_menagerie/booster_t1/t1.xml` |
| boston_dynamics | spot | 6 | 10 | medium | tabletop | 1.17 | 0.74×0.00×0.07 | 27.58 | 0.00 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/robots/spot_arm/robot.xml` |
| boston_dynamics | spot | 7 | 8 | low | tabletop | 0.18 | 0.89×0.82×0.12 | 28.97 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/random/spot_ros/spot_description/urdf/spot_arm.urdf` |
| boston_dynamics | spot | 12 | 14 | high | tabletop | 6.11 | 0.62×0.33×0.32 | 29.17 | 0.00 | BSD-3-Clause (Menagerie: boston_dynamics_spot) | `mujoco_menagerie/boston_dynamics_spot/spot.xml` |
| boston_dynamics | spot | 12 | 13 | medium | tabletop | 0.63 | 0.62×0.33×0.32 | 29.17 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/spot_boston_dynamics/spot_base_urdf/model.urdf` |
| boston_dynamics | spot | 19 | 22 | high | fullsize | 7.67 | 1.45×0.33×0.60 | 58.32 | 0.00 | BSD-3-Clause (Menagerie: boston_dynamics_spot) | `mujoco_menagerie/boston_dynamics_spot/spot_arm.xml` |
| clearpath | husky | 4 | 15 | high | tabletop | 13.66 | 0.96×0.57×0.38 | 25.13 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/husky_description/urdf/clearpathHusky.urdf` |
| clearpath | jackal | 4 | 13 | medium | tabletop | 0.59 | 0.31×0.38×0.18 | 25.13 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/jackal_description/urdf/clearpathJackal.urdf` |
| clearpath | jackal | 4 | 13 | medium | tabletop | 0.59 | 0.31×0.38×0.18 | 25.13 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/jackal_clearpath_robotics/jackal_description/urdf/jackal.urdf` |
| fanuc | crx | 6 | 10 | medium | tabletop | 1.22 | 0.70×0.15×0.95 | 41.54 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_crx10ia_support/urdf/crx10ial.urdf` |
| fanuc | fanuc | 5 | 9 | medium | tabletop | 0.65 | 0.48×0.00×0.71 | 26.39 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_lrmate200ic_support/urdf/lrmate200ic5f.urdf` |
| fanuc | fanuc | 5 | 9 | medium | tabletop | 0.65 | 0.48×0.00×0.71 | 26.39 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_lrmate200ic_support/urdf/lrmate200ic5h.urdf` |
| fanuc | fanuc | 5 | 9 | medium | tabletop | 0.65 | 0.48×0.00×0.71 | 26.39 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_lrmate200ic_support/urdf/lrmate200ic5hs.urdf` |
| fanuc | fanuc | 5 | 9 | medium | tabletop | 0.73 | 0.37×0.00×0.61 | 31.49 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_lrmate200id_support/urdf/lrmate200id4sh.urdf` |
| fanuc | fanuc | 5 | 9 | medium | tabletop | 1.26 | 0.47×0.00×0.70 | 31.94 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_lrmate200id_support/urdf/lrmate200id7h.urdf` |
| fanuc | fanuc | 5 | 9 | low | fullsize | 0.33 | 0.00×0.10×1.41 | 31.64 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_m430ia_support/urdf/m430ia2f.urdf` |
| fanuc | fanuc | 6 | 10 | medium | tabletop | 1.26 | 0.52×0.00×0.68 | 37.70 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/fanuc_lrmate200ib_support/urdf/fanucLRMate200ib.urdf` |
| fanuc | fanuc | 6 | 8 | low | fullsize | 0.15 | 0.99×0.19×1.40 | 36.48 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/fanuc_m16ib_support/urdf/fanucM16ib.urdf` |
| fanuc | fanuc | 6 | 10 | medium | fullsize | 2.74 | 1.11×0.00×2.12 | 30.42 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_cr35ia_support/urdf/cr35ia.urdf` |
| fanuc | fanuc | 6 | 10 | low | tabletop | 0.11 | 0.47×0.00×0.82 | 32.46 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_cr7ia_support/urdf/cr7ia.urdf` |
| fanuc | fanuc | 6 | 10 | low | tabletop | 0.11 | 0.55×0.00×0.93 | 32.63 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_cr7ia_support/urdf/cr7ial.urdf` |
| fanuc | fanuc | 6 | 10 | low | tabletop | 0.21 | 0.52×0.00×0.68 | 37.68 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_lrmate200i_support/urdf/lrmate200i.urdf` |
| fanuc | fanuc | 6 | 10 | medium | tabletop | 1.26 | 0.52×0.00×0.68 | 37.70 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_lrmate200ib_support/urdf/lrmate200ib.urdf` |
| fanuc | fanuc | 6 | 10 | medium | tabletop | 1.26 | 0.68×0.00×0.68 | 37.26 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_lrmate200ib_support/urdf/lrmate200ib3l.urdf` |
| fanuc | fanuc | 6 | 10 | medium | tabletop | 1.02 | 0.48×0.00×0.71 | 33.02 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_lrmate200ic_support/urdf/lrmate200ic.urdf` |
| fanuc | fanuc | 6 | 10 | medium | tabletop | 0.63 | 0.56×0.00×0.81 | 33.65 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_lrmate200ic_support/urdf/lrmate200ic5l.urdf` |
| fanuc | fanuc | 6 | 10 | medium | tabletop | 1.26 | 0.47×0.00×0.70 | 38.57 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_lrmate200id_support/urdf/lrmate200id.urdf` |
| fanuc | fanuc | 6 | 10 | medium | tabletop | 0.73 | 0.36×0.00×0.61 | 38.12 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_lrmate200id_support/urdf/lrmate200id4s.urdf` |
| fanuc | fanuc | 6 | 10 | medium | tabletop | 0.73 | 0.37×0.00×0.61 | 38.05 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_lrmate200id_support/urdf/lrmate200id4sc.urdf` |
| fanuc | fanuc | 6 | 10 | medium | tabletop | 1.11 | 0.55×0.00×0.80 | 38.71 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_lrmate200id_support/urdf/lrmate200id7l.urdf` |
| fanuc | fanuc | 6 | 10 | medium | tabletop | 1.11 | 0.56×0.00×0.80 | 38.71 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_lrmate200id_support/urdf/lrmate200id7lc.urdf` |
| fanuc | fanuc | 6 | 10 | low | fullsize | 0.35 | 0.89×0.00×1.25 | 44.19 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_m10ia_support/urdf/m10ia.urdf` |
| fanuc | fanuc | 6 | 10 | low | fullsize | 0.18 | 1.11×0.00×1.25 | 43.89 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_m10ia_support/urdf/m10ia7l.urdf` |
| fanuc | fanuc | 6 | 10 | low | fullsize | 0.32 | 0.99×0.00×1.40 | 36.48 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_m16ib_support/urdf/m16ib20.urdf` |
| fanuc | fanuc | 6 | 10 | medium | fullsize | 0.60 | 1.08×0.00×1.56 | 39.76 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_m20ia_support/urdf/m20ia.urdf` |
| fanuc | fanuc | 6 | 10 | medium | fullsize | 0.63 | 1.04×0.00×1.67 | 40.13 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_m20ib_support/urdf/m20ib25.urdf` |
| fanuc | fanuc | 6 | 10 | low | fullsize | 0.42 | 0.00×0.10×1.17 | 38.58 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_m430ia_support/urdf/m430ia2p.urdf` |
| fanuc | fanuc | 6 | 10 | low | fullsize | 0.09 | 0.86×0.00×1.15 | 40.32 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_m6ib_support/urdf/m6ib.urdf` |
| fanuc | fanuc | 6 | 10 | low | tabletop | 0.11 | 0.68×0.00×0.91 | 39.79 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_m6ib_support/urdf/m6ib6s.urdf` |
| fanuc | fanuc | 6 | 10 | low | fullsize | 0.16 | 1.62×0.00×1.88 | 34.82 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_m710ic_support/urdf/m710ic45m.urdf` |
| fanuc | fanuc | 6 | 10 | low | fullsize | 0.07 | 1.34×0.00×1.60 | 47.38 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_m710ic_support/urdf/m710ic50.urdf` |
| fanuc | fanuc | 6 | 12 | low | fullsize | 0.17 | 2.38×0.17×2.20 | 42.08 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_m900ia_support/urdf/m900ia260l.urdf` |
| fanuc | fanuc | 6 | 16 | low | fullsize | 0.22 | 2.33×0.08×2.31 | 41.15 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_m900ib_support/urdf/m900ib700.urdf` |
| fanuc | fanuc | 6 | 10 | low | fullsize | 0.10 | 1.51×0.00×1.54 | 46.34 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_r1000ia_support/urdf/r1000ia80f.urdf` |
| fanuc | fanuc | 6 | 10 | low | fullsize | 0.12 | 1.83×0.00×1.97 | 44.47 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_r2000ib_support/urdf/r2000ib210f.urdf` |
| fanuc | fanuc | 6 | 10 | low | fullsize | 0.14 | 2.26×0.00×1.97 | 43.58 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_r2000ic_support/urdf/r2000ic125l.urdf` |
| fanuc | fanuc | 6 | 10 | low | fullsize | 0.14 | 1.81×0.00×1.97 | 43.77 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_r2000ic_support/urdf/r2000ic165f.urdf` |
| fanuc | fanuc | 6 | 10 | low | fullsize | 0.14 | 1.81×0.00×1.97 | 43.77 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_r2000ic_support/urdf/r2000ic210f.urdf` |
| fanuc | fanuc | 6 | 10 | low | fullsize | 0.22 | 2.28×0.00×1.97 | 43.58 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_r2000ic_support/urdf/r2000ic210l.urdf` |
| fanuc | fanuc | 6 | 10 | low | fullsize | 0.07 | 1.83×0.00×1.97 | 43.77 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_r2000ic_support/urdf/r2000ic270f.urdf` |
| fetch_robotics | fetch | 2 | 7 | high | tabletop | 5.72 | 0.39×0.43×0.31 | 0.00 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/robotics-toolbox/fetch_description/robots/freight.urdf` |
| fetch_robotics | fetch | 5 | 7 | high | fullsize | 9.08 | 0.25×0.00×1.06 | 11.63 | 10.39 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/robotics-toolbox/fetch_description/robots/fetch_camera.urdf` |
| fetch_robotics | fetch | 10 | 19 | high | fullsize | 16.56 | 1.25×0.25×0.79 | 21.06 | 10.39 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/robotics-toolbox/fetch_description/robots/fetch.urdf` |
| fetch_robotics | fetch | 15 | 33 | medium | fullsize | 0.62 | 1.47×0.75×1.06 | 38.97 | 0.45 | MIT (Farama Foundation, 2022) | `Gymnasium-Robotics/gymnasium_robotics/envs/assets/fetch/pick_and_place.xml` |
| fetch_robotics | fetch | 15 | 33 | medium | fullsize | 0.62 | 1.47×0.75×1.06 | 38.97 | 0.45 | MIT (Farama Foundation, 2022) | `Gymnasium-Robotics/gymnasium_robotics/envs/assets/fetch/push.xml` |
| fetch_robotics | fetch | 15 | 32 | medium | fullsize | 0.62 | 1.47×0.75×1.06 | 38.97 | 0.45 | MIT (Farama Foundation, 2022) | `Gymnasium-Robotics/gymnasium_robotics/envs/assets/fetch/reach.xml` |
| fetch_robotics | fetch | 15 | 33 | medium | fullsize | 0.62 | 1.47×0.75×1.06 | 38.97 | 0.45 | MIT (Farama Foundation, 2022) | `Gymnasium-Robotics/gymnasium_robotics/envs/assets/fetch/slide.xml` |
| flexiv | flexiv | 7 | 9 | medium | fullsize | 0.97 | 0.02×0.11×1.26 | 39.64 | 0.00 | Unknown (Menagerie: flexiv_rizon4) | `mujoco_menagerie/flexiv_rizon4/flexiv_rizon4.xml` |
| fourier_intelligence | fourier_n1 | 23 | 30 | high | fullsize | 22.79 | 0.20×0.35×1.12 | 76.07 | 0.00 | Apache-2.0 (Menagerie: fourier_n1) | `mujoco_menagerie/fourier_n1/n1.xml` |
| franka | franka | 7 | 10 | high | fullsize | 7.00 | 0.09×0.00×1.03 | 33.36 | 0.00 | Apache-2.0 (Menagerie: franka_fr3) | `mujoco_menagerie/franka_fr3/fr3.xml` |
| franka | franka | 7 | 11 | high | fullsize | 7.34 | 0.09×0.00×1.03 | 33.36 | 0.00 | Apache-2.0 (Menagerie: franka_fr3_v2) | `mujoco_menagerie/franka_fr3_v2/fr3v2.xml` |
| franka | franka | 10 | 15 | high | fullsize | 27.03 | 0.24×0.00×1.41 | 39.76 | 10.04 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/robotics-toolbox/xacro_generated/franka_description/robots/frankie.urdf` |
| franka | franka | 23 | 44 | medium | huge | 2.58 | 1.71×1.33×2.92 | 45.36 | 0.61 | MIT (Farama Foundation, 2022) | `Gymnasium-Robotics/gymnasium_robotics/envs/assets/kitchen_franka/kitchen_assets/kitchen_env_model.xml` |
| franka | panda | 1 | 3 | medium | compact | 0.65 | 0.00×0.00×0.06 | 0.00 | 0.04 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/bullet_data/panda_description/urdf/hand.urdf` |
| franka | panda | 2 | 4 | low | compact | 0.37 | 0.00×0.00×0.06 | 0.00 | 0.08 | Apache-2.0 (Menagerie: franka_emika_panda) | `mujoco_menagerie/franka_emika_panda/hand.xml` |
| franka | panda | 2 | 7 | low | compact | 0.14 | 0.00×0.02×0.11 | 0.00 | 0.08 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/grippers/panda_gripper.xml` |
| franka | panda | 7 | 9 | high | fullsize | 6.03 | 0.09×0.00×1.03 | 33.48 | 0.00 | Apache-2.0 (Menagerie: franka_emika_panda) | `mujoco_menagerie/franka_emika_panda/mjx_panda_nohand.xml` |
| franka | panda | 7 | 9 | high | fullsize | 9.91 | 0.09×0.00×1.03 | 33.48 | 0.00 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/bullet_data/panda_description/urdf/panda_arm.urdf` |
| franka | panda | 7 | 11 | high | fullsize | 5.59 | 0.09×0.00×1.03 | 33.48 | 0.00 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/robots/panda/robot.xml` |
| franka | panda | 8 | 12 | high | fullsize | 10.57 | 0.09×0.00×1.03 | 33.48 | 0.04 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/bullet_data/panda_description/urdf/panda_arm_hand.urdf` |
| franka | panda | 8 | 12 | high | fullsize | 10.57 | 0.09×0.00×1.03 | 33.48 | 0.04 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/franka_description/robots/frankaEmikaPanda.urdf` |
| franka | panda | 8 | 12 | high | fullsize | 10.57 | 0.09×0.00×1.03 | 33.48 | 0.04 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/robotics-toolbox/xacro_generated/franka_description/robots/panda.urdf` |
| franka | panda | 9 | 12 | high | fullsize | 6.40 | 0.09×0.00×1.03 | 33.48 | 0.08 | Apache-2.0 (Menagerie: franka_emika_panda) | `mujoco_menagerie/franka_emika_panda/mjx_panda.xml` |
| franka | panda | 16 | 45 | high | fullsize | 21.14 | 0.09×1.00×2.03 | 66.95 | 0.08 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/franka_emika/franka_description/robots/dual_panda/dual_panda.urdf` |
| ghost_robotics | minitaur | 1 | 4 | low | compact | 0.01 | 0.21×0.22×0.00 | 0.00 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/data/quadruped/minitaur_single_motor.urdf` |
| ghost_robotics | minitaur | 8 | 27 | low | tabletop | 0.06 | 0.42×0.25×0.12 | 0.00 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/data/quadruped/minitaur_fixed_knees.urdf` |
| ghost_robotics | minitaur | 16 | 27 | low | tabletop | 0.06 | 0.42×0.25×0.12 | 0.00 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/data/quadruped/minitaur.urdf` |
| ghost_robotics | minitaur | 16 | 27 | low | tabletop | 0.06 | 0.41×0.30×0.10 | 0.00 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/data/quadruped/minitaur_derpy.urdf` |
| ghost_robotics | minitaur | 16 | 27 | low | tabletop | 0.06 | 0.48×0.31×0.10 | 0.00 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/data/quadruped/minitaur_rainbow_dash.urdf` |
| ghost_robotics | minitaur | 16 | 27 | low | tabletop | 0.06 | 0.42×0.25×0.12 | 0.00 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/data/quadruped/minitaur_v1.urdf` |
| ghost_robotics | minitaur | 20 | 31 | low | tabletop | 0.06 | 0.48×0.31×0.10 | 0.00 | 0.16 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/data/quadruped/minitaur_rainbow_dash_v1.urdf` |
| ghost_robotics | vision60 | 16 | 23 | low | tabletop | 0.17 | 0.48×0.25×0.19 | 0.00 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/data/quadruped/vision/vision60.urdf` |
| google | aloha | 16 | 21 | medium | tabletop | 2.97 | 0.94×0.04×0.44 | 58.85 | 0.16 | BSD-3-Clause (Menagerie: aloha) | `mujoco_menagerie/aloha/aloha.xml` |
| google | barkour | 12 | 14 | medium | tabletop | 3.75 | 0.60×0.31×0.05 | 36.27 | 0.00 | Apache-2.0 (Menagerie: google_barkour_v0) | `mujoco_menagerie/google_barkour_v0/barkour_v0.xml` |
| google | barkour | 12 | 16 | medium | tabletop | 1.39 | 0.74×0.38×0.12 | 36.46 | 0.00 | Apache-2.0 (Menagerie: google_barkour_vb) | `mujoco_menagerie/google_barkour_vb/barkour_vb.xml` |
| google | google_robot | 9 | 13 | low | fullsize | 0.29 | 0.00×0.40×1.68 | 41.20 | 0.00 | Apache-2.0 (Menagerie: google_robot) | `mujoco_menagerie/google_robot/robot.xml` |
| hello_robot | stretch | 17 | 27 | high | fullsize | 12.16 | 0.21×0.51×1.36 | 53.57 | 1.66 | BSD-3-Clause (Menagerie: hello_robot_stretch) | `mujoco_menagerie/hello_robot_stretch/stretch.xml` |
| hello_robot | stretch | 20 | 38 | high | fullsize | 13.01 | 0.21×0.59×1.36 | 64.73 | 1.68 | Apache-2.0 (Menagerie: hello_robot_stretch_3) | `mujoco_menagerie/hello_robot_stretch_3/stretch.xml` |
| i2rt | yam | 8 | 14 | medium | tabletop | 1.88 | 0.41×0.07×0.20 | 23.47 | 0.08 | MIT (Menagerie: i2rt_yam) | `mujoco_menagerie/i2rt_yam/yam.xml` |
| iit | softfoot | 92 | 50 | medium | compact | 1.14 | 0.20×0.07×0.08 | 94.25 | 0.00 | BSD-3-Clause (Menagerie: iit_softfoot) | `mujoco_menagerie/iit_softfoot/softfoot.xml` |
| iit | softfoot | 93 | 51 | medium | compact | 1.14 | 0.20×0.07×0.08 | 94.25 | 0.00 | BSD-3-Clause (Menagerie: iit_softfoot) | `mujoco_menagerie/iit_softfoot/scene.xml` |
| kinova | jaco | 6 | 10 | medium | compact | 0.59 | 0.05×0.11×0.18 | 0.18 | 0.00 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/grippers/jaco_three_finger_gripper.xml` |
| kinova | jaco | 7 | 11 | medium | tabletop | 1.11 | 0.00×0.01×0.41 | 39.03 | 0.00 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/robots/jaco/robot.xml` |
| kinova | kinova | 6 | 15 | high | tabletop | 7.60 | 0.14×0.00×0.41 | 34.35 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/kinova/kinova.urdf` |
| kinova | kinova | 6 | 12 | high | compact | 7.72 | 0.00×0.14×0.29 | 26.17 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/kinova_description/urdf/kinovaMicoM1N4S200.urdf` |
| kinova | kinova | 7 | 9 | medium | fullsize | 1.57 | 0.00×0.02×1.13 | 38.93 | 0.00 | BSD-3-Clause (Menagerie: kinova_gen3) | `mujoco_menagerie/kinova_gen3/gen3.xml` |
| kinova | kinova | 7 | 11 | medium | fullsize | 1.57 | 0.00×0.02×1.19 | 39.73 | 0.00 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/robots/kinova3/robot.xml` |
| kinova | kinova | 8 | 14 | high | tabletop | 8.23 | 0.00×0.14×0.50 | 39.40 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/kinova_description/urdf/kinovaJacoJ2N6S200.urdf` |
| kinova | kinova | 8 | 14 | high | tabletop | 8.30 | 0.00×0.14×0.41 | 38.73 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/kinova_description/urdf/kinovaMicoM1N6S200.urdf` |
| kinova | kinova | 8 | 12 | high | compact | 7.95 | 0.00×0.14×0.29 | 16.62 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/kinova_robotics/kinova_description/urdf/m1n4s200_standalone.urdf` |
| kinova | kinova | 9 | 16 | high | tabletop | 6.74 | 0.06×0.14×0.50 | 41.40 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/kinova_description/urdf/kinovaJacoJ2N6S300.urdf` |
| kinova | kinova | 9 | 16 | high | tabletop | 7.34 | 0.06×0.13×0.41 | 40.35 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/kinova_description/urdf/kinovaJacoJ2S6S300.urdf` |
| kinova | kinova | 9 | 16 | high | tabletop | 6.82 | 0.06×0.14×0.41 | 40.73 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/kinova_description/urdf/kinovaMicoM1N6S300.urdf` |
| kinova | kinova | 10 | 17 | high | tabletop | 7.10 | 0.06×0.14×0.50 | 47.89 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/kinova_description/urdf/kinovaJacoJ2N7S300.urdf` |
| kinova | kinova | 10 | 17 | high | tabletop | 7.70 | 0.06×0.13×0.41 | 45.03 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/kinova_description/urdf/kinovaJacoJ2S7S300.urdf` |
| kinova | kinova | 10 | 14 | high | tabletop | 6.63 | 0.06×0.13×0.41 | 20.79 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/kinova_robotics/kinova_description/urdf/j2n4s300_standalone.urdf` |
| kinova | kinova | 10 | 14 | high | tabletop | 8.60 | 0.00×0.14×0.50 | 17.28 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/kinova_robotics/kinova_description/urdf/j2n6s200_standalone.urdf` |
| kinova | kinova | 10 | 14 | high | tabletop | 9.15 | 0.00×0.14×0.41 | 22.52 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/kinova_robotics/kinova_description/urdf/j2s6s200_standalone.urdf` |
| kinova | kinova | 10 | 14 | high | tabletop | 8.60 | 0.00×0.14×0.41 | 16.62 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/kinova_robotics/kinova_description/urdf/m1n6s200_standalone.urdf` |
| kinova | kinova | 10 | 14 | high | tabletop | 6.63 | 0.06×0.13×0.41 | 33.36 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/robotics-toolbox/xacro_generated/kinova_description/urdf/j2n4s300.urdf` |
| kinova | kinova | 12 | 16 | high | tabletop | 7.28 | 0.06×0.14×0.50 | 20.79 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/kinova_robotics/kinova_description/urdf/j2n6s300_standalone.urdf` |
| kinova | kinova | 12 | 16 | high | tabletop | 7.83 | 0.06×0.13×0.41 | 26.03 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/kinova_robotics/kinova_description/urdf/j2s6s300_standalone.urdf` |
| kinova | kinova | 12 | 16 | high | tabletop | 7.27 | 0.06×0.14×0.41 | 20.13 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/kinova_robotics/kinova_description/urdf/m1n6s300_standalone.urdf` |
| kinova | kinova | 13 | 17 | high | tabletop | 7.60 | 0.06×0.14×0.50 | 21.00 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/kinova_robotics/kinova_description/urdf/j2n7s300_standalone.urdf` |
| kinova | kinova | 13 | 17 | high | tabletop | 8.16 | 0.06×0.13×0.41 | 24.42 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/kinova_robotics/kinova_description/urdf/j2s7s300_standalone.urdf` |
| kinova | kinova | 18 | 31 | high | tabletop | 13.49 | 0.54×0.06×0.50 | 82.79 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/kinova_description/urdf/kinovaJacoTwoArmExample.urdf` |
| kinova | kinova | 19 | 53 | high | fullsize | 67.58 | 1.61×0.39×1.06 | 89.46 | 0.47 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/movo_description/urdf/kinovaMovo.urdf` |
| kinova | kinova | 24 | 31 | high | tabletop | 14.55 | 0.54×0.06×0.50 | 41.59 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/kinova_robotics/kinova_description/urdf/two_arm_robot_example_standalone.urdf` |
| kuka | iiwa | 7 | 8 | medium | fullsize | 0.74 | 0.00×0.00×1.26 | 36.48 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/data/kuka_iiwa/model_for_sdf.urdf` |
| kuka | iiwa | 7 | 8 | medium | fullsize | 0.89 | 0.00×0.00×1.26 | 36.48 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/data/kuka_iiwa/model_free_base.urdf` |
| kuka | iiwa | 7 | 8 | medium | fullsize | 0.74 | 0.00×0.00×1.26 | 30.39 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/data/kuka_iiwa/model_vr_limits.urdf` |
| kuka | iiwa | 7 | 9 | medium | fullsize | 1.35 | 0.00×0.00×1.26 | 36.48 | 0.00 | BSD-3-Clause (Menagerie: kuka_iiwa_14) | `mujoco_menagerie/kuka_iiwa_14/iiwa14.xml` |
| kuka | iiwa | 7 | 11 | medium | fullsize | 4.53 | 0.00×0.06×1.26 | 36.48 | 0.00 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/robots/iiwa/robot.xml` |
| kuka | iiwa | 7 | 8 | medium | fullsize | 0.89 | 0.00×0.00×1.26 | 36.48 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/kuka_iiwa/model.urdf` |
| kuka | iiwa | 7 | 11 | high | fullsize | 5.50 | 0.00×0.00×1.31 | 36.48 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/iiwa_description/urdf/kukaIiwa14.urdf` |
| kuka | iiwa | 7 | 11 | high | fullsize | 9.46 | 0.00×0.06×1.27 | 36.48 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/iiwa_description/urdf/kukaIiwa7.urdf` |
| kuka | kuka | 6 | 10 | medium | fullsize | 2.99 | 1.18×0.00×0.44 | 37.72 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/kuka/kuka_kr10_support/urdf/kr10r1100sixx.urdf` |
| kuka | kuka | 6 | 10 | high | fullsize | 5.64 | 1.50×0.00×0.47 | 38.40 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/kuka/kuka_kr10_support/urdf/kr10r1420.urdf` |
| kuka | kuka | 6 | 10 | medium | tabletop | 1.02 | 0.99×0.00×0.43 | 37.72 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/kuka/kuka_kr10_support/urdf/kr10r900_2.urdf` |
| kuka | kuka | 6 | 10 | medium | huge | 3.73 | 2.93×0.00×0.75 | 42.59 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/kuka/kuka_kr150_support/urdf/kr150_2.urdf` |
| kuka | kuka | 6 | 10 | medium | huge | 2.99 | 3.32×0.00×0.76 | 42.64 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/kuka/kuka_kr150_support/urdf/kr150r3100_2.urdf` |
| kuka | kuka | 6 | 10 | medium | tabletop | 3.73 | 0.61×0.00×0.36 | 36.91 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/kuka/kuka_kr3_support/urdf/kr3r540.urdf` |
| kuka | kuka | 6 | 10 | medium | fullsize | 2.02 | 1.51×0.00×0.52 | 41.68 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/kuka/kuka_kr5_support/urdf/kr5_arc.urdf` |
| kuka | kuka | 6 | 10 | medium | tabletop | 3.29 | 0.79×0.00×0.44 | 37.72 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/kuka/kuka_kr6_support/urdf/kr6r700sixx.urdf` |
| kuka | kuka | 6 | 10 | medium | tabletop | 1.02 | 0.99×0.00×0.43 | 37.72 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/kuka/kuka_kr6_support/urdf/kr6r900_2.urdf` |
| kuka | kuka | 6 | 10 | medium | tabletop | 2.63 | 0.98×0.00×0.44 | 37.72 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/kuka/kuka_kr6_support/urdf/kr6r900sixx.urdf` |
| kuka | kuka | 7 | 8 | low | fullsize | 0.45 | 0.00×0.00×1.18 | 35.26 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/data/kuka_lwr/kuka.urdf` |
| mit | mini_cheetah | 12 | 17 | high | tabletop | 12.23 | 0.38×0.22×0.39 | 75.40 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/examples/pybullet/gym/pybullet_data/mini_cheetah/mini_cheetah.urdf` |
| nasa | r2_description | 3 | 12 | medium | unknown | 3.89 | 0.00×0.00×0.00 | 2.55 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2_left_gripper.urdf` |
| nasa | r2_description | 14 | 23 | medium | tabletop | 1.29 | 0.18×0.02×0.47 | 98.50 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2_left_forearm.urdf` |
| nasa | r2_description | 14 | 34 | high | fullsize | 28.09 | 0.13×0.54×1.39 | 66.69 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2c_sim_legs_control.urdf.urdf` |
| nasa | r2_description | 14 | 34 | high | fullsize | 28.09 | 0.13×0.54×1.39 | 66.69 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2c_sim_legs_dynamics.urdf.urdf` |
| nasa | r2_description | 18 | 33 | high | fullsize | 8.96 | 0.42×2.16×0.79 | 57.92 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2c5_dynamics.urdf` |
| nasa | r2_description | 32 | 64 | high | fullsize | 37.05 | 0.42×2.16×2.18 | 124.61 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2b_dynamics.urdf` |
| nasa | r2_description | 32 | 64 | high | fullsize | 37.05 | 0.42×2.16×2.18 | 124.61 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2c1_dynamics.urdf` |
| nasa | r2_description | 32 | 64 | high | fullsize | 37.05 | 0.42×2.16×2.18 | 124.61 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2c6_dynamics.urdf` |
| nasa | r2_description | 32 | 65 | high | fullsize | 37.05 | 0.42×2.16×2.18 | 124.61 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2c_sim_full_dynamics.urdf.urdf` |
| nasa | r2_description | 42 | 74 | high | fullsize | 10.99 | 0.51×2.29×0.87 | 249.34 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2c5_control.urdf` |
| nasa | r2_description | 54 | 96 | high | fullsize | 13.09 | 0.51×2.40×1.59 | 345.34 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2c5.urdf` |
| nasa | r2_description | 56 | 102 | high | fullsize | 39.08 | 0.51×2.29×2.24 | 316.03 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2b_control.urdf` |
| nasa | r2_description | 56 | 102 | high | fullsize | 39.08 | 0.51×2.29×2.24 | 316.03 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2c1_control.urdf` |
| nasa | r2_description | 56 | 105 | high | fullsize | 39.08 | 0.51×2.29×2.26 | 316.03 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2c6_control.urdf` |
| nasa | r2_description | 56 | 105 | high | fullsize | 39.08 | 0.51×2.29×2.26 | 316.03 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2c_sim_full_control.urdf.urdf` |
| nasa | r2_description | 74 | 128 | high | fullsize | 43.67 | 0.51×2.40×2.24 | 417.13 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2b.urdf` |
| nasa | r2_description | 74 | 128 | high | fullsize | 43.67 | 0.51×2.40×2.24 | 417.13 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2c1.urdf` |
| nasa | r2_description | 74 | 132 | high | fullsize | 43.67 | 0.51×2.40×2.26 | 417.13 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2c6.urdf` |
| nasa | r2_description | 74 | 133 | high | fullsize | 43.67 | 0.51×2.40×2.26 | 417.13 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2c6_valve.urdf` |
| nasa | val_description | 16 | 17 | low | tabletop | 0.30 | 0.03×0.44×0.16 | 30.20 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/val_description/model/robots/forearm_left.urdf` |
| nasa | val_description | 16 | 17 | low | tabletop | 0.30 | 0.03×0.44×0.16 | 30.20 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/val_description/model/robots/forearm_right.urdf` |
| nasa | val_description | 26 | 37 | high | fullsize | 5.10 | 0.21×1.16×1.79 | 60.52 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/val_description/model/robots/valkyrie_A_arm_mass_sims.urdf` |
| nasa | val_description | 26 | 37 | high | fullsize | 5.10 | 0.21×1.16×1.79 | 60.52 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/val_description/model/robots/valkyrie_B_arm_mass_sims.urdf` |
| nasa | val_description | 26 | 37 | high | fullsize | 5.10 | 0.21×1.16×1.79 | 60.52 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/val_description/model/robots/valkyrie_C_arm_mass_sims.urdf` |
| nasa | val_description | 26 | 37 | high | fullsize | 5.10 | 0.21×1.16×1.79 | 60.52 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/val_description/model/robots/valkyrie_D_arm_mass_sims.urdf` |
| nasa | val_description | 27 | 46 | high | fullsize | 5.15 | 0.20×1.16×1.81 | 66.80 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/val_description/model/robots/valkyrie_sim_arm_mass_sims.urdf` |
| nasa | val_description | 58 | 69 | medium | fullsize | 2.18 | 0.21×2.05×1.79 | 120.92 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/val_description/model/robots/valkyrie_A.urdf` |
| nasa | val_description | 58 | 69 | medium | fullsize | 2.18 | 0.21×2.05×1.79 | 120.92 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/val_description/model/robots/valkyrie_B.urdf` |
| nasa | val_description | 58 | 69 | medium | fullsize | 2.18 | 0.21×2.05×1.79 | 120.92 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/val_description/model/robots/valkyrie_C.urdf` |
| nasa | val_description | 58 | 69 | medium | fullsize | 2.18 | 0.21×2.05×1.79 | 120.92 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/val_description/model/robots/valkyrie_D.urdf` |
| nasa | val_description | 59 | 78 | medium | fullsize | 2.23 | 0.20×2.05×1.81 | 127.21 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/val_description/model/robots/valkyrie_sim.urdf` |
| nasa | val_description | 59 | 78 | medium | fullsize | 2.27 | 0.20×2.05×1.80 | 127.21 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/val_description/model/robots/valkyrie_sim_angled_legj1s.urdf` |
| nasa | val_description | 59 | 78 | medium | fullsize | 2.23 | 0.20×2.05×1.81 | 127.21 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/val_description/model/robots/valkyrie_sim_gazebo_sync.urdf` |
| onrobot | onrobot | 1 | 4 | medium | compact | 0.77 | 0.09×0.03×0.12 | 0.00 | 0.02 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/random/xacro_generated/juandpenan/onrobot_2fg7_description/urdf/onrobot_2fg7_upload.urdf` |
| onrobot | onrobot | 1 | 7 | medium | compact | 3.50 | 0.00×0.11×0.16 | 1.34 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/random/xacro_generated/osaka_university_onrobot/onrobot_rg2_visualization/urdf/onrobot_rg2_model.urdf` |
| onrobot | onrobot | 1 | 7 | medium | compact | 2.00 | 0.00×0.14×0.20 | 1.26 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/random/xacro_generated/osaka_university_onrobot/onrobot_rg6_visualization/urdf/onrobot_rg6_model.urdf` |
| open_robotics | turtlebot | 2 | 7 | high | compact | 7.72 | 0.08×0.16×0.18 | 12.57 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/turtlebot3_description/urdf/robotisTurtleBot3Burger.urdf` |
| open_robotics | turtlebot | 2 | 13 | high | compact | 19.40 | 0.25×0.29×0.13 | 12.57 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/turtlebot3_description/urdf/robotisTurtleBot3Waffle.urdf` |
| open_robotics | turtlebot | 2 | 13 | high | compact | 13.63 | 0.25×0.29×0.13 | 12.57 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/turtlebot3_description/urdf/robotisTurtleBot3WaffleForOpenManipulator.urdf` |
| open_robotics | turtlebot | 2 | 11 | high | compact | 10.77 | 0.25×0.29×0.13 | 12.57 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/turtlebot3_description/urdf/robotisTurtleBot3WafflePi.urdf` |
| open_robotics | turtlebot | 2 | 11 | high | compact | 10.77 | 0.25×0.29×0.13 | 12.57 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/turtlebot3_description/urdf/robotisTurtleBot3WafflePiForOpenManipulator.urdf` |
| open_robotics | turtlebot | 2 | 40 | high | tabletop | 7.17 | 0.28×0.28×0.41 | 12.57 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/turtlebot_description/robots/clearpathTurtleBot2.urdf` |
| open_robotics | turtlebot | 2 | 7 | medium | compact | 4.56 | 0.08×0.16×0.18 | 12.57 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/turtlebot3_robotis/turtlebot3_description/urdf/turtlebot3_burger.urdf` |
| open_robotics | turtlebot | 2 | 10 | medium | compact | 4.56 | 0.13×0.16×0.18 | 12.57 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/turtlebot3_robotis/turtlebot3_description/urdf/turtlebot3_burger_for_autorace.urdf` |
| open_robotics | turtlebot | 2 | 10 | medium | compact | 4.56 | 0.12×0.16×0.18 | 12.57 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/turtlebot3_robotis/turtlebot3_description/urdf/turtlebot3_burger_for_autorace_2020.urdf` |
| open_robotics | turtlebot | 2 | 13 | high | compact | 8.14 | 0.25×0.29×0.13 | 12.57 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/turtlebot3_robotis/turtlebot3_description/urdf/turtlebot3_waffle.urdf` |
| open_robotics | turtlebot | 2 | 13 | high | compact | 6.21 | 0.25×0.29×0.13 | 12.57 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/turtlebot3_robotis/turtlebot3_description/urdf/turtlebot3_waffle_for_open_manipulator.urdf` |
| open_robotics | turtlebot | 2 | 11 | high | compact | 6.00 | 0.25×0.29×0.13 | 12.57 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/turtlebot3_robotis/turtlebot3_description/urdf/turtlebot3_waffle_pi.urdf` |
| open_robotics | turtlebot | 2 | 11 | high | compact | 6.00 | 0.25×0.29×0.13 | 12.57 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/xacro_generated/turtlebot3_robotis/turtlebot3_description/urdf/turtlebot3_waffle_pi_for_open_manipulator.urdf` |
| pal_robotics | talos | 44 | 46 | medium | fullsize | 4.20 | 0.06×0.69×1.36 | 87.19 | 0.00 | Apache-2.0 (Menagerie: pal_talos) | `mujoco_menagerie/pal_talos/talos.xml` |
| pal_robotics | tiago | 15 | 38 | medium | fullsize | 4.95 | 0.36×1.98×0.99 | 44.40 | 0.35 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/robots/tiago/robot.xml` |
| pal_robotics | tiago | 22 | 24 | medium | fullsize | 1.97 | 0.36×1.01×0.99 | 91.75 | 0.44 | Apache-2.0 (Menagerie: pal_tiago) | `mujoco_menagerie/pal_tiago/tiago.xml` |
| pal_robotics | tiago | 25 | 27 | medium | fullsize | 3.48 | 0.49×2.03×0.96 | 78.30 | 0.53 | Apache-2.0 (Menagerie: pal_tiago_dual) | `mujoco_menagerie/pal_tiago_dual/tiago_dual.xml` |
| pndbotics | adam_lite | 25 | 27 | high | fullsize | 23.52 | 0.06×0.47×1.31 | 66.76 | 0.00 | MIT (Menagerie: pndbotics_adam_lite) | `mujoco_menagerie/pndbotics_adam_lite/adam_lite.xml` |
| rainbow_robotics | rb | 6 | 12 | low | compact | 0.35 | 0.25×0.00×0.19 | 17.70 | 0.02 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/robotics-toolbox/xacro_generated/interbotix_descriptions/urdf/px100.urdf` |
| rainbow_robotics | rb | 7 | 13 | medium | tabletop | 0.56 | 0.36×0.00×0.25 | 24.09 | 0.02 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/robotics-toolbox/xacro_generated/interbotix_descriptions/urdf/px150.urdf` |
| rainbow_robotics | rb | 7 | 13 | medium | tabletop | 0.58 | 0.36×0.00×0.25 | 23.49 | 0.02 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/robotics-toolbox/xacro_generated/interbotix_descriptions/urdf/rx150.urdf` |
| rainbow_robotics | rb | 7 | 13 | medium | tabletop | 0.54 | 0.41×0.00×0.30 | 23.77 | 0.02 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/robotics-toolbox/xacro_generated/interbotix_descriptions/urdf/rx200.urdf` |
| rainbow_robotics | rb | 7 | 13 | medium | tabletop | 1.34 | 0.54×0.00×0.43 | 23.68 | 0.04 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/robotics-toolbox/xacro_generated/interbotix_descriptions/urdf/vx300.urdf` |
| rainbow_robotics | rb | 7 | 13 | low | tabletop | 0.41 | 0.41×0.00×0.31 | 23.82 | 0.02 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/robotics-toolbox/xacro_generated/interbotix_descriptions/urdf/wx200.urdf` |
| rainbow_robotics | rb | 7 | 13 | low | tabletop | 0.42 | 0.46×0.00×0.36 | 24.09 | 0.02 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/robotics-toolbox/xacro_generated/interbotix_descriptions/urdf/wx250.urdf` |
| rainbow_robotics | rb | 8 | 14 | medium | tabletop | 1.43 | 0.54×0.00×0.43 | 29.97 | 0.04 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/robotics-toolbox/xacro_generated/interbotix_descriptions/urdf/vx300s.urdf` |
| rainbow_robotics | rb | 8 | 14 | medium | tabletop | 0.57 | 0.46×0.00×0.36 | 30.37 | 0.02 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/robotics-toolbox/xacro_generated/interbotix_descriptions/urdf/wx250s.urdf` |
| rainbow_robotics | rb | 18 | 33 | high | fullsize | 8.96 | 0.42×2.16×0.79 | 57.92 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2b_sim_upperbody_dynamics.urdf.urdf` |
| rainbow_robotics | rb | 18 | 34 | high | fullsize | 8.96 | 0.42×2.16×0.79 | 57.92 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2c_sim_upperbody_dynamics.urdf.urdf` |
| rainbow_robotics | rb | 42 | 71 | high | fullsize | 10.99 | 0.51×2.29×0.85 | 249.34 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2b_sim_upperbody_control.urdf.urdf` |
| rainbow_robotics | rb | 42 | 74 | high | fullsize | 10.99 | 0.51×2.29×0.87 | 249.34 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/r2_description/robots/r2c_sim_upperbody_control.urdf.urdf` |
| rainbow_robotics | rb | 44 | 46 | high | tabletop | 24.22 | 0.05×0.24×0.45 | 196.38 | 0.00 | MIT (Menagerie: toddlerbot_2xc) | `mujoco_menagerie/toddlerbot_2xc/toddlerbot_2xc.xml` |
| rethink | baxter | 14 | 26 | medium | fullsize | 4.86 | 0.80×1.98×0.75 | 62.55 | 0.00 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/robots/baxter/robot.xml` |
| rethink | baxter | 15 | 49 | high | fullsize | 28.64 | 0.92×2.04×0.82 | 65.69 | 0.00 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/bullet_data/baxter_description/urdf/baxter_arm.urdf` |
| rethink | baxter | 15 | 49 | high | fullsize | 28.64 | 0.92×2.04×0.82 | 65.69 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/baxter_description/urdf/rethinkBaxter.urdf` |
| rethink | baxter | 15 | 49 | high | fullsize | 28.64 | 0.92×2.04×0.82 | 65.69 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/oems/baxter_rethink_robotics/baxter_description/urdf/baxter.urdf` |
| rethink | sawyer | 7 | 10 | high | tabletop | 6.27 | 0.99×0.19×0.38 | 45.69 | 0.00 | Apache-2.0 (Menagerie: rethink_robotics_sawyer) | `mujoco_menagerie/rethink_robotics_sawyer/sawyer.xml` |
| rethink | sawyer | 7 | 8 | high | tabletop | 11.87 | 0.99×0.19×0.32 | 45.76 | 0.00 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/bullet_data/sawyer_description/urdf/sawyer_arm.urdf` |
| rethink | sawyer | 8 | 21 | high | fullsize | 19.59 | 1.07×0.19×0.59 | 51.87 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/sawyer_description/urdf/rethinkSawyer.urdf` |
| robotiq | robotiq | 1 | 11 | medium | compact | 1.52 | 0.00×0.19×0.18 | 0.70 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/robotiq/robotiq_2f_140_gripper_visualization/urdf/robotiq_arg2f_140_model.urdf` |
| robotiq | robotiq | 1 | 11 | medium | compact | 3.45 | 0.00×0.14×0.13 | 0.80 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/robotiq/robotiq_2f_85_gripper_visualization/urdf/robotiq_arg2f_85_model.urdf` |
| robotiq | robotiq | 6 | 12 | medium | compact | 2.10 | 0.18×0.02×0.11 | 7.74 | 0.00 | BSD-3-Clause (Menagerie: robotiq_2f85_v4) | `mujoco_menagerie/robotiq_2f85_v4/2f85.xml` |
| robotiq | robotiq | 6 | 9 | medium | compact | 0.51 | 0.00×0.19×0.27 | 0.16 | 0.00 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/grippers/robotiq_gripper_140.xml` |
| robotiq | robotiq | 8 | 15 | medium | compact | 1.15 | 0.14×0.00×0.12 | 10.42 | 0.00 | BSD-3-Clause (Menagerie: robotiq_2f85) | `mujoco_menagerie/robotiq_2f85/2f85.xml` |
| robotiq | robotiq | 11 | 15 | low | compact | 0.32 | 0.07×0.19×0.15 | 0.51 | 0.00 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/grippers/robotiq_gripper_s.xml` |
| robotis | dynamixel | 2 | 4 | medium | tabletop | 1.81 | 0.00×0.03×0.55 | 9.08 | 0.00 | Unknown (Menagerie: dynamixel_2r) | `mujoco_menagerie/dynamixel_2r/dynamixel_2r.xml` |
| robotis | op3 | 20 | 22 | high | tabletop | 16.81 | 0.04×0.36×0.41 | 125.66 | 0.00 | Apache-2.0 (Menagerie: robotis_op3) | `mujoco_menagerie/robotis_op3/op3.xml` |
| schunk | schunk | 1 | 5 | medium | compact | 0.52 | 0.00×0.01×0.20 | 0.00 | 0.03 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/random/xacro_generated/schunk_modular_robotics/schunk_description/urdf/schunk_pg70.urdf` |
| schunk | schunk | 2 | 5 | medium | compact | 1.14 | 0.00×0.00×0.19 | 10.47 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/random/xacro_generated/schunk_modular_robotics/schunk_description/urdf/schunk_pw70.urdf` |
| schunk | schunk | 6 | 9 | medium | tabletop | 1.46 | 0.00×0.01×0.81 | 34.86 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/random/xacro_generated/schunk_modular_robotics/schunk_description/urdf/schunk_lwa4p.urdf` |
| schunk | schunk | 7 | 11 | medium | fullsize | 0.64 | 0.00×0.00×1.14 | 37.31 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/random/xacro_generated/schunk_modular_robotics/schunk_description/urdf/schunk_lwa4d.urdf` |
| schunk | schunk | 7 | 10 | high | fullsize | 6.00 | 0.00×0.00×1.03 | 41.87 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/random/xacro_generated/schunk_modular_robotics/schunk_description/urdf/schunk_lwa4p_extended.urdf` |
| shadow_robot | shadow | 12 | 20 | medium | compact | 2.96 | 0.09×0.11×0.30 | 21.99 | 0.00 | Apache-2.0 (Menagerie: shadow_dexee) | `mujoco_menagerie/shadow_dexee/shadow_dexee.xml` |
| shadow_robot | shadow | 24 | 26 | medium | tabletop | 0.81 | 0.42×0.12×0.02 | 32.32 | 0.00 | Apache-2.0 (Menagerie: shadow_hand) | `mujoco_menagerie/shadow_hand/left_hand.xml` |
| softbank | nao | 25 | 79 | medium | tabletop | 0.92 | 0.27×0.28×0.57 | 56.88 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/data/humanoid/nao.urdf` |
| stanford | tidybot | 3 | 2 | low | unknown | 0.07 | 0.00×0.00×0.00 | 6.28 | 0.00 | MIT (Menagerie: stanford_tidybot) | `mujoco_menagerie/stanford_tidybot/base.xml` |
| stanford | tidybot | 18 | 23 | medium | fullsize | 2.39 | 0.14×0.02×1.63 | 55.64 | 0.00 | MIT (Menagerie: stanford_tidybot) | `mujoco_menagerie/stanford_tidybot/tidybot.xml` |
| staubli | staubli | 6 | 10 | medium | fullsize | 1.86 | 0.00×0.02×1.04 | 39.33 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/staubli/staubli_tx2_60_support/urdf/tx2_60.urdf` |
| staubli | staubli | 6 | 10 | medium | fullsize | 1.80 | 0.00×0.02×1.29 | 39.33 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/staubli/staubli_tx2_60_support/urdf/tx2_60l.urdf` |
| staubli | staubli | 6 | 10 | medium | fullsize | 2.75 | 0.05×0.05×1.43 | 39.49 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/staubli/staubli_tx2_90_support/urdf/tx2_90.urdf` |
| staubli | staubli | 6 | 10 | medium | fullsize | 2.79 | 0.05×0.05×1.63 | 39.49 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/staubli/staubli_tx2_90_support/urdf/tx2_90l.urdf` |
| staubli | staubli | 6 | 10 | medium | fullsize | 2.88 | 0.05×0.05×1.88 | 39.49 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/staubli/staubli_tx2_90_support/urdf/tx2_90xl.urdf` |
| staubli | staubli | 6 | 10 | medium | fullsize | 1.88 | 0.00×0.02×1.04 | 39.01 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/staubli/staubli_tx60_support/urdf/tx60.urdf` |
| staubli | staubli | 6 | 10 | medium | fullsize | 1.75 | 0.00×0.02×1.29 | 39.36 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/staubli/staubli_tx60_support/urdf/tx60l.urdf` |
| staubli | staubli | 6 | 10 | medium | fullsize | 2.05 | 0.05×0.05×1.43 | 39.49 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/staubli/staubli_tx90_support/urdf/tx90.urdf` |
| staubli | staubli | 6 | 10 | medium | fullsize | 1.88 | 0.05×0.05×1.63 | 39.49 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/staubli/staubli_tx90_support/urdf/tx90l.urdf` |
| staubli | staubli | 6 | 10 | medium | fullsize | 1.91 | 0.05×0.05×1.88 | 39.49 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/staubli/staubli_tx90_support/urdf/tx90xl.urdf` |
| tetheria | aero_hand | 16 | 23 | medium | compact | 1.52 | 0.20×0.15×0.07 | 24.69 | 0.00 | Apache-2.0 (Menagerie: tetheria_aero_hand_open) | `mujoco_menagerie/tetheria_aero_hand_open/left_hand.xml` |
| the_robot_studio | so101 | 6 | 9 | high | tabletop | 6.26 | 0.32×0.04×0.26 | 21.43 | 0.00 | Apache-2.0 (Menagerie: robotstudio_so101) | `mujoco_menagerie/robotstudio_so101/so101.xml` |
| the_robot_studio | trs_so_arm | 6 | 8 | medium | tabletop | 1.10 | 0.00×0.41×0.12 | 21.47 | 0.00 | Apache-2.0 (Menagerie: trs_so_arm100) | `mujoco_menagerie/trs_so_arm100/so_arm100.xml` |
| trossen | vx300 | 8 | 11 | medium | tabletop | 0.50 | 0.50×0.00×0.43 | 29.43 | 0.07 | BSD-3-Clause (Menagerie: trossen_vx300s) | `mujoco_menagerie/trossen_vx300s/vx300s.xml` |
| trossen | widowx | 7 | 10 | low | tabletop | 0.17 | 0.30×0.00×0.27 | 22.84 | 0.03 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/data/widowx/widowx.urdf` |
| trossen | wx250 | 8 | 10 | low | tabletop | 0.20 | 0.43×0.00×0.36 | 30.37 | 0.04 | BSD-3-Clause (Menagerie: trossen_wx250s) | `mujoco_menagerie/trossen_wx250s/wx250s.xml` |
| trossen | wxai | 16 | 19 | medium | fullsize | 0.75 | 1.40×0.05×0.21 | 48.35 | 0.18 | BSD-3-Clause (Menagerie: trossen_wxai) | `mujoco_menagerie/trossen_wxai/trossen_ai_bimanual.xml` |
| uc_berkeley | berkeley_humanoid | 12 | 14 | high | tabletop | 9.77 | 0.06×0.32×0.50 | 18.15 | 0.00 | BSD-3-Clause (Menagerie: berkeley_humanoid) | `mujoco_menagerie/berkeley_humanoid/berkeley_humanoid.xml` |
| ufactory | lite6 | 6 | 8 | medium | tabletop | 1.26 | 0.09×0.00×0.44 | 33.71 | 0.00 | BSD-3-Clause (Menagerie: ufactory_lite6) | `mujoco_menagerie/ufactory_lite6/lite6.xml` |
| ufactory | lite6 | 8 | 11 | medium | tabletop | 1.43 | 0.09×0.00×0.44 | 33.71 | 0.02 | BSD-3-Clause (Menagerie: ufactory_lite6) | `mujoco_menagerie/ufactory_lite6/lite6_gripper_wide.xml` |
| ufactory | xarm | 6 | 8 | medium | tabletop | 2.57 | 0.21×0.00×0.55 | 50.81 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/examples/pybullet/gym/pybullet_data/xarm/xarm6_robot.urdf` |
| ufactory | xarm | 6 | 8 | medium | tabletop | 2.57 | 0.21×0.00×0.55 | 50.81 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/examples/pybullet/gym/pybullet_data/xarm/xarm6_robot_white.urdf` |
| ufactory | xarm | 6 | 8 | medium | compact | 0.92 | 0.00×0.14×0.10 | 5.10 | 0.00 | BSD-3-Clause (Menagerie: ufactory_xarm7) | `mujoco_menagerie/ufactory_xarm7/hand.xml` |
| ufactory | xarm | 7 | 15 | high | tabletop | 5.05 | 0.21×0.14×0.55 | 51.66 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/examples/pybullet/gym/pybullet_data/xarm/xarm6_with_gripper.urdf` |
| ufactory | xarm | 7 | 9 | medium | tabletop | 0.74 | 0.21×0.00×0.56 | 38.24 | 0.00 | BSD-3-Clause (Menagerie: ufactory_xarm7) | `mujoco_menagerie/ufactory_xarm7/xarm7_nohand.xml` |
| ufactory | xarm | 7 | 10 | medium | tabletop | 0.74 | 0.21×0.00×0.56 | 38.24 | 0.00 | MIT (Stanford VLL + UT Robot Perception, 2022) | `robosuite/robosuite/models/assets/robots/xarm7/robot.xml` |
| ufactory | xarm | 13 | 16 | medium | tabletop | 1.66 | 0.21×0.14×0.56 | 43.34 | 0.00 | BSD-3-Clause (Menagerie: ufactory_xarm7) | `mujoco_menagerie/ufactory_xarm7/xarm7.xml` |
| unitree | a1 | 6 | 10 | low | fullsize | 0.20 | 1.29×0.00×1.56 | 40.33 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/fanuc/fanuc_m20ia_support/urdf/m20ia10l.urdf` |
| unitree | a1 | 12 | 22 | high | tabletop | 6.03 | 0.37×0.26×0.40 | 34.49 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/examples/pybullet/gym/pybullet_data/a1/a1.urdf` |
| unitree | a1 | 12 | 14 | medium | tabletop | 3.80 | 0.37×0.26×0.20 | 34.49 | 0.00 | BSD-3-Clause (Menagerie: unitree_a1) | `mujoco_menagerie/unitree_a1/a1.xml` |
| unitree | aliengo | 12 | 18 | medium | tabletop | 0.89 | 0.48×0.27×0.50 | 18.29 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/examples/pybullet/gym/pybullet_data/aliengo/aliengo.urdf` |
| unitree | g1 | 29 | 31 | high | fullsize | 7.08 | 0.20×0.30×1.05 | 102.68 | 0.00 | BSD-3-Clause (Menagerie: unitree_g1) | `mujoco_menagerie/unitree_g1/g1.xml` |
| unitree | g1 | 43 | 45 | high | fullsize | 11.33 | 0.37×0.31×1.05 | 127.17 | 0.00 | BSD-3-Clause (Menagerie: unitree_g1) | `mujoco_menagerie/unitree_g1/g1_with_hands.xml` |
| unitree | go1 | 12 | 14 | medium | tabletop | 3.72 | 0.38×0.25×0.21 | 35.37 | 0.00 | BSD-3-Clause (Menagerie: unitree_go1) | `mujoco_menagerie/unitree_go1/go1.xml` |
| unitree | go2 | 12 | 14 | medium | tabletop | 4.66 | 0.39×0.28×0.21 | 36.16 | 0.00 | BSD-3-Clause (Menagerie: unitree_go2) | `mujoco_menagerie/unitree_go2/go2.xml` |
| unitree | h1 | 19 | 21 | high | fullsize | 5.15 | 0.04×0.43×1.41 | 59.42 | 0.00 | BSD-3-Clause (Menagerie: unitree_h1) | `mujoco_menagerie/unitree_h1/h1.xml` |
| unitree | laikago | 12 | 13 | medium | tabletop | 3.05 | 0.27×0.21×0.58 | 0.00 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/examples/pybullet/gym/pybullet_data/laikago/laikago.urdf` |
| unitree | laikago | 12 | 17 | medium | tabletop | 3.05 | 0.27×0.46×0.60 | 0.00 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/examples/pybullet/gym/pybullet_data/laikago/laikago_toes.urdf` |
| unitree | laikago | 12 | 17 | medium | tabletop | 3.05 | 0.27×0.46×0.60 | 35.14 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/examples/pybullet/gym/pybullet_data/laikago/laikago_toes_limits.urdf` |
| unitree | laikago | 12 | 17 | medium | tabletop | 3.09 | 0.60×0.27×0.49 | 0.00 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/examples/pybullet/gym/pybullet_data/laikago/laikago_toes_zup.urdf` |
| unitree | laikago | 12 | 17 | low | tabletop | 0.19 | 0.60×0.27×0.49 | 0.00 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/examples/pybullet/gym/pybullet_data/laikago/laikago_toes_zup_lores.urdf` |
| unitree | z1 | 6 | 8 | medium | tabletop | 3.20 | 0.35×0.00×0.16 | 22.39 | 0.00 | BSD-3-Clause (Menagerie: unitree_z1) | `mujoco_menagerie/unitree_z1/z1.xml` |
| unitree | z1 | 7 | 9 | medium | tabletop | 4.41 | 0.44×0.00×0.16 | 23.91 | 0.00 | BSD-3-Clause (Menagerie: unitree_z1) | `mujoco_menagerie/unitree_z1/z1_gripper.xml` |
| universal_robots | ur10 | 6 | 8 | high | fullsize | 6.17 | 1.18×0.18×0.18 | 37.70 | 0.00 | BSD-3-Clause (Menagerie: universal_robots_ur10e) | `mujoco_menagerie/universal_robots_ur10e/ur10e.xml` |
| universal_robots | ur10 | 6 | 11 | high | fullsize | 9.06 | 1.18×0.26×0.13 | 69.12 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/ur10/ur10_robot.urdf` |
| universal_robots | ur10 | 6 | 11 | high | fullsize | 9.23 | 1.18×0.26×0.13 | 69.12 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/ur_description/urdf/universalUR10.urdf` |
| universal_robots | ur16 | 6 | 11 | high | tabletop | 9.89 | 0.84×0.29×0.18 | 69.12 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/universal_robots/ur_description/urdf/ur16e.urdf` |
| universal_robots | ur3 | 6 | 11 | high | tabletop | 5.45 | 0.46×0.19×0.15 | 69.12 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/ur_description/urdf/universalUR3.urdf` |
| universal_robots | ur3 | 6 | 11 | high | tabletop | 6.95 | 0.46×0.19×0.15 | 37.70 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/robotics-toolbox/xacro_generated/ur_description/urdf/ur3.urdf` |
| universal_robots | ur3 | 6 | 11 | high | tabletop | 10.22 | 0.46×0.22×0.15 | 69.12 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/universal_robots/ur_description/urdf/ur3e.urdf` |
| universal_robots | ur5 | 6 | 8 | high | tabletop | 5.76 | 0.82×0.14×0.16 | 37.70 | 0.00 | BSD-3-Clause (Menagerie: universal_robots_ur5e) | `mujoco_menagerie/universal_robots_ur5e/ur5e.xml` |
| universal_robots | ur5 | 6 | 11 | high | tabletop | 6.66 | 0.82×0.19×0.09 | 37.70 | 0.00 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/ur5/ur5_gripper.urdf` |
| universal_robots | ur5 | 6 | 11 | high | tabletop | 6.28 | 0.82×0.19×0.09 | 69.12 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/ur_description/urdf/universalUR5.urdf` |
| universal_robots | ur5 | 6 | 11 | high | tabletop | 6.68 | 0.82×0.19×0.09 | 37.70 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/robotics-toolbox/xacro_generated/ur_description/urdf/ur5.urdf` |
| willow_garage | pr2 | 2 | 5 | low | compact | 0.06 | 0.20×0.02×0.00 | 8.55 | 0.00 | zlib (most files); see LICENSE.txt for exceptions | `bullet3/data/pr2_gripper.urdf` |
| willow_garage | pr2 | 32 | 50 | medium | fullsize | 3.11 | 1.23×0.55×1.24 | 39.49 | 0.33 | varies by upstream repo (see README.md) | `robot-assets/urdfs/robots/pr2/pr2.urdf` |
| willow_garage | pr2 | 39 | 95 | medium | fullsize | 4.63 | 1.24×0.55×1.46 | 58.34 | 0.91 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/pr2_description/robots/willowgaragePR2.urdf` |
| willow_garage | pr2 | 39 | 88 | medium | fullsize | 3.71 | 1.24×0.55×1.29 | 58.34 | 0.91 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/robotics-toolbox/xacro_generated/pr2_description/robots/pr2.urdf` |
| wonik | allegro | 16 | 22 | low | compact | 0.40 | 0.19×0.18×0.02 | 25.54 | 0.00 | BSD-3-Clause (Menagerie: wonik_allegro) | `mujoco_menagerie/wonik_allegro/left_hand.xml` |
| yaskawa | yaskawa | 6 | 7 | high | tabletop | 9.25 | 0.44×0.00×0.68 | 40.07 | 0.00 | MIT (Daniella Tola, 2023) | `urdf_files_dataset/urdf_files/matlab/motoman_mh5_support/urdf/yaskawaMotomanMH5.urdf` |