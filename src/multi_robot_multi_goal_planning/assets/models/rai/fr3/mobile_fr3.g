base: {}
base_link: {}
base_link_0 (base_link): { mesh: 'meshes/robots/tmrv0_2/visual/tmrv0_2.dae', color: [1.0 1.0 1.0 1.0], visual: true }
base_inertia: { mass: 88.954183, inertia: [3.245524271774 0.007060314402 -0.134604220439 4.162370636125 -0.010770547630 5.841473351812],}
caster_front_left_fixed_link: { mass: 0.7, inertia: [0.00049538079 -0.00000042663 0.00020926011 0.00070424109 -0.00000050221 0.00048449394],}
caster_front_left_fixed_link_0 (caster_front_left_fixed_link): { rel: [0 0 -0.0015], shape: box
  size: [0.04 0.03 0.003 0], color: [0.3 0.3 0.3 1.0], visual: true }
caster_front_left_steering_link: { mass: 0.01, inertia: [0.0001 0.0 0.0 0.0001 0.0 0.0001],}
caster_front_left_link: { mass: 0.001, inertia: [0.0001 0.0 0.0 0.0001 0.0 0.0001],}
caster_front_left_link_0 (caster_front_left_link): { rel: "E(1.5707963267948966 0 0)", shape: cylinder
  size: [0 0 0.015 0.03], color: [1.0 0.5 0.0 1.0], visual: true }
argo_drive_front_fixed_link: { mass: 11.5, inertia: [0.07060497424 -0.00025482709 -6.283102e-05 0.03433835031 0.00249596715 0.08926325955],}
argo_drive_front_fixed_link_0 (argo_drive_front_fixed_link): { rel: [0 0.039999999999999994 0.0345], shape: box
  size: [0.17 0.25 0.077 0], color: [0.3 0.3 0.3 1.0], visual: true }
argo_drive_front_steering_link: { mass: 2.0, inertia: [0.1 0.0 0.0 0.1 0.0 0.1],}
argo_drive_front_link: { mass: 2.5, inertia: [0.0018958333333333336 0 0 0.0018958333333333336 0 0.0031250000000000006],}
argo_drive_front_link_0 (argo_drive_front_link): { rel: "E(1.5707963267948966 0 0)", shape: cylinder
  size: [0 0 0.04 0.05], color: [0.8 0.6 0.2 1.0], visual: true }
rocker_arm_link: { mass: 5.183973, inertia: [0.077004518565 -0.000040505686 -0.000026300402 0.030693821144 0.002939677599 0.083485865094],}
rocker_arm_link_0 (rocker_arm_link): { rel: "E(1.5707963267948966 0 1.5707963267948966)", shape: cylinder
  size: [0 0 0.0225 0.015], color: [0.3 0.3 0.3 1.0], visual: true }
caster_rear_right_fixed_link: { mass: 0.7, inertia: [0.00049538079 -0.00000042663 0.00020926011 0.00070424109 -0.00000050221 0.00048449394],}
caster_rear_right_fixed_link_0 (caster_rear_right_fixed_link): { rel: [0 0 -0.0015], shape: box
  size: [0.04 0.03 0.003 0], color: [0.3 0.3 0.3 1.0], visual: true }
caster_rear_right_steering_link: { mass: 0.01, inertia: [0.0001 0.0 0.0 0.0001 0.0 0.0001],}
caster_rear_right_link: { mass: 0.001, inertia: [0.0001 0.0 0.0 0.0001 0.0 0.0001],}
caster_rear_right_link_0 (caster_rear_right_link): { rel: "E(1.5707963267948966 0 0)", shape: cylinder
  size: [0 0 0.015 0.03], color: [1.0 0.5 0.0 1.0], visual: true }
argo_drive_rear_fixed_link: { mass: 11.5, inertia: [0.07060497424 -0.00025482709 6.283102e-05 0.03433835031 -0.00249596715 0.08926325955],}
argo_drive_rear_fixed_link_0 (argo_drive_rear_fixed_link): { rel: [0 -0.039999999999999994 0.0345], shape: box
  size: [0.17 0.25 0.077 0], color: [0.3 0.3 0.3 1.0], visual: true }
argo_drive_rear_steering_link: { mass: 2.0, inertia: [0.1 0.0 0.0 0.1 0.0 0.1],}
argo_drive_rear_link: { mass: 2.5, inertia: [0.0018958333333333336 0 0 0.0018958333333333336 0 0.0031250000000000006],}
argo_drive_rear_link_0 (argo_drive_rear_link): { rel: "E(1.5707963267948966 0 0)", shape: cylinder
  size: [0 0 0.04 0.05], color: [0.8 0.6 0.2 1.0], visual: true }
imu_mounting_point: {}
front_mounting_point: {}
rear_mounting_point: {}
right_mounting_point: {}
left_mounting_point: {}
lidar_front_mounting_point: {}
lidar_rear_mounting_point: {}
franka_spine: { mass: 21.909124, inertia: [3.338968145 -0.000642887 0.299962762 3.324560297 -0.004488422 0.281489258],}
franka_spine_0 (franka_spine): { mesh: 'meshes/accessories/franka_spine/visual/franka_spine.dae', visual: true }
franka_spine_support: { mass: 0.01, inertia: [0.0001 0.0 0.0 0.0001 0.0 0.0001],}
mount_link: { mass: 4.093846, inertia: [0.0224318121 1.38076e-06 -0.0015719371 0.0164193182 -1.442621e-06 0.0217407036],}
mount_link_0 (mount_link): { mesh: 'meshes/accessories/fr3_duo_mount/visual/fr3_duo_mount.dae', visual: true }
mount_link_0 (mount_link): { rel: [0 0 0.068], mesh: 'meshes/accessories/fr3_duo_mount/visual/fr3_duo_cover.dae', visual: true }
head_link: { mass: 1.0, inertia: [0.01 0.0 0.0 0.01 0.0 0.01],}
head_link_0 (head_link): { mesh: 'meshes/accessories/fr3_duo_head/visual/fr3_duo_head.dae', visual: true }
head_camera_mounting_point: { mass: 0.001, inertia: [0.001 0.0 0.0 0.001 0.0 0.001],}
left_base: {}
left_fr3v2_link0: { mass: 2.3966, inertia: [0.009 0.0 0.002 0.0115 0.0 0.0085],}
left_fr3v2_link0_0 (left_fr3v2_link0): { mesh: 'meshes/robots/fr3v2/visual/link0.dae', visual: true }
left_fr3v2_link0_sc: {}
left_fr3v2_link0_accelerometer_top: {}
left_fr3v2_link0_accelerometer_bottom: {}
left_fr3v2_link1: { mass: 2.4377, inertia: [0.02324 0.000333 -0.000676 0.02146 0.003739 0.0055],}
left_fr3v2_link1_0 (left_fr3v2_link1): { mesh: 'meshes/robots/fr3v2/visual/link1.dae', visual: true }
left_fr3v2_link1_sc: {}
left_fr3v2_link1_accelerometer_top: {}
left_fr3v2_link1_accelerometer_bottom: {}
left_fr3v2_link2: { mass: 2.2375, inertia: [0.019026 0.001241 0.000499 0.00566 -0.001804 0.017114],}
left_fr3v2_link2_0 (left_fr3v2_link2): { mesh: 'meshes/robots/fr3v2/visual/link2.dae', visual: true }
left_fr3v2_link2_sc: {}
left_fr3v2_link2_accelerometer_top: {}
left_fr3v2_link2_accelerometer_bottom: {}
left_fr3v2_link3: { mass: 2.193082, inertia: [0.010225 0.001144 0.003497 0.012398 0.002156 0.006877],}
left_fr3v2_link3_0 (left_fr3v2_link3): { mesh: 'meshes/robots/fr3v2/visual/link3.dae', visual: true }
left_fr3v2_link3_sc: {}
left_fr3v2_link3_accelerometer_top: {}
left_fr3v2_link3_accelerometer_bottom: {}
left_fr3v2_link4: { mass: 2.181537, inertia: [0.008878 -0.003749 0.001139 0.007628 -0.00046 0.010295],}
left_fr3v2_link4_0 (left_fr3v2_link4): { mesh: 'meshes/robots/fr3v2/visual/link4.dae', visual: true }
left_fr3v2_link4_sc: {}
left_fr3v2_link4_accelerometer_top: {}
left_fr3v2_link4_accelerometer_bottom: {}
left_fr3v2_link5: { mass: 2.186881, inertia: [0.031874 0.000134 -0.000265 0.029674 -0.004734 0.004751],}
left_fr3v2_link5_0 (left_fr3v2_link5): { mesh: 'meshes/robots/fr3v2/visual/link5.dae', visual: true }
left_fr3v2_link5_sc: {}
left_fr3v2_link5_accelerometer_top: {}
left_fr3v2_link5_accelerometer_bottom: {}
left_fr3v2_link6: { mass: 1.579681, inertia: [0.00286 -0.000553 -0.000638 0.00391 -0.00024 0.00453],}
left_fr3v2_link6_0 (left_fr3v2_link6): { mesh: 'meshes/robots/fr3v2/visual/link6.dae', visual: true }
left_fr3v2_link6_sc: {}
left_fr3v2_link7: { mass: 0.694841, inertia: [0.000899 0.0 -1.3e-05 0.001099 -2.2e-05 0.000702],}
left_fr3v2_link7_0 (left_fr3v2_link7): { mesh: 'meshes/robots/fr3v2/visual/link7.dae', visual: true }
left_fr3v2_link7_sc: {}
left_fr3v2_link8: {}
right_base: {}
right_fr3v2_link0: { mass: 2.3966, inertia: [0.009 0.0 0.002 0.0115 0.0 0.0085],}
right_fr3v2_link0_0 (right_fr3v2_link0): { mesh: 'meshes/robots/fr3v2/visual/link0.dae', visual: true }
right_fr3v2_link0_sc: {}
right_fr3v2_link0_accelerometer_top: {}
right_fr3v2_link0_accelerometer_bottom: {}
right_fr3v2_link1: { mass: 2.4377, inertia: [0.02324 0.000333 -0.000676 0.02146 0.003739 0.0055],}
right_fr3v2_link1_0 (right_fr3v2_link1): { mesh: 'meshes/robots/fr3v2/visual/link1.dae', visual: true }
right_fr3v2_link1_sc: {}
right_fr3v2_link1_accelerometer_top: {}
right_fr3v2_link1_accelerometer_bottom: {}
right_fr3v2_link2: { mass: 2.2375, inertia: [0.019026 0.001241 0.000499 0.00566 -0.001804 0.017114],}
right_fr3v2_link2_0 (right_fr3v2_link2): { mesh: 'meshes/robots/fr3v2/visual/link2.dae', visual: true }
right_fr3v2_link2_sc: {}
right_fr3v2_link2_accelerometer_top: {}
right_fr3v2_link2_accelerometer_bottom: {}
right_fr3v2_link3: { mass: 2.193082, inertia: [0.010225 0.001144 0.003497 0.012398 0.002156 0.006877],}
right_fr3v2_link3_0 (right_fr3v2_link3): { mesh: 'meshes/robots/fr3v2/visual/link3.dae', visual: true }
right_fr3v2_link3_sc: {}
right_fr3v2_link3_accelerometer_top: {}
right_fr3v2_link3_accelerometer_bottom: {}
right_fr3v2_link4: { mass: 2.181537, inertia: [0.008878 -0.003749 0.001139 0.007628 -0.00046 0.010295],}
right_fr3v2_link4_0 (right_fr3v2_link4): { mesh: 'meshes/robots/fr3v2/visual/link4.dae', visual: true }
right_fr3v2_link4_sc: {}
right_fr3v2_link4_accelerometer_top: {}
right_fr3v2_link4_accelerometer_bottom: {}
right_fr3v2_link5: { mass: 2.186881, inertia: [0.031874 0.000134 -0.000265 0.029674 -0.004734 0.004751],}
right_fr3v2_link5_0 (right_fr3v2_link5): { mesh: 'meshes/robots/fr3v2/visual/link5.dae', visual: true }
right_fr3v2_link5_sc: {}
right_fr3v2_link5_accelerometer_top: {}
right_fr3v2_link5_accelerometer_bottom: {}
right_fr3v2_link6: { mass: 1.579681, inertia: [0.00286 -0.000553 -0.000638 0.00391 -0.00024 0.00453],}
right_fr3v2_link6_0 (right_fr3v2_link6): { mesh: 'meshes/robots/fr3v2/visual/link6.dae', visual: true }
right_fr3v2_link6_sc: {}
right_fr3v2_link7: { mass: 0.694841, inertia: [0.000899 0.0 -1.3e-05 0.001099 -2.2e-05 0.000702],}
right_fr3v2_link7_0 (right_fr3v2_link7): { mesh: 'meshes/robots/fr3v2/visual/link7.dae', visual: true }
right_fr3v2_link7_sc: {}
right_fr3v2_link8: {}
base_joint (base base_link): { joint: transXYPhi, limits: [-1 1 -1 1 -4 4]}
base_inertia_joint (base_link base_inertia): { joint: rigid,}
caster_front_left_fixed_joint_origin (base_link): { Q: [0.3 0.2 0.08] }
caster_front_left_fixed_joint (caster_front_left_fixed_joint_origin caster_front_left_fixed_link): { joint: rigid,}
caster_front_left_steering_joint (caster_front_left_fixed_link caster_front_left_steering_link): { joint: hingeZ, ctrl_limits: [20 -1 500],}
caster_front_left_joint_origin (caster_front_left_steering_link): { Q: [-0.035 0 -0.05] }
caster_front_left_joint (caster_front_left_joint_origin caster_front_left_link): { joint: hingeY, ctrl_limits: [20 -1 500],}
argo_drive_front_fixed_joint_origin (base_link): { Q: [0.3 -0.2 0.05] }
argo_drive_front_fixed_joint (argo_drive_front_fixed_joint_origin argo_drive_front_fixed_link): { joint: rigid,}
tmrv0_2_joint_0 (argo_drive_front_fixed_link argo_drive_front_steering_link): { joint: hingeZ, ctrl_limits: [20 -1 500],}
tmrv0_2_joint_1 (argo_drive_front_steering_link argo_drive_front_link): { joint: hingeY, ctrl_limits: [20 -1 500],}
rocker_arm_joint_origin (base_link): { Q: [-0.3 0.0 0.0845] }
rocker_arm_joint (rocker_arm_joint_origin rocker_arm_link): { joint: hingeX, limits: [-0.16 0.18], ctrl_limits: [20 -1 500],}
caster_rear_right_fixed_joint_origin (rocker_arm_link): { Q: [0.0 -0.2 -0.004500000000000004] }
caster_rear_right_fixed_joint (caster_rear_right_fixed_joint_origin caster_rear_right_fixed_link): { joint: rigid,}
caster_rear_right_steering_joint (caster_rear_right_fixed_link caster_rear_right_steering_link): { joint: hingeZ, ctrl_limits: [20 -1 500],}
caster_rear_right_joint_origin (caster_rear_right_steering_link): { Q: [-0.035 0 -0.05] }
caster_rear_right_joint (caster_rear_right_joint_origin caster_rear_right_link): { joint: hingeY, ctrl_limits: [20 -1 500],}
argo_drive_rear_fixed_joint_origin (rocker_arm_link): { Q: [0.0 0.2 -0.0345] }
argo_drive_rear_fixed_joint (argo_drive_rear_fixed_joint_origin argo_drive_rear_fixed_link): { joint: rigid,}
tmrv0_2_joint_2 (argo_drive_rear_fixed_link argo_drive_rear_steering_link): { joint: hingeZ, ctrl_limits: [20 -1 500],}
tmrv0_2_joint_3 (argo_drive_rear_steering_link argo_drive_rear_link): { joint: hingeY, ctrl_limits: [20 -1 500],}
imu_mounting_point_joint_origin (base_link): { Q: "t(0.260 0.0 0.1478) E(3.141592653589793 0 0)" }
imu_mounting_point_joint (imu_mounting_point_joint_origin imu_mounting_point): { joint: rigid,}
front_mounting_point_joint_origin (base_link): { Q: "t(0.380705 0.0 0.2345) E(3.141592653589793 0 0)" }
front_mounting_point_joint (front_mounting_point_joint_origin front_mounting_point): { joint: rigid,}
rear_mounting_point_joint_origin (base_link): { Q: "t(-0.380705 0.0 0.2345) E(3.141592653589793 0 3.141592653589793)" }
rear_mounting_point_joint (rear_mounting_point_joint_origin rear_mounting_point): { joint: rigid,}
right_mounting_point_joint_origin (base_link): { Q: "t(0.0 -0.272712 0.1145) E(3.141592653589793 0 -1.5707963267948966)" }
right_mounting_point_joint (right_mounting_point_joint_origin right_mounting_point): { joint: rigid,}
left_mounting_point_joint_origin (base_link): { Q: "t(0.0 0.272712 0.1145) E(3.141592653589793 0 1.5707963267948966)" }
left_mounting_point_joint (left_mounting_point_joint_origin left_mounting_point): { joint: rigid,}
lidar_front_mounting_point_joint_origin (base_link): { Q: "t(0.3275 0.2175 0.19065) E(0 3.141592653589793 2.356194490192345)" }
lidar_front_mounting_point_joint (lidar_front_mounting_point_joint_origin lidar_front_mounting_point): { joint: rigid,}
lidar_rear_mounting_point_joint_origin (base_link): { Q: "t(-0.3275 -0.2175 0.19065) E(0 3.141592653589793 5.497787143782138)" }
lidar_rear_mounting_point_joint (lidar_rear_mounting_point_joint_origin lidar_rear_mounting_point): { joint: rigid,}
franka_spine_fixed_joint_origin (base_link): { Q: [0.138289 0.0 0.350] }
franka_spine_fixed_joint (franka_spine_fixed_joint_origin franka_spine): { joint: rigid,}
franka_spine_vertical_joint_origin (franka_spine): { Q: [0.266711 0.0 0.1] }
franka_spine_vertical_joint (franka_spine_vertical_joint_origin franka_spine_support): { joint: transZ, limits: [0.0 0.85], ctrl_limits: [0.1 -1 100],}
mount_joint (franka_spine_support mount_link): { joint: rigid,}
head_joint_origin (mount_link): { Q: [0 0 0.167] }
head_joint (head_joint_origin head_link): { joint: rigid,}
head_camera_mounting_point_joint_origin (head_link): { Q: "t(0.0498 -0.02 0.2345) E(0 0.7156 0)" }
head_camera_mounting_point_joint (head_camera_mounting_point_joint_origin head_camera_mounting_point): { joint: rigid,}
left_base_joint_origin (mount_link): { Q: "t(0.0369 0.05018 0.050885) E(-0.89334809 -0.17456074 -0.46334506)" }
left_base_joint (left_base_joint_origin left_base): { joint: rigid,}
left_fr3v2_left_base_joint (left_base left_fr3v2_link0): { joint: rigid,}
left_fr3v2_link0_sc_joint (left_fr3v2_link0 left_fr3v2_link0_sc): { joint: rigid,}
left_fr3v2_link0_accelerometer_top_joint_origin (left_fr3v2_link0): { Q: "t(-0.017402 -0.007872 0.04715) E(3.1416 0.0 0.0)" }
left_fr3v2_link0_accelerometer_top_joint (left_fr3v2_link0_accelerometer_top_joint_origin left_fr3v2_link0_accelerometer_top): { joint: rigid,}
left_fr3v2_link0_accelerometer_bottom_joint_origin (left_fr3v2_link0): { Q: "t(-0.019739 0.0015506 0.05075) E(0.0 0.0 -1.5708)" }
left_fr3v2_link0_accelerometer_bottom_joint (left_fr3v2_link0_accelerometer_bottom_joint_origin left_fr3v2_link0_accelerometer_bottom): { joint: rigid,}
left_fr3v2_link1_sc_joint (left_fr3v2_link1 left_fr3v2_link1_sc): { joint: rigid,}
left_fr3v2_link1_accelerometer_top_joint_origin (left_fr3v2_link1): { Q: "t(0.017402 -0.093352 -0.00787) E(-1.5708 0.0 3.1416)" }
left_fr3v2_link1_accelerometer_top_joint (left_fr3v2_link1_accelerometer_top_joint_origin left_fr3v2_link1_accelerometer_top): { joint: rigid,}
left_fr3v2_link1_accelerometer_bottom_joint_origin (left_fr3v2_link1): { Q: "t(0.01974 -0.08975 0.00155) E(-1.5708 1.5708 0.0)" }
left_fr3v2_link1_accelerometer_bottom_joint (left_fr3v2_link1_accelerometer_bottom_joint_origin left_fr3v2_link1_accelerometer_bottom): { joint: rigid,}
left_fr3v2_joint1_origin (left_fr3v2_link0): { Q: [0 0 0.333] }
left_fr3v2_joint1 (left_fr3v2_joint1_origin left_fr3v2_link1): { joint: hingeZ, limits: [-2.9007400166666666 2.9007400166666666], ctrl_limits: [2.62 -1 87.0],}
left_fr3v2_link2_sc_joint (left_fr3v2_link2 left_fr3v2_link2_sc): { joint: rigid,}
left_fr3v2_link2_accelerometer_top_joint_origin (left_fr3v2_link2): { Q: "t(0.017403 -0.11515 0.00787) E(1.5708 0.0 3.1416)" }
left_fr3v2_link2_accelerometer_top_joint (left_fr3v2_link2_accelerometer_top_joint_origin left_fr3v2_link2_accelerometer_top): { joint: rigid,}
left_fr3v2_link2_accelerometer_bottom_joint_origin (left_fr3v2_link2): { Q: "t(0.019737 -0.11875 -0.00155) E(1.5708 -1.5708 0.0)" }
left_fr3v2_link2_accelerometer_bottom_joint (left_fr3v2_link2_accelerometer_bottom_joint_origin left_fr3v2_link2_accelerometer_bottom): { joint: rigid,}
left_fr3v2_joint2_origin (left_fr3v2_link1): { Q: "E(-1.570796326794897 0 0)" }
left_fr3v2_joint2 (left_fr3v2_joint2_origin left_fr3v2_link2): { joint: hingeZ, limits: [-1.8360900166666667 1.8360900166666667], ctrl_limits: [2.62 -1 87.0],}
left_fr3v2_link3_sc_joint (left_fr3v2_link3 left_fr3v2_link3_sc): { joint: rigid,}
left_fr3v2_link3_accelerometer_top_joint_origin (left_fr3v2_link3): { Q: "t(0.065099 0.077352 -0.00787) E(-1.5708 0.0 0.0)" }
left_fr3v2_link3_accelerometer_top_joint (left_fr3v2_link3_accelerometer_top_joint_origin left_fr3v2_link3_accelerometer_top): { joint: rigid,}
left_fr3v2_link3_accelerometer_bottom_joint_origin (left_fr3v2_link3): { Q: "t(0.062769 0.073756 0.00155) E(1.5708 1.5708 0.0)" }
left_fr3v2_link3_accelerometer_bottom_joint (left_fr3v2_link3_accelerometer_bottom_joint_origin left_fr3v2_link3_accelerometer_bottom): { joint: rigid,}
left_fr3v2_joint3_origin (left_fr3v2_link2): { Q: "t(0 -0.316 0) E(1.570796326794897 0 0)" }
left_fr3v2_joint3 (left_fr3v2_joint3_origin left_fr3v2_link3): { joint: hingeZ, limits: [-2.9007400166666666 2.9007400166666666], ctrl_limits: [2.62 -1 87.0],}
left_fr3v2_link4_sc_joint (left_fr3v2_link4 left_fr3v2_link4_sc): { joint: rigid,}
left_fr3v2_link4_accelerometer_top_joint_origin (left_fr3v2_link4): { Q: "t(-0.0651 0.046152 -0.00787) E(-1.5708 0.0 -3.1416)" }
left_fr3v2_link4_accelerometer_top_joint (left_fr3v2_link4_accelerometer_top_joint_origin left_fr3v2_link4_accelerometer_top): { joint: rigid,}
left_fr3v2_link4_accelerometer_bottom_joint_origin (left_fr3v2_link4): { Q: "t(-0.062775 0.049753 0.00155) E(-1.5708 1.5708 0.0)" }
left_fr3v2_link4_accelerometer_bottom_joint (left_fr3v2_link4_accelerometer_bottom_joint_origin left_fr3v2_link4_accelerometer_bottom): { joint: rigid,}
left_fr3v2_joint4_origin (left_fr3v2_link3): { Q: "t(0.0825 0 0) E(1.570796326794897 0 0)" }
left_fr3v2_joint4 (left_fr3v2_joint4_origin left_fr3v2_link4): { joint: hingeZ, limits: [-3.077020016666667 -0.11693708333333333], ctrl_limits: [2.62 -1 87.0],}
left_fr3v2_link5_sc_joint (left_fr3v2_link5 left_fr3v2_link5_sc): { joint: rigid,}
left_fr3v2_link5_accelerometer_top_joint_origin (left_fr3v2_link5): { Q: "t(-0.017363 0.085872 -0.00795) E(-1.5708 0.0 0.0)" }
left_fr3v2_link5_accelerometer_top_joint (left_fr3v2_link5_accelerometer_top_joint_origin left_fr3v2_link5_accelerometer_top): { joint: rigid,}
left_fr3v2_link5_accelerometer_bottom_joint_origin (left_fr3v2_link5): { Q: "t(-0.019742 0.082275 0.00146) E(1.5708 1.5708 0.0)" }
left_fr3v2_link5_accelerometer_bottom_joint (left_fr3v2_link5_accelerometer_bottom_joint_origin left_fr3v2_link5_accelerometer_bottom): { joint: rigid,}
left_fr3v2_joint5_origin (left_fr3v2_link4): { Q: "t(-0.0825 0.384 0) E(-1.570796326794897 0 0)" }
left_fr3v2_joint5 (left_fr3v2_joint5_origin left_fr3v2_link5): { joint: hingeZ, limits: [-2.87630335 2.87630335], ctrl_limits: [5.26 -1 12.0],}
left_fr3v2_link6_sc_joint (left_fr3v2_link6 left_fr3v2_link6_sc): { joint: rigid,}
left_fr3v2_joint6_origin (left_fr3v2_link5): { Q: "E(1.570796326794897 0 0)" }
left_fr3v2_joint6 (left_fr3v2_joint6_origin left_fr3v2_link6): { joint: hingeZ, limits: [0.43982265 4.62163335], ctrl_limits: [4.18 -1 12.0],}
left_fr3v2_link7_sc_joint_origin (left_fr3v2_link7): { Q: "E(0 0 0.7853981633974483)" }
left_fr3v2_link7_sc_joint (left_fr3v2_link7_sc_joint_origin left_fr3v2_link7_sc): { joint: rigid,}
left_fr3v2_joint7_origin (left_fr3v2_link6): { Q: "t(0.088 0 0) E(1.570796326794897 0 0)" }
left_fr3v2_joint7 (left_fr3v2_joint7_origin left_fr3v2_link7): { joint: hingeZ, limits: [-3.05083335 3.05083335], ctrl_limits: [5.26 -1 12.0],}
left_fr3v2_joint8_origin (left_fr3v2_link7): { Q: [0 0 0.107] }
left_fr3v2_joint8 (left_fr3v2_joint8_origin left_fr3v2_link8): { joint: rigid,}
right_base_joint_origin (mount_link): { Q: "t(0.0369 -0.05018 0.050885) E(0.89334809 -0.17456074 0.46334506)" }
right_base_joint (right_base_joint_origin right_base): { joint: rigid,}
right_fr3v2_right_base_joint (right_base right_fr3v2_link0): { joint: rigid,}
right_fr3v2_link0_sc_joint (right_fr3v2_link0 right_fr3v2_link0_sc): { joint: rigid,}
right_fr3v2_link0_accelerometer_top_joint_origin (right_fr3v2_link0): { Q: "t(-0.017402 -0.007872 0.04715) E(3.1416 0.0 0.0)" }
right_fr3v2_link0_accelerometer_top_joint (right_fr3v2_link0_accelerometer_top_joint_origin right_fr3v2_link0_accelerometer_top): { joint: rigid,}
right_fr3v2_link0_accelerometer_bottom_joint_origin (right_fr3v2_link0): { Q: "t(-0.019739 0.0015506 0.05075) E(0.0 0.0 -1.5708)" }
right_fr3v2_link0_accelerometer_bottom_joint (right_fr3v2_link0_accelerometer_bottom_joint_origin right_fr3v2_link0_accelerometer_bottom): { joint: rigid,}
right_fr3v2_link1_sc_joint (right_fr3v2_link1 right_fr3v2_link1_sc): { joint: rigid,}
right_fr3v2_link1_accelerometer_top_joint_origin (right_fr3v2_link1): { Q: "t(0.017402 -0.093352 -0.00787) E(-1.5708 0.0 3.1416)" }
right_fr3v2_link1_accelerometer_top_joint (right_fr3v2_link1_accelerometer_top_joint_origin right_fr3v2_link1_accelerometer_top): { joint: rigid,}
right_fr3v2_link1_accelerometer_bottom_joint_origin (right_fr3v2_link1): { Q: "t(0.01974 -0.08975 0.00155) E(-1.5708 1.5708 0.0)" }
right_fr3v2_link1_accelerometer_bottom_joint (right_fr3v2_link1_accelerometer_bottom_joint_origin right_fr3v2_link1_accelerometer_bottom): { joint: rigid,}
right_fr3v2_joint1_origin (right_fr3v2_link0): { Q: [0 0 0.333] }
right_fr3v2_joint1 (right_fr3v2_joint1_origin right_fr3v2_link1): { joint: hingeZ, limits: [-2.9007400166666666 2.9007400166666666], ctrl_limits: [2.62 -1 87.0],}
right_fr3v2_link2_sc_joint (right_fr3v2_link2 right_fr3v2_link2_sc): { joint: rigid,}
right_fr3v2_link2_accelerometer_top_joint_origin (right_fr3v2_link2): { Q: "t(0.017403 -0.11515 0.00787) E(1.5708 0.0 3.1416)" }
right_fr3v2_link2_accelerometer_top_joint (right_fr3v2_link2_accelerometer_top_joint_origin right_fr3v2_link2_accelerometer_top): { joint: rigid,}
right_fr3v2_link2_accelerometer_bottom_joint_origin (right_fr3v2_link2): { Q: "t(0.019737 -0.11875 -0.00155) E(1.5708 -1.5708 0.0)" }
right_fr3v2_link2_accelerometer_bottom_joint (right_fr3v2_link2_accelerometer_bottom_joint_origin right_fr3v2_link2_accelerometer_bottom): { joint: rigid,}
right_fr3v2_joint2_origin (right_fr3v2_link1): { Q: "E(-1.570796326794897 0 0)" }
right_fr3v2_joint2 (right_fr3v2_joint2_origin right_fr3v2_link2): { joint: hingeZ, limits: [-1.8360900166666667 1.8360900166666667], ctrl_limits: [2.62 -1 87.0],}
right_fr3v2_link3_sc_joint (right_fr3v2_link3 right_fr3v2_link3_sc): { joint: rigid,}
right_fr3v2_link3_accelerometer_top_joint_origin (right_fr3v2_link3): { Q: "t(0.065099 0.077352 -0.00787) E(-1.5708 0.0 0.0)" }
right_fr3v2_link3_accelerometer_top_joint (right_fr3v2_link3_accelerometer_top_joint_origin right_fr3v2_link3_accelerometer_top): { joint: rigid,}
right_fr3v2_link3_accelerometer_bottom_joint_origin (right_fr3v2_link3): { Q: "t(0.062769 0.073756 0.00155) E(1.5708 1.5708 0.0)" }
right_fr3v2_link3_accelerometer_bottom_joint (right_fr3v2_link3_accelerometer_bottom_joint_origin right_fr3v2_link3_accelerometer_bottom): { joint: rigid,}
right_fr3v2_joint3_origin (right_fr3v2_link2): { Q: "t(0 -0.316 0) E(1.570796326794897 0 0)" }
right_fr3v2_joint3 (right_fr3v2_joint3_origin right_fr3v2_link3): { joint: hingeZ, limits: [-2.9007400166666666 2.9007400166666666], ctrl_limits: [2.62 -1 87.0],}
right_fr3v2_link4_sc_joint (right_fr3v2_link4 right_fr3v2_link4_sc): { joint: rigid,}
right_fr3v2_link4_accelerometer_top_joint_origin (right_fr3v2_link4): { Q: "t(-0.0651 0.046152 -0.00787) E(-1.5708 0.0 -3.1416)" }
right_fr3v2_link4_accelerometer_top_joint (right_fr3v2_link4_accelerometer_top_joint_origin right_fr3v2_link4_accelerometer_top): { joint: rigid,}
right_fr3v2_link4_accelerometer_bottom_joint_origin (right_fr3v2_link4): { Q: "t(-0.062775 0.049753 0.00155) E(-1.5708 1.5708 0.0)" }
right_fr3v2_link4_accelerometer_bottom_joint (right_fr3v2_link4_accelerometer_bottom_joint_origin right_fr3v2_link4_accelerometer_bottom): { joint: rigid,}
right_fr3v2_joint4_origin (right_fr3v2_link3): { Q: "t(0.0825 0 0) E(1.570796326794897 0 0)" }
right_fr3v2_joint4 (right_fr3v2_joint4_origin right_fr3v2_link4): { joint: hingeZ, limits: [-3.077020016666667 -0.11693708333333333], ctrl_limits: [2.62 -1 87.0],}
right_fr3v2_link5_sc_joint (right_fr3v2_link5 right_fr3v2_link5_sc): { joint: rigid,}
right_fr3v2_link5_accelerometer_top_joint_origin (right_fr3v2_link5): { Q: "t(-0.017363 0.085872 -0.00795) E(-1.5708 0.0 0.0)" }
right_fr3v2_link5_accelerometer_top_joint (right_fr3v2_link5_accelerometer_top_joint_origin right_fr3v2_link5_accelerometer_top): { joint: rigid,}
right_fr3v2_link5_accelerometer_bottom_joint_origin (right_fr3v2_link5): { Q: "t(-0.019742 0.082275 0.00146) E(1.5708 1.5708 0.0)" }
right_fr3v2_link5_accelerometer_bottom_joint (right_fr3v2_link5_accelerometer_bottom_joint_origin right_fr3v2_link5_accelerometer_bottom): { joint: rigid,}
right_fr3v2_joint5_origin (right_fr3v2_link4): { Q: "t(-0.0825 0.384 0) E(-1.570796326794897 0 0)" }
right_fr3v2_joint5 (right_fr3v2_joint5_origin right_fr3v2_link5): { joint: hingeZ, limits: [-2.87630335 2.87630335], ctrl_limits: [5.26 -1 12.0],}
right_fr3v2_link6_sc_joint (right_fr3v2_link6 right_fr3v2_link6_sc): { joint: rigid,}
right_fr3v2_joint6_origin (right_fr3v2_link5): { Q: "E(1.570796326794897 0 0)" }
right_fr3v2_joint6 (right_fr3v2_joint6_origin right_fr3v2_link6): { joint: hingeZ, limits: [0.43982265 4.62163335], ctrl_limits: [4.18 -1 12.0],}
right_fr3v2_link7_sc_joint_origin (right_fr3v2_link7): { Q: "E(0 0 0.7853981633974483)" }
right_fr3v2_link7_sc_joint (right_fr3v2_link7_sc_joint_origin right_fr3v2_link7_sc): { joint: rigid,}
right_fr3v2_joint7_origin (right_fr3v2_link6): { Q: "t(0.088 0 0) E(1.570796326794897 0 0)" }
right_fr3v2_joint7 (right_fr3v2_joint7_origin right_fr3v2_link7): { joint: hingeZ, limits: [-3.05083335 3.05083335], ctrl_limits: [5.26 -1 12.0],}
right_fr3v2_joint8_origin (right_fr3v2_link7): { Q: [0 0 0.107] }
right_fr3v2_joint8 (right_fr3v2_joint8_origin right_fr3v2_link8): { joint: rigid,}
