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

base_joint (base base_link): { joint: rigid,}
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

base_coll (base_link) {shape: box, size: [0.78 0.58 0.3], Q: "t(0. 0.0 0.195)", contact:-2}
tower_coll (franka_spine) {shape: box, size: [0.16 0.3 1.1], Q: "t(0.07 0.0 0.55)", contact:-2}

Include: <left_fr3.g>
Prefix: false 

left_base_joint_origin (mount_link): { Q: "t(0.0369 0.05018 0.050885) E(-0.89334809 -0.17456074 -0.46334506)" }
left_base_joint (left_base_joint_origin left_base): { joint: rigid,}

Include: <right_fr3.g>
Prefix: false 

right_base_joint_origin (mount_link): { Q: "t(0.0369 -0.05018 0.050885) E(0.89334809 -0.17456074 0.46334506)" }
right_base_joint (right_base_joint_origin right_base): { joint: rigid,}

