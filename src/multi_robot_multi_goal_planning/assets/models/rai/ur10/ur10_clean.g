base_link: { multibody:true}
base_link_0(base_link): { shape: mesh, mesh: <ur_description/meshes/ur10/visual/Base.ply>, visual: True }
shoulder_pan_joint_origin(base_link): { rel: [0, 0, 0.1273, 1, 0, 0, 0] }
shoulder_pan_joint(shoulder_pan_joint_origin): { joint: hingeZ, limits: [-3.1415, 3.1415, 2.16, -1, 330], ctrl_limits: [2.16, -1, 330] }
shoulder_link(shoulder_pan_joint): { mass: 7.778, inertia: [0.0314743, 0.0314743, 0.0218756] }
shoulder_link_0(shoulder_link): { shape: mesh, mesh: <ur_description/meshes/ur10/visual/Shoulder.ply>, visual: True }
shoulder_lift_joint_origin(shoulder_link): { rel: [0, 0.220941, 0, 0.707107, 0, 0.707107, 0] }
shoulder_lift_joint(shoulder_lift_joint_origin): { joint: hingeY, limits: [-2.1415, 2.1415, 2.16, -1, 330], ctrl_limits: [2.16, -1, 330] }
upper_arm_link(shoulder_lift_joint): { mass: 12.93, inertia: [0.421754, 0.421754, 0.0363656] }
upper_arm_link_0(upper_arm_link): { shape: mesh, mesh: <ur_description/meshes/ur10/visual/UpperArm.ply>, visual: True }
elbow_joint_origin(upper_arm_link): { rel: [0, -0.1719, 0.612, 1, 0, 0, 0] }
elbow_joint(elbow_joint_origin): { joint: hingeY, limits: [-3.1415, 3.1415, 3.15, -1, 150], ctrl_limits: [3.15, -1, 150] }
forearm_link(elbow_joint): { mass: 3.87, inertia: [0.11107, 0.11107, 0.0108844] }
forearm_link_0(forearm_link): { shape: mesh, mesh: <ur_description/meshes/ur10/visual/Forearm.ply>, visual: True }
wrist_1_joint_origin(forearm_link): { rel: [0, 0, 0.5723, 0.707107, 0, 0.707107, 0] }
wrist_1_joint(wrist_1_joint_origin): { joint: hingeY, limits: [-3.1415, 3.1415, 3.2, -1, 54], ctrl_limits: [3.2, -1, 54] }
wrist_1_link(wrist_1_joint): { mass: 1.96, inertia: [0.00510825, 0.00510825, 0.0055125] }
wrist_1_link_0(wrist_1_link): { shape: mesh, mesh: <ur_description/meshes/ur10/visual/Wrist1.ply>, visual: True }
wrist_2_joint_origin(wrist_1_link): { rel: [0, 0.1149, 0, 1, 0, 0, 0] }
wrist_2_joint(wrist_2_joint_origin): { joint: hingeZ, limits: [-3.1415, 3.1415, 3.2, -1, 54], ctrl_limits: [3.2, -1, 54] }
wrist_2_link(wrist_2_joint): { mass: 1.96, inertia: [0.00510825, 0.00510825, 0.0055125] }
wrist_2_link_0(wrist_2_link): { shape: mesh, mesh: <ur_description/meshes/ur10/visual/Wrist2.ply>, visual: True }
wrist_3_joint_origin(wrist_2_link): { rel: [0, 0, 0.1157, 1, 0, 0, 0] }
wrist_3_joint(wrist_3_joint_origin): { joint: hingeY, limits: [-3.1415, 3.1415, 3.2, -1, 54], ctrl_limits: [3.2, -1, 54] }
wrist_3_link(wrist_3_joint): { mass: 0.202, inertia: [0.000526462, 0.000526462, 0.000568125] }
wrist_3_link_0(wrist_3_link): { shape: mesh, mesh: <ur_description/meshes/ur10/visual/Wrist3.ply>, visual: True }
ee_fixed_joint_origin(wrist_3_link): { rel: [0, 0.0922, 0, 0.707107, 0, 0, 0.707107] }
ee_fixed_joint(ee_fixed_joint_origin): { joint: rigid }
ee_link(ee_fixed_joint): {  }
