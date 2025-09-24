base_link: {}
base_link_inertia: { mass: 4.0, inertia: [0.00443333156 0.0 0.0 0.00443333156 0.0 0.0072],}
base_link_inertia_0 (base_link_inertia): { rel: "E(0 0 3.141592653589793)", mesh: <ur_description/meshes/ur5e/visual/base.dae>, color: [0.7 0.7 0.7 1.0], visual: true }
shoulder_link: { mass: 3.7, inertia: [0.010267495893 0.0 0.0 0.010267495893 0.0 0.00666],}
shoulder_link_0 (shoulder_link): { rel: "E(0 0 3.141592653589793)", mesh: <ur_description/meshes/ur5e/visual/shoulder.dae>, color: [0.7 0.7 0.7 1.0], visual: true }
upper_arm_link: { mass: 8.393, inertia: [0.1338857818623325 0.0 0.0 0.1338857818623325 0.0 0.0151074],}
upper_arm_link_0 (upper_arm_link): { rel: "t(0 0 0.138) E(1.5707963267948966 0 -1.5707963267948966)", mesh: <ur_description/meshes/ur5e/visual/upperarm.dae>, color: [0.7 0.7 0.7 1.0], visual: true }
forearm_link: { mass: 2.275, inertia: [0.031209355099586295 0.0 0.0 0.031209355099586295 0.0 0.004095],}
forearm_link_0 (forearm_link): { rel: "t(0 0 0.007) E(1.5707963267948966 0 -1.5707963267948966)", mesh: <ur_description/meshes/ur5e/visual/forearm.dae>, color: [0.7 0.7 0.7 1.0], visual: true }
wrist_1_link: { mass: 1.219, inertia: [0.0025598989760400002 0.0 0.0 0.0025598989760400002 0.0 0.0021942],}
wrist_1_link_0 (wrist_1_link): { rel: "t(0 0 -0.127) E(1.5707963267948966 0 0)", mesh: <ur_description/meshes/ur5e/visual/wrist1.dae>, color: [0.7 0.7 0.7 1.0], visual: true }
wrist_2_link: { mass: 1.219, inertia: [0.0025598989760400002 0.0 0.0 0.0025598989760400002 0.0 0.0021942],}
wrist_2_link_0 (wrist_2_link): { rel: [0 0 -0.0997], mesh: <ur_description/meshes/ur5e/visual/wrist2.dae>, color: [0.7 0.7 0.7 1.0], visual: true }
wrist_3_link: { mass: 0.1879, inertia: [9.890410052167731e-05 0.0 0.0 9.890410052167731e-05 0.0 0.0001321171875],}
wrist_3_link_0 (wrist_3_link): { rel: "t(0 0 -0.0989) E(1.5707963267948966 0 0)", mesh: <ur_description/meshes/ur5e/visual/wrist3.dae>, color: [0.7 0.7 0.7 1.0], visual: true }
base: {}
flange: {}
tool0: {}
base_link-base_link_inertia_origin (base_link): { Q: "E(0 0 3.141592653589793)" }
base_link-base_link_inertia (base_link-base_link_inertia_origin base_link_inertia): { joint: rigid,}
shoulder_pan_joint_origin (base_link_inertia): { Q: [0 0 0.1625] }
shoulder_pan_joint (shoulder_pan_joint_origin shoulder_link): { joint: hingeZ, limits: [-6.283185307179586 6.283185307179586], ctrl_limits: [3.141592653589793 -1 150.0],}
shoulder_lift_joint_origin (shoulder_link): { Q: "E(1.570796327 0 0)" }
shoulder_lift_joint (shoulder_lift_joint_origin upper_arm_link): { joint: hingeZ, limits: [-6.283185307179586 6.283185307179586], ctrl_limits: [3.141592653589793 -1 150.0],}
elbow_joint_origin (upper_arm_link): { Q: [-0.425 0 0] }
elbow_joint (elbow_joint_origin forearm_link): { joint: hingeZ, limits: [-3.141592653589793 3.141592653589793], ctrl_limits: [3.141592653589793 -1 150.0],}
wrist_1_joint_origin (forearm_link): { Q: [-0.3922 0 0.1333] }
wrist_1_joint (wrist_1_joint_origin wrist_1_link): { joint: hingeZ, limits: [-6.283185307179586 6.283185307179586], ctrl_limits: [3.141592653589793 -1 28.0],}
wrist_2_joint_origin (wrist_1_link): { Q: "t(0 -0.0997 -2.044881182297852e-11) E(1.570796327 0 0)" }
wrist_2_joint (wrist_2_joint_origin wrist_2_link): { joint: hingeZ, limits: [-6.283185307179586 6.283185307179586], ctrl_limits: [3.141592653589793 -1 28.0],}
wrist_3_joint_origin (wrist_2_link): { Q: "t(0 0.0996 -2.042830148012698e-11) E(1.570796326589793 3.141592653589793 3.141592653589793)" }
wrist_3_joint (wrist_3_joint_origin wrist_3_link): { joint: hingeZ, limits: [-6.283185307179586 6.283185307179586], ctrl_limits: [3.141592653589793 -1 28.0],}
base_link-base_fixed_joint_origin (base_link): { Q: "E(0 0 3.141592653589793)" }
base_link-base_fixed_joint (base_link-base_fixed_joint_origin base): { joint: rigid,}
wrist_3-flange_origin (wrist_3_link): { Q: "E(0 -1.5707963267948966 -1.5707963267948966)" }
wrist_3-flange (wrist_3-flange_origin flange): { joint: rigid,}
flange-tool0_origin (flange): { Q: "E(1.5707963267948966 0 1.5707963267948966)" }
flange-tool0 (flange-tool0_origin tool0): { joint: rigid,}
