base: {multibody: true}

base_link (base): { mass: 4.0, inertia: [0.00443333156 0.0 0.0 0.00443333156 0.0 0.0072],}
base_link_0 (base_link): { mesh: <meshes/ur5/visual/base.dae>, color: [0.7 0.7 0.7 1.0], visual: true }
shoulder_link: { mass: 3.7, inertia: [0.010267495893 0.0 0.0 0.010267495893 0.0 0.00666],}
shoulder_link_0 (shoulder_link): { mesh: <meshes/ur5/visual/shoulder.dae>, color: [0.7 0.7 0.7 1.0], visual: true }
upper_arm_link: { mass: 8.393, inertia: [0.22689067591036005 0.0 0.0 0.22689067591036005 0.0 0.0151074],}
upper_arm_link_0 (upper_arm_link): { mesh: <meshes/ur5/visual/upperarm.dae>, color: [0.7 0.7 0.7 1.0], visual: true }
forearm_link: { mass: 2.275, inertia: [0.04944331355599999 0.0 0.0 0.04944331355599999 0.0 0.004095],}
forearm_link_0 (forearm_link): { mesh: <meshes/ur5/visual/forearm.dae>, color: [0.7 0.7 0.7 1.0], visual: true }
wrist_1_link: { mass: 1.219, inertia: [0.11117275553087999 0.0 0.0 0.11117275553087999 0.0 0.21942],}
wrist_1_link_0 (wrist_1_link): { mesh: <meshes/ur5/visual/wrist1.dae>, color: [0.7 0.7 0.7 1.0], visual: true }
wrist_2_link: { mass: 1.219, inertia: [0.11117275553087999 0.0 0.0 0.11117275553087999 0.0 0.21942],}
wrist_2_link_0 (wrist_2_link): { mesh: <meshes/ur5/visual/wrist2.dae>, color: [0.7 0.7 0.7 1.0], visual: true }
wrist_3_link: { mass: 0.1879, inertia: [0.017136473145408 0.0 0.0 0.017136473145408 0.0 0.033822],}
wrist_3_link_0 (wrist_3_link): { mesh: <meshes/ur5/visual/wrist3.dae>, color: [0.7 0.7 0.7 1.0], visual: true }

#ee_link: {}
#tool0: {}
#fts_robotside: { mass: 0.65, inertia: [0.000661171875 0 0 0.000661171875 0 0.00117],}
#fts_robotside_0 (fts_robotside): { mesh: <meshes/robotiq_fts300.stl>, visual: true }
#robotiq_force_torque_frame_id: {}
#fts_toolside: {}
#robotiq_85_base_link: { mass: 0.636951, inertia: [0.000380 0.000000 0.000000 0.001110 0.000000 0.001171],}
#robotiq_85_base_link_0 (robotiq_85_base_link): { mesh: <meshes/robotiq/robotiq_85_base_link_fine.STL>, visual: true }
#robotiq_85_left_knuckle_link: { mass: 0.018491, inertia: [0.000009 -0.000001 0.000000 0.000001 0.000000 0.000010],}
#robotiq_85_left_knuckle_link_0 (robotiq_85_left_knuckle_link): { mesh: <meshes/robotiq/outer_knuckle_fine.STL>, visual: true }
#robotiq_85_right_knuckle_link: { mass: 0.018491, inertia: [0.000009 -0.000001 0.000000 0.000001 0.000000 0.000010],}
#robotiq_85_right_knuckle_link_0 (robotiq_85_right_knuckle_link): { mesh: <meshes/robotiq/outer_knuckle_fine.STL>, visual: true }
#robotiq_85_left_finger_link: { mass: 0.027309, inertia: [0.000003 -0.000002 0.000000 0.000021 0.000000 0.000020],}
#robotiq_85_left_finger_link_0 (robotiq_85_left_finger_link): { mesh: <meshes/robotiq/outer_finger_fine.STL>, visual: true }
#robotiq_85_right_finger_link: { mass: 0.027309, inertia: [0.000003 -0.000002 0.000000 0.000021 0.000000 0.000020],}
#robotiq_85_right_finger_link_0 (robotiq_85_right_finger_link): { mesh: <meshes/robotiq/outer_finger_fine.STL>, visual: true }
#robotiq_85_left_inner_knuckle_link: { mass: 0.029951, inertia: [0.000039 0.000000 0.000000 0.000005 0.000000 0.000035],}
#robotiq_85_left_inner_knuckle_link_0 (robotiq_85_left_inner_knuckle_link): { mesh: <meshes/robotiq/inner_knuckle_fine.STL>, visual: true }
#robotiq_85_right_inner_knuckle_link: { mass: 0.029951, inertia: [0.000039 0.000000 0.000000 0.000005 0.000000 0.000035],}
#robotiq_85_right_inner_knuckle_link_0 (robotiq_85_right_inner_knuckle_link): { mesh: <meshes/robotiq/inner_knuckle_fine.STL>, visual: true }
#robotiq_85_left_finger_tip_link: { mass: 0.019555, inertia: [0.000002 0.000000 0.000000 0.000005 0.000000 0.000006],}
#robotiq_85_left_finger_tip_link_0 (robotiq_85_left_finger_tip_link): { mesh: <meshes/robotiq/inner_finger_fine.STL>, visual: true }
#robotiq_85_right_finger_tip_link: { mass: 0.019555, inertia: [0.000002 0.000000 0.000000 0.000005 0.000000 0.000006],}
#robotiq_85_right_finger_tip_link_0 (robotiq_85_right_finger_tip_link): { mesh: <meshes/robotiq/inner_finger_fine.STL>, visual: true }

shoulder_pan_joint_origin (base_link): { Q: "t(0.0 0.0 0.089159) E(0 0.0 0)" }
shoulder_pan_joint (shoulder_pan_joint_origin shoulder_link): { joint: hingeZ, limits: [-3.14159265 3.14159265], ctrl_limits: [0.5 -1 150.0],}
shoulder_lift_joint_origin (shoulder_link): { Q: "t(0.0 0.13585 0.0) E(0.0 1.570796325 0.0)" }
shoulder_lift_joint (shoulder_lift_joint_origin upper_arm_link): { joint: hingeY, limits: [-3 0.0], ctrl_limits: [0.5 -1 150.0],}
elbow_joint_origin (upper_arm_link): { Q: "t(0.0 -0.1197 0.425) E(0.0 0.0 0.0)" }
elbow_joint (elbow_joint_origin forearm_link): { joint: hingeY, limits: [-3.14159265 3.14159265], ctrl_limits: [0.5 -1 150.0],}
wrist_1_joint_origin (forearm_link): { Q: "t(0.0 0.0 0.39225) E(0.0 1.570796325 0.0)" }
wrist_1_joint (wrist_1_joint_origin wrist_1_link): { joint: hingeY, limits: [-3.14159265 3.14159265], ctrl_limits: [0.5 -1 28.0],}
wrist_2_joint_origin (wrist_1_link): { Q: "t(0.0 0.093 0.0) E(0.0 0.0 0.0)" }
wrist_2_joint (wrist_2_joint_origin wrist_2_link): { joint: hingeZ, limits: [-3.14159265 3.14159265], ctrl_limits: [0.5 -1 28.0],}
wrist_3_joint_origin (wrist_2_link): { Q: "t(0.0 0.0 0.09465) E(0.0 0.0 0.0)" }
wrist_3_joint (wrist_3_joint_origin wrist_3_link): { joint: hingeY, limits: [-3.14159265 3.14159265], ctrl_limits: [0.5 -1 28.0],}
#ee_fixed_joint_origin (wrist_3_link): { Q: "t(0.0 0.0823 0.0) E(0.0 0.0 1.570796325)" }
#ee_fixed_joint (ee_fixed_joint_origin ee_link): { joint: rigid,}
#wrist_3_link-tool0_fixed_joint_origin (wrist_3_link): { Q: "t(0 0.0823 0) E(-1.570796325 0 0)" }
#wrist_3_link-tool0_fixed_joint (wrist_3_link-tool0_fixed_joint_origin tool0): { joint: rigid,}

#fts_fix_origin (ee_link): { Q: "t(0.035 0 0.0) E(0.0 0 -1.57)" }
#fts_fix (fts_fix_origin fts_robotside): { joint: rigid,}
#measurment_joint_origin (fts_robotside): { Q: [0 0 0.01625] }
#measurment_joint (measurment_joint_origin robotiq_force_torque_frame_id): { joint: rigid,}
#toolside_joint_origin (fts_robotside): { Q: [0 0 0.0375] }
#toolside_joint (toolside_joint_origin fts_toolside): { joint: rigid,}
#robotiq_85_base_joint_origin (fts_toolside): { Q: "t(0.0 0.0 -0.037) E(-1.57 0.0 0.0)" }
#robotiq_85_base_joint (robotiq_85_base_joint_origin robotiq_85_base_link): { joint: rigid,}
#robotiq_85_left_knuckle_joint_origin (robotiq_85_base_link): { Q: [0.0306011444260539 0 0.0627920162695395] }
#robotiq_85_left_knuckle_joint (robotiq_85_left_knuckle_joint_origin robotiq_85_left_knuckle_link): { joint: rigid,}
#robotiq_85_left_finger_joint_origin (robotiq_85_left_knuckle_link): { Q: [0.0316910442266543 0 -0.00193396375724605] }
#robotiq_85_left_finger_joint (robotiq_85_left_finger_joint_origin robotiq_85_left_finger_link): { joint: rigid,}
#robotiq_85_left_inner_knuckle_joint_origin (robotiq_85_base_link): { Q: [0.0127000000001501 0 0.0693074999999639] }
#robotiq_85_left_inner_knuckle_joint (robotiq_85_left_inner_knuckle_joint_origin robotiq_85_left_inner_knuckle_link): { joint: rigid,}
#robotiq_85_left_finger_tip_joint_origin (robotiq_85_left_inner_knuckle_link): { Q: [0.034585310861294 0 0.0454970193817975] }
#robotiq_85_left_finger_tip_joint (robotiq_85_left_finger_tip_joint_origin robotiq_85_left_finger_tip_link): { joint: rigid,}
#robotiq_85_right_inner_knuckle_joint_origin (robotiq_85_base_link): { Q: "t(-0.0126999999998499 0 0.0693075000000361) E(0 0 3.14159265358979)" }
#robotiq_85_right_inner_knuckle_joint (robotiq_85_right_inner_knuckle_joint_origin robotiq_85_right_inner_knuckle_link): { joint: rigid,}
#robotiq_85_right_finger_tip_joint_origin (robotiq_85_right_inner_knuckle_link): { Q: [0.0341060475457406 0 0.0458573878541688] }
#robotiq_85_right_finger_tip_joint (robotiq_85_right_finger_tip_joint_origin robotiq_85_right_finger_tip_link): { joint: rigid,}
#robotiq_85_right_knuckle_joint_origin (robotiq_85_base_link): { Q: "t(-0.0306011444258893 0 0.0627920162695395) E(0 0 3.14159265358979)" }
#robotiq_85_right_knuckle_joint (robotiq_85_right_knuckle_joint_origin robotiq_85_right_knuckle_link): { joint: rigid,}
#robotiq_85_right_finger_joint_origin (robotiq_85_right_knuckle_link): { Q: [0.0317095909367246 0 -0.0016013564954687] }
#robotiq_85_right_finger_joint (robotiq_85_right_finger_joint_origin robotiq_85_right_finger_link): { joint: rigid,}


coll1(shoulder_pan_joint_origin)  { shape:capsule color:[1.,1.,1.,.5] size:[.08 .07] Q:"t(0 0 0) d(0 0 1 0)", contact:-2 }
coll1_1(shoulder_lift_joint)      { shape:capsule color:[1.,1.,1.,.5] size:[.16 .065] Q:"t(0 -.07 0.0) d(90 1 0 0)", contact:-2 }
coll2(shoulder_lift_joint)        { shape:capsule color:[1.,1.,1.,.5] size:[.35 .065] Q:"t(0 .0 .2) d(0 0 1 0)", contact:-2 }
coll3(elbow_joint_origin)         { shape:capsule color:[1.,1.,1.,.5] size:[.16 .065] Q:"t(0 .08 0.0) d(90 1 0 0)", contact:-2 }
coll4(elbow_joint)                { shape:capsule color:[1.,1.,1.,.5] size:[.37 .06] Q:"t(0 .0 .19) d(0 0 1 0)", contact:-2 }
coll5(wrist_1_joint)              { shape:capsule color:[1.,1.,1.,.5] size:[.1 .047] Q:"t(0 0.02 -.0) d(90 1 0 0)", contact:-2 }
coll6(wrist_2_joint)              { shape:capsule color:[1.,1.,1.,.5] size:[.12 .05] Q:"t(0. -.0 0.02) d(90 0 0 1)", contact:-2 }
coll7(wrist_3_joint)              { shape:capsule color:[1.,1.,1.,.5] size:[.1 .05] Q:"t(0 0.02 -.0) d(90 1 0 0)", contact:-3 }


Edit shoulder_pan_joint: { q: 0.0 }
Edit shoulder_lift_joint: { q: -2. }
Edit elbow_joint: { q: 1.0 }
Edit wrist_1_joint: { q: -1 }
Edit wrist_2_joint: { q: -1.571 }
Edit wrist_3_joint: { q: 0.0 }

ee_marker (wrist_3_joint) { shape:marker, size:[.05], Q:"d(-90 1 0 0) t(0 0 .26)"}

#Prefix: "ur_"
Include: <../robotiq/robotiq.g>
Edit robotiq_base (wrist_3_joint) { Q:"d(90 0 0 1) d(90 0 1 0) t(-.0 -.0 .13)" }
