# OpenArm v2 right arm + pinch gripper (fingers deactivated open).
right_base: {}
openarm_right_base_joint (right_base openarm_right_base_link): { joint: rigid, }
openarm_right_base_link: { mass: 1.1432284943239561, inertia: [0.00113607806226198 3.81487993180929e-07 1.09161247709715e-08 0.00159935208714031 2.7521421737084e-08 0.000750541980863571],}
openarm_right_base_link_0 (openarm_right_base_link): { rel: "d(-90 1 0 0)", mesh: <meshes/arm/visual/base_link.dae>, color: [0.752941176470588 0.752941176470588 0.752941176470588 1], visual: true }
openarm_right_link1: { mass: 1.1416684646202298, inertia: [0.0010943212038551 9.41628673674543e-06 4.53422182483102e-08 0.000759199728161993 6.66755698859637e-08 0.000670659201019629],}
openarm_right_link1_0 (openarm_right_link1): { rel: "d(-90 1 0 0)", mesh: <meshes/arm/visual/link1.dae>, color: [0.752941176470588 0.752941176470588 0.752941176470588 1], visual: true }
openarm_right_link2: { mass: 0.2775092746011571, inertia: [0.000153588833784759 -2.94422592188677e-10 5.21056076779068e-06 0.000113062180225725 3.75816911469744e-10 8.22999351863579e-05],}
openarm_right_link2_0 (openarm_right_link2): { rel: "d(-90 1 0 0)", mesh: <meshes/arm/visual/link2.dae>, color: [0.752941176470588 0.752941176470588 0.752941176470588 1], visual: true }
openarm_right_link3: { mass: 1.073863338202347, inertia: [0.00128939636591746 -2.29951046666492e-07 -6.76161926984358e-05 0.00114018992179734 -1.06260802872572e-05 0.000559765478676303],}
openarm_right_link3_0 (openarm_right_link3): { rel: "d(-90 1 0 0)", mesh: <meshes/arm/visual/link3.dae>, color: [0.752941176470588 0.752941176470588 0.752941176470588 1], visual: true }
openarm_right_link4: { mass: 0.6348534566833373, inertia: [0.0004690036047182 5.63626910265645e-06 -1.77877991597693e-06 0.000433926270522842 3.41757797773822e-06 0.000245211429759716],}
openarm_right_link4_0 (openarm_right_link4): { rel: "d(-90 1 0 0)", mesh: <meshes/arm/visual/link4.dae>, color: [0.752941176470588 0.752941176470588 0.752941176470588 1], visual: true }
openarm_right_link5: { mass: 0.6156588026168502, inertia: [0.000279653368752989 3.31745289972445e-06 7.68097311403736e-07 0.000334599623120095 -1.49934409279756e-07 0.000176662464388138],}
openarm_right_link5_0 (openarm_right_link5): { rel: "d(-90 1 0 0)", mesh: <meshes/arm/visual/link5.dae>, color: [0.752941176470588 0.752941176470588 0.752941176470588 1], visual: true }
openarm_right_link6: { mass: 0.475202773187987, inertia: [0.000179275941446964 6.58439346712228e-08 -3.37101167120887e-08 0.000136247512758587 3.32253600295544e-08 0.000135340695937386],}
openarm_right_link6_0 (openarm_right_link6): { rel: "d(-90 1 0 0)", mesh: <meshes/arm/visual/link6.dae>, color: [0.752941176470588 0.752941176470588 0.752941176470588 1], visual: true }
openarm_right_ee_base_link: { mass: 0.52832, inertia: [0.00021911 4.1683e-06 1.4043e-06 0.00015627 -1.5124e-06 0.0001659],}
openarm_right_ee_base_link_0 (openarm_right_ee_base_link): { rel: "d(-90 1 0 0) t(0.0205 0.0 0.0) E(0.0 -0.0 0.0)", mesh: <meshes/ee/visual/ee_base_link.dae>, color: [0.752941176470588 0.752941176470588 0.752941176470588 1], visual: true }
openarm_right_ee_link1: { mass: 0.125979047366671, inertia: [3.61105363396623e-05 1.35880208293387e-06 7.13916341634141e-06 3.62137424319396e-05 -3.60343788693423e-06 1.74958961632907e-05],}
openarm_right_ee_link1_0 (openarm_right_ee_link1): { rel: "d(-90 1 0 0)", mesh: <meshes/ee/visual/finger_inner.dae>, color: [0.752941176470588 0.752941176470588 0.752941176470588 1], visual: true }
openarm_right_ee_link2: { mass: 0.11372, inertia: [3.4565e-05 -1.3392e-06 7.1576e-06 3.5306e-05 3.7172e-06 1.6571e-05],}
openarm_right_ee_link2_0 (openarm_right_ee_link2): { rel: "d(-90 1 0 0)", mesh: <meshes/ee/visual/finger_outer.dae>, color: [0.752941176470588 0.752941176470588 0.752941176470588 1], visual: true }

openarm_right_joint1_origin (openarm_right_base_link): { Q: [0 -0.0625 0] }
openarm_right_joint1 (openarm_right_joint1_origin openarm_right_link1): { joint: hingeY, limits: [-3.4907 1.3963], ctrl_limits: [16.755 -1 40],}
openarm_right_joint2_origin (openarm_right_link1): { Q: [0 -0.0600000000000001 0] }
openarm_right_joint2 (openarm_right_joint2_origin openarm_right_link2): { joint: hingeX, limits: [-3.3161 0.17453], ctrl_limits: [16.755 -1 40],}
openarm_right_joint3_origin (openarm_right_link2): { Q: [0 0 -0.0662500000000008] }
openarm_right_joint3 (openarm_right_joint3_origin openarm_right_link3): { joint: hingeZ, limits: [-1.5708 1.5708], ctrl_limits: [5.4454 -1 27],}
openarm_right_joint4_origin (openarm_right_link3): { Q: [0 0 -0.15375] }
openarm_right_joint4 (openarm_right_joint4_origin openarm_right_link4): { joint: hingeY, limits: [-2.4435 -0.0], ctrl_limits: [5.4454 -1 27],}
openarm_right_joint5_origin (openarm_right_link4): { Q: [0 0 -0.0955000000000005] }
openarm_right_joint5 (openarm_right_joint5_origin openarm_right_link5): { joint: hingeZ, limits: [-1.5708 1.5708], ctrl_limits: [20.944 -1 7],}
openarm_right_joint6_origin (openarm_right_link5): { Q: [0 0 -0.120499999999998] }
# The calibrated limits (j6 +/-0.7854, j7 +/-1.5708) lose all SSIK branches for some
# demonstrated Car Mini tool poses. Fresh search + optimized MP succeeds at +/-1.8 for both;
# j7 keeps another 0.05rad because that stricter validation path touched its lower bound.
openarm_right_joint6 (openarm_right_joint6_origin openarm_right_link6): { joint: hingeY, limits: [-1.8 1.8], ctrl_limits: [20.944 -1 7],}
openarm_right_joint7 (openarm_right_link6 openarm_right_ee_base_link): { joint: hingeX, limits: [-1.85 1.85], ctrl_limits: [20.944 -1 7],}
openarm_right_finger_joint1_origin (openarm_right_ee_base_link): { Q: "t(-0.0014300000000005003 0.0180000000000009 -0.0680000000000174) E(0.0 -0.0 0.0)" }
openarm_right_finger_joint1 (openarm_right_finger_joint1_origin openarm_right_ee_link1): { joint: rigid, q: 0.5 }
openarm_right_finger_joint2_origin (openarm_right_ee_base_link): { Q: "t(-0.0014300000000005003 -0.0180000000000012 -0.0680000000000215) E(0.0 -0.0 0.0)" }
openarm_right_finger_joint2 (openarm_right_finger_joint2_origin openarm_right_ee_link2): { joint: rigid, q: -0.5 }

right_joint1_coll (openarm_right_joint1): { shape: capsule, color: [1.,1.,1.,.5], size: [0.02, 0.07], Q: "t(0 0 -0.0) d(90 1 0 0)", contact: -2 }
right_joint2_coll (openarm_right_joint2): { shape: capsule, color: [1.,1.,1.,.5], size: [0.05, 0.06], Q: "t(0 0 0.0)", contact: -2 }
right_joint3_coll (openarm_right_joint3): { shape: capsule, color: [1.,1.,1.,.5], size: [0.16, 0.055], Q: "t(0 0 -0.06)", contact: -2 }
##right_joint4_coll (openarm_right_joint4): { shape: capsule, color: [1.,1.,1.,1], size: [0.14, 0.03], Q: "t(0 0 0.0)", contact: -2 }
# Keep the double-# joint4/joint6 guesses disabled.  This single-# candidate is active, but
# fitted to link5's 129x76mm collision mesh rather than the old 260x100mm over-approximation.
right_joint5_coll (openarm_right_joint5): { shape: capsule, color: [1.,1.,1.,.5], size: [0.06, 0.038], Q: "t(0 0 -0.055)", contact: -2 }
##right_joint6_coll (openarm_right_joint6): { shape: capsule, color: [1.,1.,1.,.5], size: [0.08, 0.04], Q: "t(0 0 0.0)", contact: -2 }

right_ee_coll (openarm_right_ee_base_link): { shape: capsule, color: [1.,1.,1.,.5], size: [.02, .04], Q: "t(.005 0. -.03) d(0 1 0 0)", contact: -2 }
right_ee_coll1 (openarm_right_ee_base_link): { shape: capsule, color: [1.,1.,1.,.5], size: [.06, .03], Q: "t(-.005 0 -.075) d(90 1 0 0)", contact: -2 }
right_ee_coll2 (openarm_right_ee_base_link): { shape: capsule, color: [1.,1.,1.,.5], size: [.06, .03], Q: "t(.02 0 -.075) d(90 1 0 0)", contact: -2 }

right_finger1_coll (openarm_right_finger_joint1_origin): { shape: capsule, color: [1.,1.,1.,.5], size: [.06, .015], Q: "t(-0.009 0.04 -.0525)", contact: -2 }
right_finger1_coll2 (openarm_right_finger_joint1_origin): { shape: capsule, color: [1.,1.,1.,.5], size: [.06, .015], Q: "t(0.01 0.04 -.0525)", contact: -2 }
right_finger2_coll1 (openarm_right_finger_joint2_origin): { shape: capsule, color: [1.,1.,1.,.5], size: [.06, .015], Q: "t(-0.009 -0.04 -.0525)", contact: -2 }
right_finger2_coll2 (openarm_right_finger_joint2_origin): { shape: capsule, color: [1.,1.,1.,.5], size: [.06, .015], Q: "t(0.01 -0.04 -.0525)", contact: -2 }

gripper_center (openarm_right_ee_base_link): { shape: marker, size: [.03], color: [.9,.9,.9], Q: "t(-.0014 0 -.14) d(180 1 0 0)" }

Edit openarm_right_joint1: { q: -0.0 }
Edit openarm_right_joint2: { q: -0.2 }
Edit openarm_right_joint3: { q:  0.0 }
Edit openarm_right_joint4: { q:  -1.2 }
Edit openarm_right_joint5: { q:  0.0 }
Edit openarm_right_joint6: { q:  -0.5 }
