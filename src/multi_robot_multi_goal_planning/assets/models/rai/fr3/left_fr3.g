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

#panda_coll0(left_fr3v2_link0): { shape: capsule, color: [1.,1.,1.,.1], size: [.1, .11], Q: "t(-.04 .0 .03) d(90 0 1 0)", contact: -2 }
#panda_coll0b(panda_link0): { shape: capsule, color: [1.,1.,1.,.1], size: [.2, .06], Q: "t(-.2 -.12 .0) d(90 0 1 0)", contact: -2 }

panda_coll1(left_fr3v2_joint1): { shape: capsule, color: [1.,1.,1.,.5], size: [.14, .08], Q: "t(0 0 -.15)", contact: -2 }
panda_coll3(left_fr3v2_joint3): { shape: capsule, color: [1.,1.,1.,.5], size: [.15, .08], Q: "t(0 0 -.15)", contact: -2 }
panda_coll5(left_fr3v2_joint5): { shape: capsule, color: [1.,1.,1.,.5], size: [.22, .09], Q: "t(0 .02 -.2)", contact: -2 }

panda_coll2(left_fr3v2_joint2): { shape: capsule, color: [1.,1.,1.,.5], size: [.12, .08], Q: "t(0 0 .0)", contact: -2 }
panda_coll4(left_fr3v2_joint4): { shape: capsule, color: [1.,1.,1.,.5], size: [.12, .08], Q: "t(0 0 .0)", contact: -2 }
panda_coll6(left_fr3v2_joint6): { shape: capsule, color: [1.,1.,1.,.5], size: [.1, .07], Q: "t(0 .0 -.04)", contact: -2 }
panda_coll7(left_fr3v2_joint7): { shape: capsule, color: [1.,1.,1.,.5], size: [.1, .07], Q: "t(0 .0 .01)", contact: -2 }

Edit left_fr3v2_joint2: { q: -0.5 }
Edit left_fr3v2_joint4: { q: -1 }
