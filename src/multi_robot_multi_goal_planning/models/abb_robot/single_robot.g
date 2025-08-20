base_link: {}
tmp (base_link) {joint: transX, limits: [-1.5, 1.5]}
base_link_0 (tmp): { mesh: './abb_crb15000_dual_support/meshes/visual/base_link.stl', color: [0.3450980 0.3647059 0.3686275 1], visual: true}
link_1: {}
link_1_0 (link_1): { mesh: './abb_crb15000_dual_support/meshes/visual/link_1.stl', color: [0.3450980 0.3647059 0.3686275 1], visual: true }
link_2: {}
link_2_0 (link_2): { mesh: './abb_crb15000_dual_support/meshes/visual/link_2.stl', color: [0.3450980 0.3647059 0.3686275 1], visual: true }
link_3: {}
link_3_0 (link_3): { mesh: './abb_crb15000_dual_support/meshes/visual/link_3.stl', color: [0.3450980 0.3647059 0.3686275 1], visual: true }
link_4: {}
link_4_0 (link_4): { mesh: './abb_crb15000_dual_support/meshes/visual/link_4.stl', color: [0.7725490 0.7803922 0.7686275 1], visual: true }
link_5: {}
link_5_0 (link_5): { mesh: './abb_crb15000_dual_support/meshes/visual/link_5.stl', color: [0.7725490 0.7803922 0.7686275 1], visual: true }
link_6: {}
link_6_0 (link_6): { mesh: './abb_crb15000_dual_support/meshes/visual/link_6.stl', color: [0.8392157 0.8352941 0.7921569 1], visual: true }
base: {}
flange: {}
tool0: {}

joint_1_origin (tmp): { Q: [0 0 0.265] }
joint_1 (joint_1_origin link_1): { joint: hingeZ, limits: [-3.559265359 3.5159265359], ctrl_limits: [2.18166156499 -1 150],}
joint_2 (link_1 link_2): { joint: hingeY, limits: [-3.559265359 3.5159265359], ctrl_limits: [2.18166156499 -1 150],}
joint_3_origin (link_2): { Q: [0 0 0.444] }
joint_3 (joint_3_origin link_3): { joint: hingeY, limits: [-3.92699081699 5.4835298642], ctrl_limits: [2.44346095279 -1 150],}
joint_4_origin (link_3): { Q: [0 0 0.110] }
joint_4 (joint_4_origin link_4): { joint: hingeX, limits: [-3.559265359 3.5159265359], ctrl_limits: [3.49065850399 -1 150],}
joint_5_origin (link_4): { Q: [0.470 0 0] }
joint_5 (joint_5_origin link_5): { joint: hingeY, limits: [-3.559265359 3.5159265359], ctrl_limits: [3.49065850399 -1 150],}
joint_6_origin (link_5): { Q: [0.101 0 0.080] }
joint_6 (joint_6_origin link_6): { joint: hingeX, limits: [-3.559265359 3.5159265359], ctrl_limits: [3.49065850399 -1 150],}
base_link-base (base_link base): { joint: rigid,}
joint_6-flange (link_6 flange): { joint: rigid,}
flange-tool0_origin (flange): { Q: 'E(0 1.57079632679 0)' }
flange-tool0 (flange-tool0_origin tool0): { joint: rigid,}

coll0(tmp)   { shape:capsule color:[1.,1.,1.,.6] size:[.15 .08] Q:"t(-.0 .0 .1) d(90 0 0 1)", contact:-2  }
coll1(link_1)   { shape:capsule color:[1.,1.,1.,.6] size:[.18 .07] Q:"t(-.0 -.03 .0) d(90 1 0 0)", contact:-2  }
coll2_0(link_2)   { shape:capsule color:[1.,1.,1.,.6] size:[.44 .04] Q:"t(-.03 -.15 .22) d(90 0 0 1)", contact:-2  }
coll2_1(link_2)   { shape:capsule color:[1.,1.,1.,.6] size:[.44 .04] Q:"t(.03 -.15 .22) d(90 0 0 1)", contact:-2  }
coll3(link_3)   { shape:capsule color:[1.,1.,1.,.6] size:[.18 .08] Q:"t(-.0 -.03 .0) d(90 1 0 0)", contact:-2  }
coll4_0(link_4)   { shape:capsule color:[1.,1.,1.,.6] size:[.19 .06] Q:"t(.13 -.0 .0) d(90 0 1 0)", contact:-2  }
coll4_1(link_4)   { shape:capsule color:[1.,1.,1.,.6] size:[.25 .03] Q:"t(.37 .1 -0.02) d(90 0 1 0)", contact:-2  }
coll4_2(link_4)   { shape:capsule color:[1.,1.,1.,.6] size:[.25 .03] Q:"t(.37 .1 .02) d(90 0 1 0)", contact:-2  }
coll5(link_5)   { shape:capsule color:[1.,1.,1.,.6] size:[.1 .06] Q:"t(.0 -.0 .0) d(90 1 0 0)", contact:-2  }
coll6(link_6)   { shape:capsule color:[1.,1.,1.,.6] size:[.13 .06] Q:"t(-0.115 -.0 .0) d(90 0 1 0)", contact:-2  }

ee_center (joint_6){
  Q:[0.03, 0, 0.0]
}

ee_marker (ee_center){
    shape: marker,
    size:[0.05]
}
