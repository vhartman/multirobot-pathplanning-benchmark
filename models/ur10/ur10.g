ur_base: {multibody: true}

Prefix: "ur_"
Include: <ur10_clean.g>
Prefix: false

Edit ur_base_link(ur_base): {}

ur_coll0(ur_base_link)   { shape:capsule color:[1.,1.,1.,.6] size:[.12 .09] Q:"t(-.0 .0 .1) d(90 0 0 1)", contact:-1  }

ur_coll2(ur_shoulder_lift_joint_origin)   { shape:capsule color:[1.,1.,1.,.5] size:[.17 .09] Q:"t(-.0 -.12 .01) d(90 1 0 0)", contact:-1  }

ur_coll3(ur_shoulder_lift_joint)   { shape:capsule color:[1.,1.,1.,.5] size:[.5 .065] Q:"t(-.0 -.04 .3) d(90 0 0 1)", contact:-1  }

ur_coll4(ur_elbow_joint_origin)   { shape:capsule color:[1.,1.,1.,.5] size:[.16 .065] Q:"t(-.0 .06 .0) d(90 1 0 0)", contact:-1  }

ur_coll5(ur_elbow_joint)   { shape:capsule color:[1.,1.,1.,.5] size:[.5 .06] Q:"t(-.0 -.00 .3) d(90 0 0 1)", contact:-2  }

ur_coll6(ur_wrist_1_joint)   { shape:capsule color:[1.,1.,1.,.5] size:[.1 .05] Q:"t(-.0 .02 -.00) d(90 1 0 0)", contact:-2  }

ur_coll7(ur_wrist_2_joint)   { shape:capsule color:[1.,1.,1.,0.5] size:[.12 .05] Q:"t(-.00 -.0 .04) d(90 0 0 1)", contact:-2  }

ur_coll8(ur_wrist_3_joint)   { shape:capsule color:[1.,1.,1.,.5] size:[.09 .05] Q:"t(-.0 .02 -.0) d(90 1 0 0)", contact:-2  }

#Edit ur_world_joint_origin: { rel: [0, 0, 0] }

Edit ur_shoulder_pan_joint: { q: 0.0 }
Edit ur_shoulder_lift_joint: { q: -2.0 }
Edit ur_elbow_joint: { q: 1.0 }
Edit ur_wrist_1_joint: { q: -1.0 }
Edit ur_wrist_2_joint: { q: 1.0 }
Edit ur_wrist_3_joint: { q: 0.0 }

Edit ur_shoulder_pan_joint: {limits: [-3.28319, 3.28319, 2.16, -1, 330]}
Edit ur_shoulder_lift_joint: {limits: [-3.28319, 3.28319, 2.16, -1, 330]}
Edit ur_elbow_joint: {limits: [-3.28319, 3.28319, 3.15, -1, 150]}
Edit ur_wrist_1_joint: {limits: [-3.28319, 3.28319, 3.2, -1, 54]}
Edit ur_wrist_2_joint: {limits: [-3.28319, 3.28319, 3.2, -1, 54]}
Edit ur_wrist_3_joint: {limits: [-3.28319, 3.28319, 3.2, -1, 54]}
