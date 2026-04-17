ur_base: {multibody: true}

Prefix: "ur_"
Include: <ur5e_clean.g>
Prefix: false

Edit ur_base_link(ur_base): {}

coll1(ur_shoulder_pan_joint_origin)  { shape:capsule color:[1.,1.,1.,.5] size:[.18 .08] Q:"t(0 0 0) d(90 90 1 0)", contact:-2 }
coll1_1(ur_shoulder_lift_joint)      { shape:capsule color:[1.,1.,1.,.5] size:[.16 .065] Q:"t(0 .0 0.01) d(0 0 1 0)", contact:-2 }
coll2(ur_shoulder_lift_joint)        { shape:capsule color:[1.,1.,1.,.5] size:[.35 .065] Q:"t(0 .18 .064) d(90 90 1 0)", contact:-2 }
coll3(ur_elbow_joint_origin)         { shape:capsule color:[1.,1.,1.,.5] size:[.16 .065] Q:"t(0 .0 0.01) d(0 0 1 0)", contact:-2 }
coll4(ur_elbow_joint)                { shape:capsule color:[1.,1.,1.,.5] size:[.37 .06] Q:"t(0 .2 -.07) d(90 90 1 0)", contact:-2 }
coll5(ur_wrist_1_joint)              { shape:capsule color:[1.,1.,1.,.5] size:[.1 .047] Q:"t(0 0 -.02) d(0 90 1 0)", contact:-2 }
coll6(ur_wrist_2_joint)              { shape:capsule color:[1.,1.,1.,.5] size:[.12 .05] Q:"t(0 -.01 0) d(90 90 1 0)", contact:-2 }
coll7(ur_wrist_3_joint)              { shape:capsule color:[1.,1.,1.,.5] size:[.1 .05] Q:"t(0 0 -.02) d(0 0 1 0)", contact:-2 }

ur_ee_marker (ur_wrist_3_joint) { shape:marker, size:[.05], Q:"t(0 0 .24)"}

Prefix: "ur_"
Include: <../robotiq/robotiq.g>
Edit ur_robotiq_base (ur_wrist_3_joint) { Q:"d(90 0 0 1) t(-.0 -.0 .09)" }

Edit ur_shoulder_pan_joint: { q: 0.0 }
Edit ur_shoulder_lift_joint: { q: -0.5 }
Edit ur_elbow_joint: { q: 1.0 }
Edit ur_wrist_1_joint: { q: 0.5 }
Edit ur_wrist_2_joint: { q: -1.571 }
Edit ur_wrist_3_joint: { q: 0.0 }