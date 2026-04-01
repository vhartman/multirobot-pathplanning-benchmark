base: {multibody: true}

Include: <ur5e_clean.g>

coll1(shoulder_pan_joint_origin)  { shape:capsule color:[1.,1.,1.,.5] size:[.18 .08] Q:"t(0 0 0) d(90 90 1 0)", contact:-2 }
coll1_1(shoulder_lift_joint)      { shape:capsule color:[1.,1.,1.,.5] size:[.16 .065] Q:"t(0 .0 0.01) d(0 0 1 0)", contact:-2 }
coll2(shoulder_lift_joint)        { shape:capsule color:[1.,1.,1.,.5] size:[.35 .065] Q:"t(0 .18 .064) d(90 90 1 0)", contact:-2 }
coll3(elbow_joint_origin)         { shape:capsule color:[1.,1.,1.,.5] size:[.16 .065] Q:"t(0 .0 0.01) d(0 0 1 0)", contact:-2 }
coll4(elbow_joint)                { shape:capsule color:[1.,1.,1.,.5] size:[.37 .06] Q:"t(0 .2 -.07) d(90 90 1 0)", contact:-2 }
coll5(wrist_1_joint)              { shape:capsule color:[1.,1.,1.,.5] size:[.1 .047] Q:"t(0 0 -.02) d(0 90 1 0)", contact:-2 }
coll6(wrist_2_joint)              { shape:capsule color:[1.,1.,1.,.5] size:[.12 .05] Q:"t(0 -.01 0) d(90 90 1 0)", contact:-2 }
coll7(wrist_3_joint)              { shape:capsule color:[1.,1.,1.,.5] size:[.1 .05] Q:"t(0 0 -.02) d(0 0 1 0)", contact:-2 }

ee_marker (wrist_3_joint) { shape:marker, size:[.05], Q:"t(0 0 .24)"}

Prefix: "ur_"
Include: <../robotiq/robotiq.g>
Edit ur_robotiq_base (wrist_3_joint) { Q:"d(90 0 0 1) t(-.0 -.0 .09)" }
