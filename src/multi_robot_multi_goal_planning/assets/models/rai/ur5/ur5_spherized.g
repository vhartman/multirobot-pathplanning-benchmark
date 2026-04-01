base: {multibody: true}

Include: <ur5e_clean.g>

# Spherized UR5 collision geometry, matching the VAMP ur5_spherized.urdf decomposition.
# Sphere positions are translated from URDF link frames to RAI joint frames:
#   - upper_arm/forearm: z_URDF -> y_RAI (arm extends along y in RAI shoulder_lift/elbow frames)
#   - base/shoulder: no rotation at joint, coords identical
#   - wrist links: coordinates kept as in URDF link frame (UR5 vs UR5e wrist difference
#     is negligible relative to sphere radii r=0.04)

# base_link: 1 sphere r=0.08 at origin
sph_base(shoulder_pan_joint_origin)   { color:[1.,1.,1.,.5] shape:sphere size:[0.08]  Q:"t(0 -0.04 0)",      contact:-2 }

# shoulder_link: 1 sphere r=0.08 at origin
sph_shoulder(shoulder_pan_joint)      { color:[1.,1.,1.,.5] shape:sphere size:[0.08]  Q:"t(0 0.07 0.)",      contact:-2 }

# upper_arm_link: 5 spheres r=0.08, spaced 0.105 m along arm (z_URDF -> y_RAI)
sph_ua0(shoulder_lift_joint)          { color:[1.,1.,1.,.5] shape:sphere size:[0.08]  Q:"t(0 0    0.06)",      contact:-2 }
sph_ua1(shoulder_lift_joint)          { color:[1.,1.,1.,.5] shape:sphere size:[0.08]  Q:"t(0 .105 0.06)",   contact:-2 }
sph_ua2(shoulder_lift_joint)          { color:[1.,1.,1.,.5] shape:sphere size:[0.08]  Q:"t(0 .21  0.06)",    contact:-2 }
sph_ua3(shoulder_lift_joint)          { color:[1.,1.,1.,.5] shape:sphere size:[0.08]  Q:"t(0 .315 0.06)",   contact:-2 }
sph_ua4(shoulder_lift_joint)          { color:[1.,1.,1.,.5] shape:sphere size:[0.08]  Q:"t(0 .42  0.06)",    contact:-2 }

# forearm_link: 1 large sphere r=0.08 at elbow + 8 smaller spheres r=0.04 (z_URDF -> y_RAI)
sph_fa0(elbow_joint)                  {color:[1.,1.,1.,.5] shape:sphere size:[0.08]  Q:"t(0 0 -0.05)",      contact:-2 }
sph_fa1(elbow_joint)                  {color:[1.,1.,1.,.5] shape:sphere size:[0.04]  Q:"t(0 .1  -0.067)",     contact:-2 }
sph_fa2(elbow_joint)                  {color:[1.,1.,1.,.5] shape:sphere size:[0.04]  Q:"t(0 .14 -0.067)",    contact:-2 }
sph_fa3(elbow_joint)                  {color:[1.,1.,1.,.5] shape:sphere size:[0.04]  Q:"t(0 .18 -0.067)",    contact:-2 }
sph_fa4(elbow_joint)                  {color:[1.,1.,1.,.5] shape:sphere size:[0.04]  Q:"t(0 .22 -0.067)",    contact:-2 }
sph_fa5(elbow_joint)                  {color:[1.,1.,1.,.5] shape:sphere size:[0.04]  Q:"t(0 .26 -0.067)",    contact:-2 }
sph_fa6(elbow_joint)                  {color:[1.,1.,1.,.5] shape:sphere size:[0.04]  Q:"t(0 .3  -0.067)",     contact:-2 }
sph_fa7(elbow_joint)                  {color:[1.,1.,1.,.5] shape:sphere size:[0.04]  Q:"t(0 .34 -0.067)",    contact:-2 }
sph_fa8(elbow_joint)                  {color:[1.,1.,1.,.5] shape:sphere size:[0.04]  Q:"t(0 .38 -0.067)",    contact:-2 }

# wrist_1_link: 3 spheres r=0.04 covering the wrist_1 joint area
sph_w1a(wrist_1_joint)                { color:[1.,1.,1.,.5] shape:sphere size:[0.04]  Q:"t(0 .0 .03)",  contact:-2 }
sph_w1b(wrist_1_joint)                { color:[1.,1.,1.,.5] shape:sphere size:[0.04]  Q:"t(0 .0 -.03)", contact:-2 }
sph_w1c(wrist_1_joint)                { color:[1.,1.,1.,.5] shape:sphere size:[0.04]  Q:"t(0 .0 0)",    contact:-2 }

# wrist_2_link: 3 spheres r=0.04 covering the wrist_2 joint area
sph_w2a(wrist_2_joint)                { color:[1.,1.,1.,.5] shape:sphere size:[0.04]  Q:"t(0 .03  -0.0)",  contact:-2 }
sph_w2b(wrist_2_joint)                { color:[1.,1.,1.,.5] shape:sphere size:[0.04]  Q:"t(0 -.03 -0.0)", contact:-2 }
sph_w2c(wrist_2_joint)                { color:[1.,1.,1.,.5] shape:sphere size:[0.04]  Q:"t(0 0    -0.0)",    contact:-2 }

# wrist_3_link: 1 sphere r=0.04
sph_w31(wrist_3_joint)                 { color:[1.,1.,1.0,.5] shape:sphere size:[0.04]  Q:"t(0 .0 -0.07)",    contact:-2 }
sph_w32(wrist_3_joint)                 { color:[1.,1.,1.0,.5] shape:sphere size:[0.04]  Q:"t(0 .0 -0.02)",    contact:-2 }
sph_w33(wrist_3_joint)                 { color:[1.,1.,1.0,.5] shape:sphere size:[0.04]  Q:"t(0 .0 0.025)",    contact:-2 }

sph_ee22(wrist_3_joint) { color:[1.,1.,1.,.5] shape:sphere size:[0.04]  Q:"t( 0.0001  0. 0.0491)", contact:-2 }
sph_ee23(wrist_3_joint) { color:[1.,1.,1.,.5] shape:sphere size:[0.04]  Q:"t( 0.0002  0. 0.0864)", contact:-2 }
sph_ee24(wrist_3_joint) { color:[1.,1.,1.,.5] shape:sphere size:[0.04]  Q:"t( 0.0002  0. 0.1664)", contact:-2 }
sph_ee25(wrist_3_joint) { color:[1.,1.,1.,.5] shape:sphere size:[0.04]  Q:"t( 0.0002  0. 0.1264)", contact:-2 }
sph_ee26(wrist_3_joint) { color:[1.,1.,1.,.5] shape:sphere size:[0.02]  Q:"t( 0.0329  0. 0.1958)", contact:-2 }
sph_ee27(wrist_3_joint) { color:[1.,1.,1.,.5] shape:sphere size:[0.015] Q:"t( 0.0475  0. 0.2463)", contact:-2 }
sph_ee28(wrist_3_joint) { color:[1.,1.,1.,.5] shape:sphere size:[0.015] Q:"t( 0.0475  0. 0.2213)", contact:-2 }
sph_ee29(wrist_3_joint) { color:[1.,1.,1.,.5] shape:sphere size:[0.02]  Q:"t( 0.0308  0. 0.1692)", contact:-2 }
sph_ee30(wrist_3_joint) { color:[1.,1.,1.,.5] shape:sphere size:[0.015] Q:"t( 0.0625  0. 0.2073)", contact:-2 }
sph_ee31(wrist_3_joint) { color:[1.,1.,1.,.5] shape:sphere size:[0.015] Q:"t( 0.0625  0. 0.1673)", contact:-2 }
sph_ee32(wrist_3_joint) { color:[1.,1.,1.,.5] shape:sphere size:[0.015] Q:"t( 0.0625  0. 0.1873)", contact:-2 }
sph_ee33(wrist_3_joint) { color:[1.,1.,1.,.5] shape:sphere size:[0.02]  Q:"t(-0.0325  0. 0.1958)", contact:-2 }
sph_ee34(wrist_3_joint) { color:[1.,1.,1.,.5] shape:sphere size:[0.015] Q:"t(-0.0466  0. 0.2466)", contact:-2 }
sph_ee35(wrist_3_joint) { color:[1.,1.,1.,.5] shape:sphere size:[0.015] Q:"t(-0.0466  0. 0.2216)", contact:-2 }
sph_ee36(wrist_3_joint) { color:[1.,1.,1.,.5] shape:sphere size:[0.02]  Q:"t(-0.0304  0. 0.1692)", contact:-2 }
sph_ee37(wrist_3_joint) { color:[1.,1.,1.,.5] shape:sphere size:[0.015] Q:"t(-0.0621  0. 0.2076)", contact:-2 }
sph_ee38(wrist_3_joint) { color:[1.,1.,1.,.5] shape:sphere size:[0.015] Q:"t(-0.0621  0. 0.1676)", contact:-2 }
sph_ee39(wrist_3_joint) { color:[1.,1.,1.,.5] shape:sphere size:[0.015] Q:"t(-0.0621  0. 0.1876)", contact:-2 }


ee_marker(wrist_3_joint) { shape:marker, size:[.05], Q:"t(0 0 .24)"}

#Prefix: "ur_"
#Include: <../robotiq/robotiq.g>
#Edit ur_robotiq_base (wrist_3_joint) { Q:"d(90 0 0 1) t(-.0 -.0 .09)" }
