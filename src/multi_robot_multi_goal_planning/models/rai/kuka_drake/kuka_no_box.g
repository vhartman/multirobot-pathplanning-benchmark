world{ X:<t(0 0 .1)> }
        
base (world) { }
frame iiwa_base_joint(base) 	{  joint=rigid ctrl_H=1 }
frame iiwa_link_0(iiwa_base_joint) 	{  mass=5  inertiaTensor:[ 0.05 0 0 0.06 0 0.03 ] }
frame iiwa_link_0_1(iiwa_base_joint) 	{  shape=mesh size=[  1 1 1 0.1  ]  mesh:'kuka_meshes/visual/link_0_x.tri'  colorName:[ 0.4 0.4 0.4 1 ] }
frame iiwa_link_0_0_coll(iiwa_base_joint) 	{  shape=capsule size=[  0 0 0.3 0.139  ]  color:[1.,1.,1.,.2] contact:-1,  Q=<T -0.015 0 0.19 1 0 0 0 >   colorName:[ 0.4 0.4 0.4 1 ] }
frame >iiwa_joint_1(iiwa_base_joint) 	{  Q=<T 0 0 0.1575 0.707107 0 -0.707107 0 >  }
frame iiwa_joint_1(>iiwa_joint_1) 	{  joint=hingeX ctrl_H=1 limits=[  -5.93215 5.93215 10 300 1 10 300 1  ]  ctrl_limits:[ 10 300 1 ] }
frame iiwa_link_1(iiwa_joint_1) 	{  mass=5.76 Q=<T 0 0 0 -0.707107 0 -0.707107 0 >   inertiaTensor:[ 0.033 0 0 0.0333 0 0.0123 ] }
frame iiwa_link_1_1(iiwa_joint_1) 	{  shape=mesh size=[  1 1 1 0.1  ] mesh:'kuka_meshes/visual/link_1_x.tri' Q=<T 0 0 0 -0.707107 0 -0.707107 0 >   colorName:[ 0.4 0.4 0.4 1 ] }
frame >iiwa_joint_2(iiwa_joint_1) 	{  Q=<T 0.2025 0 4.4964e-17 7.30527e-14 -0.707107 -0.707107 -7.33302e-14 >  }
frame iiwa_joint_2(>iiwa_joint_2) 	{  joint=hingeX ctrl_H=1 limits=[  -1.55949 1.55949 10 300 1 10 300 1  ]  ctrl_limits:[ 10 300 1 ] }
#frame iiwa_joint_2(>iiwa_joint_2) 	{  joint=hingeX ctrl_H=1 limits=[  -2.05949 2.05949 10 300 1 10 300 1  ]  ctrl_limits:[ 10 300 1 ] }
frame iiwa_link_2(iiwa_joint_2) 	{  mass=6.35 Q=<T 0 0 0 -0.707107 0 -0.707107 0 >   inertiaTensor:[ 0.0305 0 0 0.0304 0 0.011 ] }
frame iiwa_link_2_1(iiwa_joint_2) 	{  shape=mesh size=[  1 1 1 0.1  ]  mesh:'kuka_meshes/visual/link_2_x.tri' Q=<T 0 0 0 -0.707107 0 -0.707107 0 >   colorName:[ 1 0.423529 0.0392157 1 ] }
frame >iiwa_joint_3(iiwa_joint_2) 	{  Q=<T 0 0.2045 0 7.30527e-14 -0.707107 -0.707107 -7.33302e-14 >  }
frame iiwa_joint_3(>iiwa_joint_3) 	{  joint=hingeX ctrl_H=1 limits=[  -2.05949 2.05949 10 300 1 10 300 1  ]  ctrl_limits:[ 10 300 1 ] }
frame iiwa_link_3(iiwa_joint_3) 	{  mass=3.5 Q=<T 0 0 0 -0.707107 0 -0.707107 0 >   inertiaTensor:[ 0.025 0 0 0.0238 0 0.0076 ] }
frame iiwa_link_3_1(iiwa_joint_3) 	{  shape=mesh size=[  1 1 1 0.1  ] mesh:'kuka_meshes/visual/link_3_x.tri' Q=<T 0 0 0 -0.707107 0 -0.707107 0 >   colorName:[ 0.4 0.4 0.4 1 ] } #5th
frame iiwa_link_3_0_coll(iiwa_joint_3) 	{  shape=capsule size=[  0 0 0.29 0.09  ]  color:[1.,1.,1.,.2] contact:-3,  Q=<T 0 0.02 0 -0.707107 0 -0.707107 0 >   colorName:[ 0.4 0.4 0.4 1 ] }
frame >iiwa_joint_4(iiwa_joint_3) 	{  Q=<T 0.2155 0 4.78506e-17 -0.707107 -1.11022e-16 -5.55112e-17 0.707107 >  }
frame iiwa_joint_4(>iiwa_joint_4) 	{  joint=hingeX ctrl_H=1 limits=[  -2.05949 2.05949 10 300 1 10 300 1  ]  ctrl_limits:[ 10 300 1 ] }
frame iiwa_link_4(iiwa_joint_4) 	{  mass=3.5 Q=<T 0 0 0 -0.707107 0 -0.707107 0 >   inertiaTensor:[ 0.017 0 0 0.0164 0 0.006 ] }
frame iiwa_link_4_1(iiwa_joint_4) 	{  shape=mesh size=[  1 1 1 0.1  ]  mesh:'kuka_meshes/visual/link_4_x.tri' Q=<T 0 0 0 -0.707107 0 -0.707107 0 >   colorName:[ 1 0.423529 0.0392157 1 ] } #4th
frame iiwa_link_4_0_coll(iiwa_joint_4) 	{  shape=capsule size=[  0 0 0.12 0.08  ]  color:[1.,1.,1.,.2] contact:-1,  Q=<T 0 0 0 -0.707107 0 -0.707107 0 >   colorName:[ 1 0.423529 0.0392157 1 ] }
frame >iiwa_joint_5(iiwa_joint_4) 	{  Q=<T 0 0.1845 0 7.32192e-14 -0.707107 -0.707107 7.29417e-14 >  }
frame iiwa_joint_5(>iiwa_joint_5) 	{  joint=hingeX ctrl_H=1 limits=[  -4.05949 4.05949 10 300 1 10 300 1  ]  ctrl_limits:[ 10 300 1 ] }
frame iiwa_link_5(iiwa_joint_5) 	{  mass=3.5 Q=<T 0 0 0 -0.707107 0 -0.707107 0 >   inertiaTensor:[ 0.01 0 0 0.0087 0 0.00449 ] }
frame iiwa_link_5_1(iiwa_joint_5) 	{  shape=mesh size=[  1 1 1 0.1  ]  mesh:'kuka_meshes/visual/link_5_x.tri' Q=<T 0 0 0 -0.707107 0 -0.707107 0 >   colorName:[ 0.4 0.4 0.4 1 ] } #3rd
frame iiwa_link_5_0_coll(iiwa_joint_5) 	{  shape=capsule size=[  0 0 0.29 0.09  ]  color:[1.,1.,1.,.2] contact:-2,  Q=<T 0 0.02 0 -0.707107 0 -0.707107 0 >   colorName:[ 0.4 0.4 0.4 1 ] }
frame >iiwa_joint_6(iiwa_joint_5) 	{  Q=<T 0.2155 0 4.78506e-17 -0.707107 -1.11022e-16 -5.55112e-17 0.707107 >  }
frame iiwa_joint_6(>iiwa_joint_6) 	{  joint=hingeX ctrl_H=1 limits=[  -2.05949 2.05949 10 300 1 10 300 1  ]  ctrl_limits:[ 10 300 1 ] }
frame iiwa_link_6(iiwa_joint_6) 	{  mass=1.8 Q=<T 0 0 0 -0.707107 0 -0.707107 0 >   inertiaTensor:[ 0.0049 0 0 0.0047 0 0.0036 ] }
frame iiwa_link_6_1(iiwa_joint_6) 	{  shape=mesh size=[  1 1 1 0.1  ]  mesh:'kuka_meshes/visual/link_6_x.tri' Q=<T 0 0 0 -0.707107 0 -0.707107 0 >   colorName:[ 1 0.423529 0.0392157 1 ] } # 2nd to last
frame iiwa_link_6_0_coll(iiwa_joint_6) 	{  shape=capsule size=[  0 0 0.08 0.09  ]  color:[1.,1.,1.,.2]  mesh:'kuka_meshes/collision/link_6_x.tri' contact:-1,  Q=<T -0.02 0.02 -4.44089e-18 -0.5 -0.5 -0.5 0.5 >   colorName:[ 1 0.423529 0.0392157 1 ] }
frame >iiwa_joint_7(iiwa_joint_6) 	{  Q=<T 0 0.081 0 7.32192e-14 -0.707107 -0.707107 7.29417e-14 >  }
frame iiwa_joint_7(>iiwa_joint_7) 	{  joint=hingeX ctrl_H=1 limits=[  -3.01942 3.01942 10 300 1 10 300 1  ]  ctrl_limits:[ 10 300 1 ] }
frame iiwa_link_7(iiwa_joint_7) 	{  mass=1.2 Q=<T 0 0 0 -0.707107 0 -0.707107 0 >   inertiaTensor:[ 0.001 0 0 0.001 0 0.001 ] }
frame iiwa_link_7_1(iiwa_joint_7) 	{  shape=mesh size=[  1 1 1 0.1  ] mesh:'kuka_meshes/visual/link_7_x.tri' Q=<T 0 0 0 -0.707107 0 -0.707107 0 >   colorName:[ 0.4 0.4 0.4 1 ] } # just before endeffector
frame >iiwa_joint_ee(iiwa_joint_7) 	{  Q=<T 0.045 0 9.99201e-18 1 0 0 0 >  }
frame >tool0_joint(iiwa_joint_7) 	{  Q=<T 0.045 0 9.99201e-18 1 0 0 0 >  }
frame iiwa_joint_ee(>iiwa_joint_ee) 	{  joint=rigid ctrl_H=1 }
frame tool0_joint(>tool0_joint) 	{  joint=rigid ctrl_H=1 }

