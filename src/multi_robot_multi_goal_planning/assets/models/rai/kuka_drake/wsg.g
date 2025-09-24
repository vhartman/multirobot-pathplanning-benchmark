frame world 	{  }
frame wsg_50_base_link(world) 	{  mass=1.2 }
frame wsg_50_base_link_1(world) 	{  shape=mesh size=[  1 1 1 0.1  ] color=[ 0.5 0.5 0.5 1 ]  mesh='wsg_meshes/WSG50_110.off'  meshscale=[ 1 1 1 ]  }
#frame wsg_50_base_link_0(world) 	{  shape=mesh size=[  1 1 1 0.1  ] color=[ 0.8 0.2 0.2 0.5 ]  mesh='wsg_meshes/WSG50_110.off'  meshscale=[ 1 1 1 ] contact,  }
frame >wsg_50_base_joint_gripper_right(world) 	{  Q=<T 0 0 0 8.12428e-23 1 -1.32679e-06 6.12323e-17 >  }
frame wsg_50_base_joint_gripper_left(wsg_50_base_link) 	{  joint=transX ctrl_H=1 limits=[  -0.055 -0.0027 1 1 1  ] }
frame wsg_50_base_joint_gripper_right(>wsg_50_base_joint_gripper_right) 	{  joint=transX ctrl_H=1 limits=[  0.0027 0.055 1 1 1  ] }
frame wsg_50_gripper_left(wsg_50_base_joint_gripper_left) 	{  mass=0.1 }
frame wsg_50_finger_left(wsg_50_base_joint_gripper_left) 	{  Q=<T 0 0 0.023 1 0 0 0 >  mass=0.1 }
frame wsg_50_gripper_left_1(wsg_50_base_joint_gripper_left) 	{  shape=mesh size=[  1 1 1 0.1  ] color=[ 0 0 0 1 ]  mesh='wsg_meshes/GUIDE_WSG50_110.off'  meshscale=[ 0.001 0.001 0.001 ]  }
#frame wsg_50_gripper_left_0(wsg_50_base_joint_gripper_left) 	{  shape=mesh size=[  1 1 1 0.1  ] color=[ 0.8 0.2 0.2 0.5 ]  mesh='wsg_meshes/GUIDE_WSG50_110.off'  meshscale=[ 0.001 0.001 0.001 ] contact,  }
frame wsg_50_finger_left_1(wsg_50_base_joint_gripper_left) 	{  Q=<T 0 0 0.023 1 0 0 0 >  shape=mesh size=[  1 1 1 0.1  ] color=[ 0 0 0 1 ]  mesh='wsg_meshes/WSG-FMF.off'  meshscale=[ 0.001 0.001 0.001 ]  }
#frame wsg_50_finger_left_0(wsg_50_base_joint_gripper_left) 	{  Q=<T 0 0 0.023 1 0 0 0 >  shape=mesh size=[  1 1 1 0.1  ] color=[ 0.8 0.2 0.2 0.5 ]  mesh='wsg_meshes/WSG-FMF.off'  meshscale=[ 0.001 0.001 0.001 ] contact,  }
frame wsg_50_gripper_right(wsg_50_base_joint_gripper_right) 	{  Q=<T -0 -0 -0 -6.12323e-17 0 -1 0 >  mass=0.1 }
frame wsg_50_finger_right(wsg_50_base_joint_gripper_right) 	{  Q=<T 2.81669e-18 0 -0.023 -6.12323e-17 0 -1 0 >  mass=0.1 }
frame wsg_50_gripper_right_1(wsg_50_base_joint_gripper_right) 	{  Q=<T -0 -0 -0 -6.12323e-17 0 -1 0 >  shape=mesh size=[  1 1 1 0.1  ] color=[ 0 0 0 1 ]  mesh='wsg_meshes/GUIDE_WSG50_110.off'  meshscale=[ 0.001 0.001 0.001 ]  }
#frame wsg_50_gripper_right_0(wsg_50_base_joint_gripper_right) 	{  Q=<T -0 -0 -0 -6.12323e-17 0 -1 0 >  shape=mesh size=[  1 1 1 0.1  ] color=[ 0.8 0.2 0.2 0.5 ]  mesh='wsg_meshes/GUIDE_WSG50_110.off'  meshscale=[ 0.001 0.001 0.001 ] contact,  }
frame wsg_50_finger_right_1(wsg_50_base_joint_gripper_right) 	{  Q=<T 2.81669e-18 0 -0.023 -6.12323e-17 0 -1 0 >  shape=mesh size=[  1 1 1 0.1  ] color=[ 0 0 0 1 ]  mesh='wsg_meshes/WSG-FMF.off'  meshscale=[ 0.001 0.001 0.001 ]  }
#frame wsg_50_finger_right_0(wsg_50_base_joint_gripper_right) 	{  Q=<T 2.81669e-18 0 -0.023 -6.12323e-17 0 -1 0 >  shape=mesh size=[  1 1 1 0.1  ] color=[ 0.8 0.2 0.2 0.5 ]  mesh='wsg_meshes/WSG-FMF.off'  meshscale=[ 0.001 0.001 0.001 ] contact,  }

