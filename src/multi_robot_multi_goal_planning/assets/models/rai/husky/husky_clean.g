base_link 	{  }
base_link_1 (base_link) 	{ , shape:mesh, mesh:'meshes/base_link_1.ply', visual }
base_link_0 (base_link) 	{ Q:[0, 0, 0.1, 0, 0, 0, 1]  , shape:box, size:[1, 0.5709, 0.22375, 0], color:[0.8, 0.2, 0.2, 0.5], contact:-2 }
base_link>front_left_wheel (base_link) 	{  Q:[0.256, 0.2854, 0.03282, 0.707107, 0, 0, 0.707107] }
base_link>front_right_wheel (base_link) 	{  Q:[0.256, -0.2854, 0.03282, 0.707107, 0, 0, 0.707107] }
base_link>rear_left_wheel (base_link) 	{  Q:[-0.256, 0.2854, 0.03282, 0.707107, 0, 0, 0.707107] }
base_link>rear_right_wheel (base_link) 	{  Q:[-0.256, -0.2854, 0.03282, 0.707107, 0, 0, 0.707107] }
inertial_link (base_link) 	{ , mass:46.034, inertiaTensor:[0.6022, -0.02364, -0.1197, 1.7386, -0.001544, 2.0296] }
top_chassis_link_1 (base_link) 	{ , shape:mesh, mesh:'meshes/top_chassis_link_1.ply', visual }
front_bumper_link_1 (base_link) 	{  Q:[0.486, 0, 0.091, 1, 0, 0, 0], shape:mesh, mesh:'meshes/front_bumper_link_1.ply', visual }
front_bumper_extension_link_1 (base_link) 	{  Q:[0.486, 0, 0.091, 1, 0, 0, 0], shape:mesh, mesh:'meshes/front_bumper_extension_link_1.ply', visual }
rear_bumper_link_1 (base_link) 	{  Q:[-0.48, 0, 0.091, 1.32679e-06, 0, 0, 1], shape:mesh, mesh:'meshes/rear_bumper_link_1.ply', visual }
dual_arm_bulkhead_link_1 (base_link) 	{  Q:[0, 0, 0.224, 1, 0, 0, 0], shape:mesh, mesh:'meshes/dual_arm_bulkhead_link_1.ply', visual }
#dual_arm_bulkhead_link_0 (base_link) 	{  Q:[0, 0, 0.224, 1, 0, 0, 0], shape:mesh, color:[0.8, 0.2, 0.2, 0.5], mesh:'meshes/dual_arm_bulkhead_link_0.ply', contact:-2 }
top_coll_middle (base_link) 	{ Q:[0, 0, 0.4, 0, 0, 0, 1]  , shape:box, size:[0.8, 0.2, 0.05, 0], color:[0.8, 0.2, 0.2, 0.5], contact:-2 }
top_coll_right (base_link) 	{ Q:[0, 0.15, 0.35, 0, 0, 0.383,  0.924]  , shape:box, size:[0.8, 0.2, 0.05, 0], color:[0.8, 0.2, 0.2, 0.5], contact:-2 }
top_coll_left (base_link) 	{ Q:[0, -0.15, 0.35, 0, 0, -0.383,  0.924]  , shape:box, size:[0.8, 0.2, 0.05, 0], color:[0.8, 0.2, 0.2, 0.5], contact:-2 }

base_link>dual_arm_bulkhead_joint (base_link) 	{  Q:[0, 0, 0.224, 1, 0, 0, 0] }
dual_arm_bulkhead_joint (base_link>dual_arm_bulkhead_joint) 	{ , joint:rigid }
dual_arm_bulkhead_link (dual_arm_bulkhead_joint) 	{  }
dual_arm_bulkhead_link>left_arm_bulkhead_joint (dual_arm_bulkhead_link) 	{  Q:[0.1225, 0.14891, 0.13371, 0.653281, -0.270598, -0.270598, -0.653281] }
dual_arm_bulkhead_link>right_arm_bulkhead_joint (dual_arm_bulkhead_link) 	{  Q:[0.1225, -0.14891, 0.13371, 0.653281, 0.270598, 0.270598, -0.653281] }
left_arm_bulkhead_joint (dual_arm_bulkhead_link>left_arm_bulkhead_joint) 	{ , joint:rigid }
right_arm_bulkhead_joint (dual_arm_bulkhead_link>right_arm_bulkhead_joint) 	{ , joint:rigid }
left_arm_bulkhead_link (left_arm_bulkhead_joint) 	{  }
right_arm_bulkhead_link (right_arm_bulkhead_joint) 	{  }

front_left_wheel (base_link>front_left_wheel) 	{ }
front_right_wheel (base_link>front_right_wheel) 	{ }
rear_left_wheel (base_link>rear_left_wheel) 	{  }
rear_right_wheel (base_link>rear_right_wheel) 	{  }
front_left_wheel_link (front_left_wheel) 	{  Q:[-0, -0, -0, -0.707107, 0, 0, -0.707107], mass:2.637, inertiaTensor:[0.02467, 0, 0, 0.04411, 0, 0.02467] }
front_left_wheel_link_1 (front_left_wheel) 	{  Q:[-0, -0, -0, -0.707107, 0, 0, 0.707107], shape:mesh, mesh:'meshes/front_left_wheel_link_1.ply', visual }
front_left_wheel_link_0 (front_left_wheel) 	{  Q:[-0, -0, -0, 0, 0.707107, 0, 0.707107], shape:cylinder, size:[0, 0, 0.1143, 0.1651], color:[0.8, 0.2, 0.2, 0.5], contact:-2 }
front_right_wheel_link (front_right_wheel) 	{  Q:[-0, -0, -0, -0.707107, 0, 0, 0.707107], mass:2.637, inertiaTensor:[0.02467, 0, 0, 0.04411, 0, 0.02467] }
front_right_wheel_link_1 (front_right_wheel) 	{  Q:[-0, -0, -0, -0.707107, 0, 0, 0.707107], shape:mesh, mesh:'meshes/front_right_wheel_link_1.ply', visual }
front_right_wheel_link_0 (front_right_wheel) 	{  Q:[-0, -0, -0, 0, 0.707107, 0, 0.707107], shape:cylinder, size:[0, 0, 0.1143, 0.1651], color:[0.8, 0.2, 0.2, 0.5], contact:-2 }
rear_left_wheel_link (rear_left_wheel) 	{  Q:[-0, -0, -0, -0.707107, 0, 0, 0.707107], mass:2.637, inertiaTensor:[0.02467, 0, 0, 0.04411, 0, 0.02467] }
rear_left_wheel_link_1 (rear_left_wheel) 	{  Q:[-0, -0, -0, -0.707107, 0, 0, 0.707107], shape:mesh, mesh:'meshes/rear_left_wheel_link_1.ply', visual }
rear_left_wheel_link_0 (rear_left_wheel) 	{  Q:[-0, -0, -0, 0, 0.707107, 0, 0.707107], shape:cylinder, size:[0, 0, 0.1143, 0.1651], color:[0.8, 0.2, 0.2, 0.5], contact:-2 }
rear_right_wheel_link (rear_right_wheel) 	{  Q:[-0, -0, -0, -0.707107, 0, 0, 0.707107], mass:2.637, inertiaTensor:[0.02467, 0, 0, 0.04411, 0, 0.02467] }
rear_right_wheel_link_1 (rear_right_wheel) 	{  Q:[-0, -0, -0, -0.707107, 0, 0, 0.707107], shape:mesh, mesh:'meshes/rear_right_wheel_link_1.ply', visual }
rear_right_wheel_link_0 (rear_right_wheel) 	{  Q:[-0, -0, -0, 0, 0.707107, 0, 0.707107], shape:cylinder, size:[0, 0, 0.1143, 0.1651], color:[0.8, 0.2, 0.2, 0.5], contact:-2 }

