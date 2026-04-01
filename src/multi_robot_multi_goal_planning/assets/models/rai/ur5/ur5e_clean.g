base 	{  }
base_link_0 (base) 	{  Q:[0, 0, 0, 0.707107, 0.707107, 0, 0], shape:mesh, color:[0.7, 0.7, 0.7, 1], mesh:'meshes/eSeries_UR5e_baseNoCable.obj', visual }
shoulder_pan_joint_origin (base) 	{  Q:[0, 0, 0.0991, 0.707107, 0.707107, 0, 0] }
shoulder_pan_joint (shoulder_pan_joint_origin) 	{ , joint:hingeY, limits:[-6.2832, 6.2832, 3.14, -1, 150], ctrl_limits:[3.14, -1, 150] }
shoulder_link_0 (shoulder_pan_joint) 	{  Q:[0, -0.0991, 0, -1, 0, 0, 0], shape:mesh, color:[0.7, 0.7, 0.7, 1], mesh:'meshes/eSeries_UR5e_link1.obj', visual }
shoulder_lift_joint_origin (shoulder_pan_joint) 	{  Q:[0, 0.0634, 0.0744, -1, 0, 0, 0] }
shoulder_lift_joint (shoulder_lift_joint_origin) 	{ , joint:hingeZ, limits:[-2, 0.75, 3.14, -1, 150], ctrl_limits:[3.14, -1, 150] }
upper_arm_link_0 (shoulder_lift_joint) 	{  Q:[0, -0.1625, -0.0744, -1, 0, 0, 0], shape:mesh, color:[0.7, 0.7, 0.7, 1], mesh:'meshes/eSeries_UR5e_link2.obj', visual }
elbow_joint_origin (shoulder_lift_joint) 	{  Q:[0, 0.425, 0, -1, 0, 0, 0] }
elbow_joint (elbow_joint_origin) 	{ , joint:hingeZ, limits:[-3.1416, 3.1416, 3.14, -1, 150], ctrl_limits:[3.14, -1, 150] }
forearm_link_0 (elbow_joint) 	{  Q:[0, -0.5875, -0.0744, -1, 0, 0, 0], shape:mesh, color:[0.7, 0.7, 0.7, 1], mesh:'meshes/eSeries_UR5e_link3.obj', visual }
wrist_1_joint_origin (elbow_joint) 	{  Q:[0, 0.3922, -0.0215, -1, 0, 0, 0] }
wrist_1_joint (wrist_1_joint_origin) 	{ , joint:hingeZ, limits:[-3.2832, 3.2832, 6.28, -1, 28], ctrl_limits:[6.28, -1, 28] }
wrist_1_link_0 (wrist_1_joint) 	{  Q:[0, -0.9797, -0.0529, -1, 0, 0, 0], shape:mesh, color:[0.7, 0.7, 0.7, 1], mesh:'meshes/eSeries_UR5e_link4.obj', visual }
wrist_2_joint_origin (wrist_1_joint) 	{  Q:[0, 0.0463, 0.0804, -1, 0, 0, 0] }
wrist_2_joint (wrist_2_joint_origin) 	{ , joint:hingeY, limits:[-3.2832, 3.2832, 6.28, -1, 28], ctrl_limits:[6.28, -1, 28] }
wrist_2_link_0 (wrist_2_joint) 	{  Q:[0, -1.026, -0.1333, -1, 0, 0, 0], shape:mesh, color:[0.7, 0.7, 0.7, 1], mesh:'meshes/eSeries_UR5e_link5.obj', visual }
wrist_3_joint_origin (wrist_2_joint) 	{  Q:[0, 0.0534, 0.0463, -1, 0, 0, 0] }
wrist_3_joint (wrist_3_joint_origin) 	{ , joint:hingeZ, limits:[-6.2832, 6.2832, 6.28, -1, 28], ctrl_limits:[6.28, -1, 28] }
wrist_3_link_0 (wrist_3_joint) 	{  Q:[0, -1.0794, -0.1796, -1, 0, 0, 0], shape:mesh, color:[0.7, 0.7, 0.7, 1], mesh:'meshes/eSeries_UR5e_link6.obj', visual }

