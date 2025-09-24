robotiq_arg2f_base_link: { mass: 0.22652, inertia: [0.00020005, -4.2442e-10, -2.9069e-10, 0.00017832, -3.4402e-08, 0.00013478] }
robotiq_arg2f_base_link_0(robotiq_arg2f_base_link): { shape: mesh, color: [0.1, 0.1, 0.1, 1], mesh: <meshes/visual/robotiq_arg2f_85_base_link.ply>, meshscale: 0.001, visual: True }
finger_joint_origin(robotiq_arg2f_base_link): { rel: [0, -0.0306011, 0.054904, 6.12323e-17, 0, 0, 1] }
left_inner_knuckle_joint_origin(robotiq_arg2f_base_link): { rel: [0, -0.0127, 0.06142, 6.12323e-17, 0, 0, 1] }
right_outer_knuckle_joint_origin(robotiq_arg2f_base_link): { rel: [0, 0.0306011, 0.054904, 1, 0, 0, 0] }
right_inner_knuckle_joint_origin(robotiq_arg2f_base_link): { rel: [0, 0.0127, 0.06142, 1, 0, 0, 0] }
finger_joint(finger_joint_origin): { }
left_inner_knuckle_joint(left_inner_knuckle_joint_origin): {  }
right_outer_knuckle_joint(right_outer_knuckle_joint_origin): {  }
right_inner_knuckle_joint(right_inner_knuckle_joint_origin): { }
left_outer_knuckle_0(finger_joint): { shape: mesh, color: [0.792157, 0.819608, 0.933333, 1], mesh: <meshes/visual/robotiq_arg2f_85_outer_knuckle.ply>, meshscale: 0.001, visual: True }
left_outer_finger_0(finger_joint): { rel: [0, 0.0315, -0.0041, 1, 0, 0, 0], shape: mesh, color: [0.1, 0.1, 0.1, 1], mesh: <meshes/visual/robotiq_arg2f_85_outer_finger.ply>, meshscale: 0.001, visual: True }
left_inner_finger_joint_origin(finger_joint): { rel: [0, 0.0376, 0.043, 1, 0, 0, 0] }
left_inner_knuckle_0(left_inner_knuckle_joint): { shape: mesh, color: [0.1, 0.1, 0.1, 1], mesh: <meshes/visual/robotiq_arg2f_85_inner_knuckle.ply>, meshscale: 0.001, visual: True }
right_outer_knuckle_0(right_outer_knuckle_joint): { shape: mesh, color: [0.792157, 0.819608, 0.933333, 1], mesh: <meshes/visual/robotiq_arg2f_85_outer_knuckle.ply>, meshscale: 0.001, visual: True }
right_outer_finger_0(right_outer_knuckle_joint): { rel: [0, 0.0315, -0.0041, 1, 0, 0, 0], shape: mesh, color: [0.1, 0.1, 0.1, 1], mesh: <meshes/visual/robotiq_arg2f_85_outer_finger.ply>, meshscale: 0.001, visual: True }
right_inner_finger_joint_origin(right_outer_knuckle_joint): { rel: [0, 0.0376, 0.043, 1, 0, 0, 0] }
right_inner_knuckle_0(right_inner_knuckle_joint): { shape: mesh, color: [0.1, 0.1, 0.1, 1], mesh: <meshes/visual/robotiq_arg2f_85_inner_knuckle.ply>, meshscale: 0.001, visual: True }
left_inner_finger_joint(left_inner_finger_joint_origin): { }
right_inner_finger_joint(right_inner_finger_joint_origin): {}
left_inner_finger_0(left_inner_finger_joint): { shape: mesh, color: [0.1, 0.1, 0.1, 1], mesh: <meshes/visual/robotiq_arg2f_85_inner_finger.ply>, meshscale: 0.001, visual: True }
left_inner_finger_pad_0(left_inner_finger_joint): { rel: [0, -0.0220203, 0.03242, 1, 0, 0, 0], shape: box, size: [0.022, 0.00635, 0.0375, 0], color: [0.9, 0.9, 0.9, 1], visual: True }
right_inner_finger_0(right_inner_finger_joint): { shape: mesh, color: [0.1, 0.1, 0.1, 1], mesh: <meshes/visual/robotiq_arg2f_85_inner_finger.ply>, meshscale: 0.001, visual: True }
right_inner_finger_pad_0(right_inner_finger_joint): { rel: [0, -0.0220203, 0.03242, 1, 0, 0, 0], shape: box, size: [0.022, 0.00635, 0.0375, 0], color: [0.9, 0.9, 0.9, 1], visual: True }