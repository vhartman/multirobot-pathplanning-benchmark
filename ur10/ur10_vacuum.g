Include: <ur10.g>

#Edit ur_world_joint_origin: { rel: [0, 0, 0] }

Edit ur_shoulder_pan_joint: { q: 0.0 }
Edit ur_shoulder_lift_joint: { q: -2.0 }
Edit ur_elbow_joint: { q: 1.0 }
Edit ur_wrist_1_joint: { q: -1.0 }
Edit ur_wrist_2_joint: { q: 4.8415 }
Edit ur_wrist_3_joint: { q: 1.0 }

gripper_fill (ur_ee_link){ shape:cylinder, color:[.1, .1, .1 , 1], Q:"d(90 0 1 0) t(-.0 -.0 .025)",
	size:[.05 .021], 
	contact:-2
}

# pen
ur_vacuum (ur_ee_link){ 
    shape:sphere,
    color:[.9, 0, 0 ,1], 
    Q:"t(.06 0.0 0.)",
    size:[0.005],
    contact:0
}

ur_ee_marker (ur_ee_link){
    shape: marker,
    size:[0.05]
}
