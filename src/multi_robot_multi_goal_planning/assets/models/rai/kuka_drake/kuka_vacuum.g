kuka_base: {multibody: true}

Prefix: "kuka_"
Include: <kuka.g>
Prefix: false

#Edit ur_world_joint_origin: { rel: [0, 0, 0] }

gripper_fill (kuka_tool0_joint){ shape:cylinder, color:[.1, .1, .1 , 1], Q:"d(90 0 1 0) t(-.0 -.0 .025)",
	size:[.05 .021], 
	contact:-1
}

# pen
ur_vacuum (kuka_tool0_joint){ 
    shape:sphere,
    color:[.9, 0, 0 ,1], 
    Q:"t(.06 0.0 0.)",
    size:[0.005],
    contact:0
}

ur_ee_marker (kuka_tool0_joint){
    shape: marker,
    size:[0.05]
}
