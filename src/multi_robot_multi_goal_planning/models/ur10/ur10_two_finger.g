Include: <ur10.g>

#Edit ur_world_joint_origin: { rel: [0, 0, 0] }

Edit ur_shoulder_pan_joint: { q: 0.0 }
Edit ur_shoulder_lift_joint: { q: -2.0 }
Edit ur_elbow_joint: { q: 1.0 }
Edit ur_wrist_1_joint: { q: -1.0 }
Edit ur_wrist_2_joint: { q: -1.571 }
Edit ur_wrist_3_joint: { q: 0.0 }

Prefix: "ur_"
Include: <../robotiq/robotiq.g>
Edit ur_robotiq_base (ur_ee_link) { Q:"d(90 0 1 0) t(-.0 -.0 .036)" }
