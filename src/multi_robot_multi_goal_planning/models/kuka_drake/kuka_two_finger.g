kuka_base: {multibody: true}

Prefix: "kuka_"
Include: <kuka.g>
Prefix: false

Edit kuka_world(kuka_base): {}

#Edit ur_world_joint_origin: { rel: [0, 0, 0] }

Prefix: "kuka_"
Include: <../robotiq/robotiq.g>
Edit kuka_robotiq_base (kuka_tool0_joint) { Q:"d(90 0 1 0) t(-.0 -.0 .036)" }
