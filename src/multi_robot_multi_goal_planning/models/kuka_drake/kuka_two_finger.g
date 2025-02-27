kuka_two_finger_base: {multibody: true}

Prefix: "kuka_"
Include: <kuka.g>
Prefix: false

Edit kuka_world(kuka_two_finger_base): {}

Edit kuka_iiwa_joint_1: { q: 0.0 }
Edit kuka_iiwa_joint_2: { q: -0.4 }
Edit kuka_iiwa_joint_3: { q: 0.0 }
Edit kuka_iiwa_joint_4: { q: -1.0 }
Edit kuka_iiwa_joint_5: { q: -0.0 }
Edit kuka_iiwa_joint_6: { q: 1.3 }
Edit kuka_iiwa_joint_7: { q: 0.0 }


Prefix: "kuka_"
Include: <../robotiq/robotiq.g>
Edit kuka_robotiq_base (kuka_tool0_joint) { Q:"d(90 0 1 0) t(-.0 -.0 .036)" }
