Include='kuka.g'
Include='wsg.g'

#this connects the kuka with the wsg (which is build on 'world')
joint (iiwa_link_7 world){ Q=<T t(0 0 .045) E(0 -1.57079632679 0) t(0.005 0 0) E(1.57079632679 0.3926 1.57079632679) > joint:rigid }

Edit wsg_50_base_joint_gripper_right { mimic="wsg_50_base_joint_gripper_left" }
Edit wsg_50_base_joint_gripper_left { limits=[0 .08] }

Edit >wsg_50_base_joint_gripper_right { Q=<T d(180 1 0 0) d(180 0 0 1)>  }
Edit wsg_50_finger_right_1 { Q=<T t(0 0 -0.023) d(180 0 1 0) d(180 0 0 1)> }

#frame endeff(iiwa_link_7){ shape=marker Q=<T t(0 0 .12) d(180 1 0 0) d(180 0 0 1)> color=[1 1 0] size=[.05 .1 .1 0] }

frame endeff(world){ shape:ssBox Q=<T d(90 0 0 1) t(0 0 .08)> size=[.05 .05 .1 .02] color=[1. 1. 0] }

## zero position

Edit iiwa_joint_1 { q= 0.0 }
Edit iiwa_joint_2 { q= 0.5 }
Edit iiwa_joint_3 { q= 0.5 }
Edit iiwa_joint_4 { q= -0.5 }
Edit iiwa_joint_5 { q= -0.5 }
Edit iiwa_joint_6 { q= 0.7 }
Edit iiwa_joint_7 { q= .0 }
Edit wsg_50_base_joint_gripper_left { q=.08 }
