Include='z.5.g'

#Delete iiwa_link_6_0
Delete iiwa_link_7_0
Delete iiwa_link_1_0
Delete iiwa_link_2_0
#Delete shape visual
#Delete shape collision

Edit iiwa_link_0_0 (iiwa_link_0) {
    color=[.8 .2 .2 .2], rel=<T t(-0.015 0 0.19)> shape:capsule size=[0 0 0.3 0.139]  }
Edit iiwa_link_3_0 (iiwa_link_2) {
    color=[.8 .2 .2 .3], rel=<T t(0 0.02 0.0)> shape:capsule size=[0 0 0.29 0.09]   }
Edit iiwa_link_4_0 (iiwa_link_4) {
    color=[.8 .2 .2 .2], rel=<T t(0 0.0 0.0)> shape:capsule size=[0 0 0.12 0.08] colorName=[1.0 0.423529411765 0.0392156862745 1.0]   }
Edit iiwa_link_5_0 (iiwa_link_5) {
    color=[.8 .2 .2 .2], rel=<T t(0 0.02 0)> shape:capsule size=[0 0 0.29 0.09]   }
Edit iiwa_link_6_0 (iiwa_link_6) {
    color=[.8 .2 .2 .2], rel=<T d(90 1 0 0) t(0 -.02 -.02)> shape:capsule size=[0 0 0.08 0.09]   }
