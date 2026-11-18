# Authoritative Panda sphere geometry, derived from the CoMotion/VAMP Panda model.
# Welding tool geometry is edited in welding_tool.g; all lengths are metres.
panda_link0 {joint:rigid}
panda_joint1_origin (panda_link0) {Q:[0 0 0.33300000000000002 1 0 0 0]}
panda_joint1 (panda_joint1_origin) {joint:hingeZ limits:[-2.8973 2.8973]}
panda_link1 (panda_joint1) {}
panda_joint2_origin (panda_link1) {Q:[0 0 0 0.70710678118654791 -0.70710678118654724 0 0]}
panda_joint2 (panda_joint2_origin) {joint:hingeZ limits:[-1.7628 1.7628]}
panda_link2 (panda_joint2) {}
panda_joint3_origin (panda_link2) {Q:[0 -0.316 0 0.70710678118654757 0.70710678118654746 0 0]}
panda_joint3 (panda_joint3_origin) {joint:hingeZ limits:[-2.8973 2.8973]}
panda_link3 (panda_joint3) {}
panda_joint4_origin (panda_link3) {Q:[0.082500000000000004 0 0 0.70710678118654757 0.70710678118654746 0 0]}
panda_joint4 (panda_joint4_origin) {joint:hingeZ limits:[-3.0718 -0.0698]}
panda_link4 (panda_joint4) {}
panda_joint5_origin (panda_link4) {Q:[-0.082500000000000004 0.38400000000000001 0 0.70710678118654791 -0.70710678118654724 0 0]}
panda_joint5 (panda_joint5_origin) {joint:hingeZ limits:[-2.8973 2.8973]}
panda_link5 (panda_joint5) {}
panda_joint6_origin (panda_link5) {Q:[0 0 0 0.70710678118654757 0.70710678118654746 0 0]}
panda_joint6 (panda_joint6_origin) {joint:hingeZ limits:[-0.0175 3.7525]}
panda_link6 (panda_joint6) {}
panda_joint7_origin (panda_link6) {Q:[0.087999999999999995 0 0 0.70710678118654757 0.70710678118654746 0 0]}
panda_joint7 (panda_joint7_origin) {joint:hingeZ limits:[-2.8973 2.8973]}
panda_link7 (panda_joint7) {}
Include: <welding_tool.g>
Edit welding_tool (panda_link7) {joint:rigid Q:[0 0 0.099999999999999784 4.6299999950378053e-05 0 0 0.99999999892815505]}
welding_tcp (welding_tool) {joint:rigid Q:[0.048489999999999971 -3.2781869124635932e-17 0.15241099999999977 0.93969256961929659 -1.8027873682080184e-21 0.34202028390474654 3.6055747364160369e-21]}
panda_link0_sphere_0 (panda_link0) {shape:sphere size:[0.08] Q:[0 0 0.050000000000000003 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link1_sphere_0 (panda_link1) {shape:sphere size:[0.06] Q:[0 -0.080000000000000002 0 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link1_sphere_1 (panda_link1) {shape:sphere size:[0.06] Q:[0 -0.029999999999999999 0 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link1_sphere_2 (panda_link1) {shape:sphere size:[0.06] Q:[0 0 -0.12 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link1_sphere_3 (panda_link1) {shape:sphere size:[0.06] Q:[0 0 -0.17000000000000001 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link2_sphere_0 (panda_link2) {shape:sphere size:[0.06] Q:[0 0 0.029999999999999999 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link2_sphere_1 (panda_link2) {shape:sphere size:[0.06] Q:[0 0 0.080000000000000002 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link2_sphere_2 (panda_link2) {shape:sphere size:[0.06] Q:[0 -0.12 0 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link2_sphere_3 (panda_link2) {shape:sphere size:[0.06] Q:[0 -0.17000000000000001 0 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link3_sphere_0 (panda_link3) {shape:sphere size:[0.06] Q:[0 0 -0.10000000000000001 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link3_sphere_1 (panda_link3) {shape:sphere size:[0.05] Q:[0 0 -0.059999999999999998 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link3_sphere_2 (panda_link3) {shape:sphere size:[0.055] Q:[0.080000000000000002 0.059999999999999998 0 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link3_sphere_3 (panda_link3) {shape:sphere size:[0.055] Q:[0.080000000000000002 0.02 0 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link4_sphere_0 (panda_link4) {shape:sphere size:[0.06] Q:[-0.080000000000000002 0.095000000000000001 0 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link4_sphere_1 (panda_link4) {shape:sphere size:[0.055] Q:[0 0 0.02 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link4_sphere_2 (panda_link4) {shape:sphere size:[0.055] Q:[0 0 0.059999999999999998 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link4_sphere_3 (panda_link4) {shape:sphere size:[0.055] Q:[-0.080000000000000002 0.059999999999999998 0 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link5_sphere_0 (panda_link5) {shape:sphere size:[0.06] Q:[0 0.055 0 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link5_sphere_1 (panda_link5) {shape:sphere size:[0.06] Q:[0 0.074999999999999997 0 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link5_sphere_2 (panda_link5) {shape:sphere size:[0.06] Q:[0 0 -0.22 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link5_sphere_3 (panda_link5) {shape:sphere size:[0.05] Q:[0 0.050000000000000003 -0.17999999999999999 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link5_sphere_4 (panda_link5) {shape:sphere size:[0.025] Q:[0.01 0.080000000000000002 -0.14000000000000001 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link5_sphere_5 (panda_link5) {shape:sphere size:[0.025] Q:[0.01 0.085000000000000006 -0.11 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link5_sphere_6 (panda_link5) {shape:sphere size:[0.025] Q:[0.01 0.089999999999999997 -0.080000000000000002 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link5_sphere_7 (panda_link5) {shape:sphere size:[0.025] Q:[0.01 0.095000000000000001 -0.050000000000000003 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link5_sphere_8 (panda_link5) {shape:sphere size:[0.025] Q:[-0.01 0.080000000000000002 -0.14000000000000001 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link5_sphere_9 (panda_link5) {shape:sphere size:[0.025] Q:[-0.01 0.085000000000000006 -0.11 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link5_sphere_10 (panda_link5) {shape:sphere size:[0.025] Q:[-0.01 0.089999999999999997 -0.080000000000000002 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link5_sphere_11 (panda_link5) {shape:sphere size:[0.025] Q:[-0.01 0.095000000000000001 -0.050000000000000003 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link6_sphere_0 (panda_link6) {shape:sphere size:[0.05] Q:[0 0 0 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link6_sphere_1 (panda_link6) {shape:sphere size:[0.05] Q:[0.080000000000000002 -0.01 0 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link6_sphere_2 (panda_link6) {shape:sphere size:[0.052] Q:[0.080000000000000002 0.035000000000000003 0 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link7_sphere_0 (panda_link7) {shape:sphere size:[0.05] Q:[0 0 0.070000000000000007 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link7_sphere_1 (panda_link7) {shape:sphere size:[0.025] Q:[0.02 0.040000000000000001 0.080000000000000002 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link7_sphere_2 (panda_link7) {shape:sphere size:[0.025] Q:[0.040000000000000001 0.02 0.080000000000000002 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link7_sphere_3 (panda_link7) {shape:sphere size:[0.02] Q:[0.040000000000000001 0.059999999999999998 0.085000000000000006 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
panda_link7_sphere_4 (panda_link7) {shape:sphere size:[0.02] Q:[0.059999999999999998 0.040000000000000001 0.085000000000000006 1 0 0 0] color:[0.75 0.78 0.82 1] contact:1}
