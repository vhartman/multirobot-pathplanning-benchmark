base_link {
}

base_link_1 (base_link) {  
type:mesh
mesh:'../meshes/visual/base_link.stl'
color:[0.3450980 0.3647059 0.3686275 1]
visual
}

base_link_0 (base_link) {  
color:[.8 .2 .2 .5]
type:mesh
mesh:'../meshes/collision/base_link.stl'
contact:-2
}

body link_1 {
}

link_1_1 (link_1) {  
type:mesh
mesh:'../meshes/visual/link_1.stl'
color:[0.3450980 0.3647059 0.3686275 1]
visual
}

link_1_0 (link_1) {  
color:[.8 .2 .2 .5]
type:mesh
mesh:'../meshes/collision/link_1.stl'
contact:-2
}

link_2 {
}

link_2_1 (link_2) {  
type:mesh
mesh:'../meshes/visual/link_2.stl'
color:[0.3450980 0.3647059 0.3686275 1]
visual
}

link_2_0 (link_2) {  
color:[.8 .2 .2 .5]
type:mesh
mesh:'../meshes/collision/link_2.stl'
contact:-2
}

body link_3 {
}

link_3_1 (link_3) {  
type:mesh
mesh:'../meshes/visual/link_3.stl'
color:[0.3450980 0.3647059 0.3686275 1]
visual
}

link_3_0 (link_3) {  
color:[.8 .2 .2 .5]
type:mesh
mesh:'../meshes/collision/link_3.stl'
contact:-2
}

link_4 {
}

link_4_1 (link_4) {  
type:mesh
mesh:'../meshes/visual/link_4.stl'
color:[0.7725490 0.7803922 0.7686275 1]
visual
}

link_4_0 (link_4) {  
color:[.8 .2 .2 .5]
type:mesh
mesh:'../meshes/collision/link_4.stl'
contact:-2
}

link_5 {
}

link_5_1 (link_5) {  
type:mesh
mesh:'../meshes/visual/link_5.stl'
color:[0.7725490 0.7803922 0.7686275 1]
visual
}

link_5_0 (link_5) {  
color:[.8 .2 .2 .5]
type:mesh
mesh:'../meshes/collision/link_5.stl'
contact:-2
}

body link_6 {
}

link_6_1 (link_6) {  
type:mesh
mesh:'../meshes/visual/link_6.stl'
color:[0.8392157 0.8352941 0.7921569 1]
visual
}

link_6_0 (link_6) {  
color:[.8 .2 .2 .5]
type:mesh
mesh:'../meshes/collision/link_6.stl'
contact:-2
}

base {
}

flange {
}

tool0 {
}

joint_1 (base_link link_1) {  
joint:hingeX
axis:[0 0 1]
Q:"t(0 0 0.265) E(0 0 0)"
limits:[-3.14159265359 3.14159265359]
ctrl_limits:[2.18166156499 0 1]
}

joint_2 (link_1 link_2) {  
joint:hingeX
axis:[0 1 0]
Q:"t(0 0 0) E(0 0 0)"
limits:[-3.14159265359 3.14159265359]
ctrl_limits:[2.18166156499 0 1]
}

joint_3 (link_2 link_3) {  
joint:hingeX
axis:[0 1 0]
Q:"t(0 0 0.444) E(0 0 0)"
limits:[-3.92699081699 1.4835298642]
ctrl_limits:[2.44346095279 0 1]
}

joint_4 (link_3 link_4) {  
joint:hingeX
axis:[1 0 0]
Q:"t(0 0 0.110) E(0 0 0)"
limits:[-3.14159265359 3.14159265359]
ctrl_limits:[3.49065850399 0 1]
}

joint_5 (link_4 link_5) {  
joint:hingeX
axis:[0 1 0]
Q:"t(0.470 0 0) E(0 0 0)"
limits:[-3.14159265359 3.14159265359]
ctrl_limits:[3.49065850399 0 1]
}

 joint_6 (link_5 link_6) {  
joint:hingeX
axis:[1 0 0]
Q:"t(0.101 0 0.080) E(0 0 0)"
limits:[-3.14159265359 3.14159265359]
ctrl_limits:[3.49065850399 0 1]
}

base_link-base (base_link base) {  
joint:rigid
Q:"t(0 0 0) E(0 0 0)"
}

joint_6-flange (link_6 flange) {  
joint:rigid
Q:"t(0 0 0) E(0 0 0)"
}

flange-tool0 (flange tool0) {  
joint:rigid
Q:"t(0 0 0) E(0 1.57079632679 0)"
}

