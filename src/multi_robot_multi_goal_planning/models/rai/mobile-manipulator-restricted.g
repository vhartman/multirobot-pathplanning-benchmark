## mobile base

world { X:"t (0 0 .2)" }

base (world){ shape:ssBox size:[.35 .35 .4 .05], color:[1.0,0.5,0.5,1.0], joint:transXYPhi limits: [-2 4 -2 2 -3.14 3.14]}
base_coll(base){ shape:ssBox size:[.4 .4 .4 .05], color:[1.,1.,1.,0.1], contact:1 }

#rot (base) {joint: hingeZ limits: [-4, 4]}

arm0 (base){ Q:"t (0 0 .3)" shape:capsule size:[.4 .08]}
arm0_coll(arm0){ shape:capsule, color:[1.,1.,1.,.2], size:[.2 .1], contact:1 }
arm0_col(base){ shape:cylinder, color:[.6, .6, .6,1], size:[.5 .1]}

joint1 (arm0) {
    A:"t (0 0 .2)"
    joint: hingeX q:1
    limits: [0 1.7]
}
#.5

cyl_col (joint1){ Q:"t (0 0 0) d(90 0 90  1)" shape:cylinder color:[1.0,0.5,0.5,1.0] size:[.18 .07] }
cyl_noncol (joint1){ Q:"t (0 0 0) d(90 0 90  1)" shape:cylinder color:[.6,0.6,0.6,1.0] size:[.17 .081] }

arm1 (joint1){
    Q:"t (0 0 .4)" shape:capsule size:[.8 .06] }
arm1_coll(arm1){ shape:capsule, color:[1.,1.,1.,.2], size:[.5 .08], contact:-2 }

joint2(arm1) {
    A:"T t(0 0 .4)"
    joint:hingeX q:1
    limits:[0 1.7]
} 

cyl2_col (joint2){ Q:"t (0 0 0) d(90 0 90  1)" shape:cylinder color:[1.0,0.5,0.5,1.0] size:[.141 .06] }
cyl2_noncol (joint2){ Q:"t (0 0 0) d(90 0 90  1)" shape:cylinder color:[.6,0.6,0.6,1.0] size:[.14 .066] }

arm2(joint2){
    Q:"t (0 0 .2)" shape:capsule, size:[.4 .04] }

arm2_coll(arm2){ shape:capsule, color:[1.,1.,1.,.2], size:[.3 .06], contact:1 }
#.25 .13
#.25 .08

joint2a(arm2) {
    A:"T t(0 0 .25)"
    joint:hingeZ q:0
    limits:[-3.14 3.14]
}

#arm3(joint2c){
#    shape:ssBox size:[.1  .1 .1 .01] }

gripper (joint2a){
    shape:sphere, size:[.03]
    Q:"t(1.0 1. 1.0)"
    contact:1
    color:[1.0,0.5,0.5,1.0]
}

gripper (gripper){
    shape:marker, size:[.3]
    contact:0
    color:[1.0,0.5,0.5,1.0]
}

#Include: '../../../scenarios/gripper.g'

Edit gripper(joint2a) { Q:"t(.0 .00 .01) d(180 1 0 0)" }
#joint joint3(arm2 gripper){
#    joint:hingeZ limits:[-1, 1], A:"t(0 0 .25) d(180 1 0 0)" }


