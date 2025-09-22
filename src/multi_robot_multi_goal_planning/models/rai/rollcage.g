base {multibody: true}
floor (base){type:ssBox Q:"t(.0 .0 .05)" size:[1 .6 .02 .002], color:[.3 .3 .3] contact:1, logical:{ }}

bar_l_1(floor) {type:cylinder Q:"t(.5 .3 .4)" size:[.75.01], color:[.3 .3 .3] contact:1, logical:{ }}
bar_l_2(floor) {type:cylinder Q:"t(.5 .15 .4)" size:[.75.01], color:[.3 .3 .3] contact:1, logical:{ }}
bar_l_3(floor) {type:cylinder Q:"t(.5 .0 .4)" size:[.75.01], color:[.3 .3 .3] contact:1, logical:{ }}
bar_l_4(floor) {type:cylinder Q:"t(.5 -.15 .4)" size:[.75.01], color:[.3 .3 .3] contact:1, logical:{ }}
bar_l_5(floor) {type:cylinder Q:"t(.5 -.3 .4)" size:[.75.01], color:[.3 .3 .3] contact:1, logical:{ }}

bar_r_1(floor) {type:cylinder Q:"t(-.5 .3 .4)" size:[.75.01], color:[.3 .3 .3] contact:1, logical:{ }}
bar_r_2(floor) {type:cylinder Q:"t(-.5 .15 .4)" size:[.75.01], color:[.3 .3 .3] contact:1, logical:{ }}
bar_r_3(floor) {type:cylinder Q:"t(-.5 .0 .4)" size:[.75.01], color:[.3 .3 .3] contact:1, logical:{ }}
bar_r_4(floor) {type:cylinder Q:"t(-.5 -.15 .4)" size:[.75.01], color:[.3 .3 .3] contact:1, logical:{ }}
bar_r_5(floor) {type:cylinder Q:"t(-.5 -.3 .4)" size:[.75.01], color:[.3 .3 .3] contact:1, logical:{ }}

bar_b_0(floor) {type:cylinder Q:"t(-.4 -.3 .4)" size:[.75.01], color:[.3 .3 .3] contact:1, logical:{ }}
bar_b_1(floor) {type:cylinder Q:"t(-.2 -.3 .4)" size:[.75.01], color:[.3 .3 .3] contact:1, logical:{ }}
bar_b_2(floor) {type:cylinder Q:"t(0 -.3 .4)" size:[.75.01], color:[.3 .3 .3] contact:1, logical:{ }}
bar_b_3(floor) {type:cylinder Q:"t(.2 -.3 .4)" size:[.75.01], color:[.3 .3 .3] contact:1, logical:{ }}
bar_b_4(floor) {type:cylinder Q:"t(.4 -.3 .4)" size:[.75.01], color:[.3 .3 .3] contact:1, logical:{ }}

bar_b_4(floor) {type:cylinder Q:"t(.0 -.3 .4) d(90 0 1 0)" size:[.85.01], color:[.3 .3 .3] contact:1, logical:{ }}
bar_b_5(floor) {type:cylinder Q:"t(.0 -.3 .6) d(90 0 1 0)" size:[.85.01], color:[.3 .3 .3] contact:1, logical:{ }}
bar_b_6(floor) {type:cylinder Q:"t(.0 -.3 .2) d(90 0 1 0)" size:[.85.01], color:[.3 .3 .3] contact:1, logical:{ }}

bar_l_7(floor) {type:cylinder Q:"t(.5 0 .4) d(90 1 0 0)" size:[.65.01], color:[.3 .3 .3] contact:1, logical:{ }}
bar_l_8(floor) {type:cylinder Q:"t(.5 0 .6) d(90 1 0 0)" size:[.65.01], color:[.3 .3 .3] contact:1, logical:{ }}
bar_l_9(floor) {type:cylinder Q:"t(.5 0 .2) d(90 1 0 0)" size:[.65.01], color:[.3 .3 .3] contact:1, logical:{ }}

bar_r_7(floor) {type:cylinder Q:"t(-.5 0 .4) d(90 1 0 0)" size:[.65.01], color:[.3 .3 .3] contact:1, logical:{ }}
bar_r_8(floor) {type:cylinder Q:"t(-.5 0 .6) d(90 1 0 0)" size:[.65.01], color:[.3 .3 .3] contact:1, logical:{ }}
bar_r_9(floor) {type:cylinder Q:"t(-.5 0 .2) d(90 1 0 0)" size:[.65.01], color:[.3 .3 .3] contact:1, logical:{ }}

bar_r_10(floor) {type:cylinder Q:"t(.0 0.3 .2) d(90 0 1 0)" size:[.75.01], color:[.3 .3 .3] contact:1, logical:{ }}
