World 	{}    

table (World){
    shape:ssBox, Q:[0 0 -.025], size:[5 5 .05 .002], color:[.3 .3 .3]
    contact:1, logical:{ }
}

Prefix: "a0_"
Include: <single_robot.g>

Edit a0_base_link (World) {Q:"t(0 0 0.006) d(90 0 0 1)", , joint:rigid}

Prefix: "a1_"
Include: <single_robot.g>

Edit a1_base_link (World) {Q:"t(1.20434792	0.00855868803	0.00636037738) d(90 0 0 1)", , joint:rigid}
