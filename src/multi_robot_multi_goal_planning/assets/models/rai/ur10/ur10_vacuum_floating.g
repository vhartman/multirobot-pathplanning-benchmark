ur_base { multibody }

ur_floatX (ur_base){ joint:transX, limits:[-2 2], mass:.01 }
ur_floatY (ur_floatX){ joint:transY, limits:[-2 2], mass:.01 }
ur_floatZ (ur_floatY){ joint:transZ, limits:[0 3], mass:.01, q: 1 }
ur_floatRX (ur_floatZ){ joint:hingeX, limits:[-3.14159 3.14159], mass:.01 }
ur_floatRY (ur_floatRX){ joint:hingeY, limits:[-3.14159 3.14159], mass:.01 }
ur_ee_link (ur_floatRY){ joint:hingeZ, limits:[-3.14159 3.14159], mass:.01 }

gripper_fill (ur_ee_link){ shape:cylinder, color:[.1, .1, .1 , 1], Q:"d(90 0 1 0) t(-.0 -.0 .025)",
	size:[.05 .021],
	contact:-1
}

# pen
ur_vacuum (ur_ee_link){
    shape:sphere,
    color:[.9, 0, 0 ,1],
    Q:"t(.06 0.0 0.)",
    size:[0.005],
    contact:0
}

ur_ee_marker (ur_ee_link){
    shape: marker,
    size:[0.05]
}
