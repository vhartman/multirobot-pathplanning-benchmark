base: {}
table(base): {
 shape: ssBox, Q: "t(0 0. .6)", size: [2.5, 2.5, .1, .02], color: [.3, .3, .3],
 contact: 1, logical: { },
 friction: .1
}
Include: <./ur10_vacuum.g>

Edit unique_name_base(table): {Q: "t(0 -.2 .05) d(90 0 0 1)" joint:rigid}