# Original MuJoCo tool meshes (visual only) and editable collision spheres.
# All sphere Q translations and size radii are in the welding_tool frame.
# The tool mounting pose and TCP are preserved by the scene converter.
welding_tool {}

visual_0 (welding_tool) {Q:[0.0136001825232 9.40896432712e-06 0.0726354053279 0.0457008550835 0.949801986779 -0.260341198571 -0.167362116627] shape:mesh mesh:<tool_visual_0.off> color:[0 0.0219199992716 0.152329996228 1] contact:0}
visual_1 (welding_tool) {Q:[4.54759612265e-06 1.35733625418e-05 0.0162542878308 0.0891084294883 0.701481819393 -0.0890111034592 0.701469862729] shape:mesh mesh:<tool_visual_1.off> color:[0.0522200018167 0.0522200018167 0.0522200018167 1] contact:0}
visual_2 (welding_tool) {Q:[0.0348604499753 -3.16640860689e-08 0.129614139852 -0.0770705069602 0.915522157829 0.211757864246 -0.333223532198] shape:mesh mesh:<tool_visual_2.off> color:[1 0.135140001774 0 1] contact:0}

# Edit these spheres; contact:1 marks collision geometry.
tool_sphere_00 (welding_tool) {Q:"t(0.031541396919694591 9.9338329988089835e-17 0.12195528940425686)" shape:sphere size:[0.009022321314768941] color:[0.2 0.85 0.3 0.4] contact:1}
tool_sphere_01 (welding_tool) {Q:"t(0.025093529358572939 7.827939685345342e-17 0.10423991609843258)" shape:sphere size:[0.010022321314768941] color:[0.2 0.85 0.3 0.4] contact:1}
tool_sphere_02 (welding_tool) {Q:"t(0.018645661797451224 5.7214746246423039e-17 0.086524542792608308)" shape:sphere size:[0.010022321314768941] color:[0.2 0.85 0.3 0.4] contact:1}
tool_sphere_03 (welding_tool) {Q:"t(0.012197794236329511 3.6150095639392657e-17 0.068809169486784019)" shape:sphere size:[0.010022321314768941] color:[0.2 0.85 0.3 0.4] contact:1}
tool_sphere_04 (welding_tool) {Q:"t(0.0057499266752077983 1.5085656790599089e-17 0.051093796180959855)" shape:sphere size:[0.010022321314768941] color:[0.2 0.85 0.3 0.4] contact:1}
tool_sphere_05 (welding_tool) {Q:"t(-0.00069794088591385907 1.0504873495090271e-16 0.033378422875135573)" shape:sphere size:[0.010022321314768941] color:[0.2 0.85 0.3 0.4] contact:1}

tool_sphere_12 (welding_tool) {Q:"t(0.036766529312731722 1.0836422100155029e-12 0.13188571711663102)" shape:sphere size:[0.007052810210297362] color:[0.2 0.85 0.3 0.4] contact:1}
tool_sphere_13 (welding_tool) {Q:"t(0.045075312606372173 1.0836127798557506e-12 0.14164794823188112)" shape:sphere size:[0.005752810210297362] color:[0.2 0.85 0.3 0.4] contact:1}
