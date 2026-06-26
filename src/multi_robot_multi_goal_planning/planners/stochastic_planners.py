"""
Stochastic path planning

1. CL reactive planner (MDP solved via DP)
2. OL conservative planner (worst-case envelope avoidance)

Flowchart: 
1. Collect active robot rollouts, simulating the stochastic skill
2. Bin the active configuration (spatial binning) and calculate transitions
3. Build a local grid for the inactive robot and find adjacent neighbors
4. (TBD) collision check between active bins and inactive grid nodes
5. Run Backward Induction to compute optimal value and policy tables
6. Execute CL or OL policies on fresh rollouts


IMPLEMENTATION:

Imports
Configuration (dataclass)

Core part / algorithms (ipad notes)
- bin_rollouts
- transition_probs
- build_grid
- dijkstra
- 
- backward_induction

Stochastic planner class
- init
- setup?
- plan (reactive vs. conservatice)
- evaluate
"""