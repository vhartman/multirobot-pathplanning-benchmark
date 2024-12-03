# Benchmarks for multi robot multi goal motion planning

This repository provides some multi-robot-multi-goal motion planning problems, and some (naive) baselines for finding a solution.

The corresponding paper can be found at ~~not available yet~~.

# Installation
The problems we propose here are built on top of [rai](https://marctoussaint.github.io/robotic/index.html) ([github](https://github.com/MarcToussaint/robotic)), which therefore needs to be installed as prerequisite using

```
pip install robotic
```

I would recommend using the virtual environment of your choice to make sure nothing break with rai.
Other than that, this module should work out of the box after cloning/forking.

# Overview and Usage

We formulate some multi robot multi goal motion planning problems, and try to provide some baselines and base-classes to formulate your own problems.
At the moment, all of thi sis in python, with collision checks and other performance critical parts happening in cpp.

#### Nomenclature

- **Configuration**: A configuration describes the joint-pose that all robots are in.
- **Task**: A task specifies the robots that are involved, a goal (which can be a single pose, a region, or a set of constraints)
- **Mode**: A mode describes what each robot is doing at a time.
- **Path**: A path maps time (or an index) to a configuration for each robot, and a mode.

#### Problem & Formulation

The basic formulation is the following: We assume that we have $n$ robots that have to fulfill some tasks.
We assume that the tasks are assigned to a robot, and in the current state, a task essentially means to reach a goal pose (compared to e.g., fulfilling a constraint along a path).

We then propose two scenarios of how task ordering can be specified:
- using a sequence, i.e., a total ordering of all tasks, i.e., task $A$ has to be done before task $B$
- using a dependency graph, which only tells us an ordering for some of the tasks.

The goal of a planner is then to find a path for all robots that fulfills the tasks at hand.
A path consists of a configuration for all robot at a time-index, and the index of the task that each robot is currently doing.

#### Files

- `planning_env.py` implements the abstract base class for a planning problem.
There are two main requirements:
  - we need a scene that we are planning in, and
  - we need a description o fthe tast sequence, respectively the dependence between tasks

- `rai_envs.py` implements some concrete environments, ans some utilities to have a look at them.

- `benchmark.py` gives a test harness for benchmarking algorithms on a set of planning problems.

#### Using your own robot environment

To implement your own problem, you need to implement the required functions of the `env` base-class.

#### Specifying your own problems

A problem consists of the initial scene, and a task sequence or a dependency graph.

#### Implementing your own planner

To implement your own planner, you need to specialize the `planner` base-class.

# Extension & Future work

#### Path constraints
This framework technically supports both path and goal constraints, but at the moment, only goal constraints are implemented and used.
However, in some applications, this is necessary to e.g. formulate a constraint of 'hold the glass upright'

#### Kinodynamic motion planning
Similarly as above, the formulation we propose here allows for kinodynamic motion planning, but we do not have a scene at the moment that tests this.

#### More flexible task planning
In the moment, we only support formulating the task structure as dependency graph or as sequence.
It would theoretically be possible to us ethe formulation we propse here to implement and benchmark task and motion planning solvers.