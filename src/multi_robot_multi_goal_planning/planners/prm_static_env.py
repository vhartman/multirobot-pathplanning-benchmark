import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .baseplanner import BasePlanner
from .termination_conditions import (
    PlannerTerminationCondition,
)

from multi_robot_multi_goal_planning.problems.util import interpolate_path, path_cost
from multi_robot_multi_goal_planning.planners import shortcutting
from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseProblem,
    Mode,
    State,
)

class Edge:
  def __init__(self):
    self.q0 = 0
    self.q1 = 0

    self.path = [] # this includes the start and end configuration

    self.is_valid = None

class Node:
  def __init__(self):
    self.q = 0
    self.mode = 0

class Graph:
  def __init__(self):
    self.edges = []
    self.nodes = []
    self.transition_nodes = []

    self.modes = set()

    self.root = None

    self.env = env

  def contains_mode(self, mode):
    if mode in self.modes:
      return True

    return False

  def _get_neighbors(self, node):
    node_neighbors = []

    # compute all node_distances
    node_dists = []
    sorted_node_dists = []

    node_neighbors = []

    # compute transition distances
    transition_dists = []
    sorted_transition_dists = []

    transition_neighbors = []

    return node_neighbors + transition_neighbors

  def search(self, start, goal):
    queue = []

    path = []

    while queue:
      # get current best edge from queue
      edge = queue.pop()

      v0, v1 = edge

      if not env.is_edge_collision_free():
        continue

      if v1 is goal:
        # construct path: represented by sequence of edges
        path = []
        return path

      # get neighbors
      neighbors = self._get_neighbors(v1)
      for n in neighbors:
        # compute cost
        cost = 0

        # add to queue
        pass

      pass

    return None

  def add_path(self, path):
    pass

  def add_node(self, node):
    self.nodes.append(node)

  def add_nodes(self, nodes):
    self.nodes.extend(nodes)

  def add_transition(self, transition_node):
    self.transitions.append(transition_node)

  def add_transitions(self, transition_nodes):
    self.transition_nodes.extend(transition_nodes)

@dataclass
class StaticEnvCompositePRMConfig:
    batch_size: int = 500
    informed_sampling: bool = True
    shortcutting: bool = True

    use_k_nearest: bool = True

class StaticEnvCompositePRM(BasePlanner):
  def __init__(self, config):
    pass

  def _sample_valid_transitions(self, num_nodes):
    num_attempts = 0

    transitions = []

    # TODO: enable batching: it should be cheapter to sample points than
    # to collision check them
    while num_attempt < num_nodes:
      # sample mode
      all_modes = []
      mode_to_expand = all_modes.choice()

      # sample transition from this mode
      next_mode = mode_to_expand.get_following_mode()
      active_task = env.get_active_task()
      
      q = sample_transition(active_task)

      #check validity: we do not consider the mode here, since we are assuming
      # a non-manipulation env
      if env.is_collision_free(q):
        # TODO: need to add info for next mode here
        transitions.append(State(q, next_mode))

    return transitions

  def _sample_valid_state(self, num_nodes):
    num_attempts = 0

    new_nodes = []

    while num_attempt < num_nodes:
      # TODO: sample mode -> only need to do this if 
      # we want to consider constrained planning.
      # Lets ignore this for a second and just sample a config
      mode = None

      q = env.sample_config_uniform_in_limits()

      # check_validity
      if env.is_collision_free(q):
        node = State(q)
        new_nodes.append(node)

    return new_nodes

  def _refine_approximation(self, graph):
    #TODO: consider adding a cost check here to not sample _any_ config
    transition_nodes = self._sample_valid_transitions(self.config.num_transitions)
    free_space_nodes = self._sample_valid_state(self.config.num_free_space)

    graph.add_transitions(transition_nodes)
    graph.add_nodes(free_space_nodes)

  # we can only prune if a vertex is not useful in _any_ mode
  # so currenlty, we only remove if the cost to reach the node and
  # the goal is higher than the best cost
  def _prune(self, graph, best_cost):
    to_remove = []
    for n in graph.nodes:
      lb_cost_for_node = 0
      if best_cost < lb_cost_for_node:
        to_remove.append(n)

    for n in to_remove:
      print("NOT REMOVING")
      pass

    transitions_to_remove = []
    for n in graph.transition_nodes:
      lb_cost_for_node = 0
      if best_cost < lb_cost_for_node:
        transitions_to_remove.append(n)

    for n in transitions_to_remove:
      print("NOT REMOVING")
      pass


  def _is_valid_path(self, path, graph):
    for i in range(len(path)):
      # check edges
      edge = path[i]
      if not is_valid_edge(edge):
        # mark as in collision
        return False
      else:
        # mark as not in collision
        pass

    return True

  def _initialize_graph_with_transitions(self, graph):
    # TODO: do we also want a minimum number of transitions?
    reached_terminal_mode = False
    while not reached_terminal_mode:
        if ptc.should_terminate(cnt, time.time() - start_time):
          return False
      
        transitions = self._sample_valid_transitions(inital_num_transition_samples)
        graph.add_transitions(transitions)

        reached_terminal_mode = graph.contains_mode(terminal_mode)
        if reached_terminal_mode:
          return True

  def plan(
      self,
      ptc: PlannerTerminationCondition,
      optimize: bool = True,
  ) -> Tuple[List[State] | None, Dict[str, Any]]:
      """
      Main entry point for the PRM planner in composite space
      with the assumption that we are only dealing with static environments.
      
      This means no mode changes, only multi goal planning ->
      This thus means that we can always stay on the same roadmap, and 
      possibly reuse collision checks/computations.

      This simplification should allow us to fous on:
      - (better) search
      - how to avoid getting stuck in a local minimum
      - (better) heuristics?
      """
      # collect start position and mode
      q0 = self.env.get_start_pos()
      m0 = self.env.get_start_mode()

      # check initial state/mode
      assert self.env.is_collision_free(q0, m0)

      graph = Graph(
          State(q0, m0),
          lambda a, b: batch_config_dist(a, b, self.config.distance_metric),
          use_k_nearest=self.config.use_k_nearest,
      )

      start_time = time.time()

      info = {"costs": [], "times": [], "paths": []}

      # check if we reached the terminal mode
      initialized = self._initialize_graph_with_transitions(graph)
      if not initialized:
        # TODO LOGGING
        return None

      while True:
        if ptc.should_terminate(cnt, time.time() - start_time):
            break
            
        # TODO: possibly remove for initial version
        # prune
        if current_best_path is not None and current_best_cost is not None:
            self._prune(graph, current_best_cost)
        
        # add new samples
        self._refine_approximation(graph, current_best_cost)

        # run graph search
        path = graph.search(start, goal, current_best_cost)

        # we did not find a path that reaches the goal
        if path is None:
          continue

        # add path to graph, and do some logging
        if path_cost < current_best_cost:
          current_best_path = path
          
          info["costs"].append(path_cost)
          info["times"].append(time.time())
          info["paths"].append(path)
          
          # shortcut
          shortcut_path = shortcutting.robot_mode_shortcut()

          if shortcut_path_cost < path_cost:
            current_best_path = shortcut_path

            # add shortcutted path to graph
            graph.add_path(shortcut_path)

            # add info
            info["costs"].append(shortcut_path_cost)
            info["times"].append(time.time())
            info["paths"].append(path)

      return current_best_path, info 