import numpy as np
import math
from collections import Counter
from kdtree import kdtree
import json
import time

class Node:
    _agents = None
    _mode = None

    def __init__(self, q):
        self.state = q
        self.cost = 0            
        self.parent = None  
        self.children = []    
        self.mode = Node._mode    
        self.transition_mode = None
        self.agent_cost = {i: 0 for i in range(len(self._agents))}
        self.mode_cost = 0

    @property
    def coords(self):
        return tuple(np.concatenate(list(self.state.values())))
    
    @classmethod
    def set_defaults(cls, mode, agents):
        cls._mode = mode
        cls._agents = agents

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        return self.coords[i]

    def __repr__(self):
        return f"<Node - {self.coords}, Cost: {self.cost}>"
 
class Metric:
    @staticmethod
    def EuclideanDistanceNAgents(n1, n2): # All agents
        n1_state = n1.data.state if hasattr(n1, 'data') else n1.state
        return sum(Metric.EuclideanDistance1Agent(n1, n2, agent)[1] for agent in range(len(n1_state)))

    @staticmethod
    def EuclideanDistance1Agent(n1, n2, agent): # One agent
        n1_state = n1.data.state if hasattr(n1, 'data') else n1.state
        n2_state = n2.data.state if hasattr(n2, 'data') else n2.state
        dim = len(n1_state[agent])

        if dim in {2, 3}:  # q = (x, y) or  = (x, y, theta)
            diff = np.array(n2_state[agent][:2]) - np.array(n1_state[agent][:2])
            if dim == 3:
                orient_diff = (n2_state[agent][-1] - n1_state[agent][-1] + np.pi) % (2 * np.pi) - np.pi
                diff = np.hstack((diff, orient_diff))
                
        else: # q = (q1, q2, q3, q4, ...) with qi as joint angles
            diff = np.array([(n2_state[agent][i] - n1_state[agent][i] + np.pi) % (2 * np.pi) - np.pi for i in range(dim)])

        return diff, round(np.linalg.norm(diff), 5)
    
    @staticmethod
    def EuclideanDistanceState(n1, n2, agents, num_DOF):
        diff = []
        if num_DOF == 2: # q = (x, y)
            diff = np.array(list(n2.coords)) - np.array(list(n1.coords))
        elif num_DOF == 3: # q = (x, y, theta      
            for i in range(len(n1.coords)):
                if (i + 1) % 3 !=0:
                    diff.append(n2.coords[i]-n1.coords[i])
                else:
                    diff.append((n2.coords[i] - n1.coords[i] + np.pi) % (2 * np.pi) - np.pi)
        else: # q = (q1, q2, q3, q4, ...) with qi as joint angles
            diff = [(n2.coords[i] - n1.coords[i] + np.pi) % (2 * np.pi) - np.pi for i in range(len(agents))]

        return np.array(diff), round(np.linalg.norm(diff), 5)

    @staticmethod
    def Interpolation(start, end, agents, num_DOF, num_states=2):
        # Multi-point interpolation (Collision checking)   
        diff, _ = Metric.EuclideanDistanceState(start, end, agents, num_DOF)    
        interpolated_states = []         
        for i in range(num_states +1):
            t = i/num_states
            interpolated_state = list(start.coords) + t * diff
            interpolated_states.append(Transformation.ListToDict(interpolated_state, len(agents)))
        return interpolated_states

class RRTstar:
    def __init__(self, env):
        self.env = env
        self.goal_radius = self.env.goal_radius
        self.step_size = config.get('step_size')
        self.max_iter = config.get('max_iter')    
        self.probability_new_mode = config.get('mode_probability')
        self.ptc_threshold = config.get("ptc_threshold")
        self.num_agents = num_agents
        self.cost_function = cost_function
        self.num_DOF = num_DOF
        self.dim = self.num_agents*self.num_DOF # sum of #DOF of all agents
        # self.gamma = ((2 *(1 + 1/self.dim))**(1/self.dim) * (self.FreeSpace()/self.UnitBallVolume())**(1/self.dim))*1.1 
        self.r = self.step_size * 6
        self.directory = directory
        self.analysis = Analysis(self.env, config, directory, self.cost_function)
        self.operation = Operation(mode_sequence, task_sequence, self.env.agents)
        self.tree = kdtree.create(dimensions=self.dim)# Only needed for visualization
        self.tree_size = 0
        self.logger = logger # Only needed for visualization
        self.optimization_mode_idx = 0
        self.start = time.time()
        
    def Cost(self, n1, n2):
        if self.cost_function == 1:
            return Metric.EuclideanDistanceNAgents(n1, n2)
        if self.cost_function == 2:
            return self.AgentCost(n1, n2, worst_agent=True)
            
    def SearchHeuristic(self, n1, n2, check_constraint = False, constrained_agent = None):
        if self.cost_function == 1:
            if check_constraint: #needed to check if goal of constrained agent is reached
                return Metric.EuclideanDistance1Agent(n1, n2, constrained_agent)[1]
            return Metric.EuclideanDistanceNAgents(n1, n2)
        if self.cost_function == 2:
            if check_constraint: #needed to check if goal of constrained agent is reached
                return Metric.EuclideanDistance1Agent(n1, n2, constrained_agent)[1]
            return Metric.EuclideanDistanceNAgents(n1, n2)
               
    def AgentCost(self, n1, n2, worst_agent=False): 
        if self.cost_function == 1:
            cost = {}
            for agent in range(self.num_agents):
                cost[agent] = Metric.EuclideanDistance1Agent(n1, n2, agent)[1]
        if self.cost_function == 2 and not worst_agent:
            cost = {}
            for agent in range(self.num_agents):
                cost[agent] = Metric.EuclideanDistance1Agent(n1, n2, agent)[1]
        if self.cost_function == 2 and worst_agent:
            cost = 0
            for agent in range(self.num_agents):
                cost_agent = Metric.EuclideanDistance1Agent(n1, n2, agent)[1]
                if cost_agent > cost:
                    cost = cost_agent
        return cost

    def Nearest(self, n_rand):
        nearest_node, distance = self.operation.current_mode.subtree.search_nn(n_rand.data, self.SearchHeuristic)
        return  nearest_node
    
    def Steer(self, n_nearest, n_rand):
        dist = self.SearchHeuristic(n_nearest, n_rand)
        if dist <= self.step_size:
            return n_rand
        #Interpolation
        diff, norm = Metric.EuclideanDistanceState(n_nearest.data, n_rand.data, self.env.agents, self.num_DOF)    
        num_steps = int(np.ceil(norm / self.step_size))
        t = 1 / num_steps
        interpolated_state = list(n_nearest.data.coords) + t * diff

        new_state = Transformation.ListToDict(interpolated_state, len(self.env.agents))
        n_new = Node(new_state)
        # print([n_nearest.data.state, n_new.state, n_rand.data.state])
        return kdtree.KDNode(data=n_new, dimensions=len(n_new.coords))
    
    def Near(self, n_new):
        # n_nodes = sum(1 for _ in self.operation.current_mode.subtree.inorder()) + 1
        #TODO generalize the radius!!!!!
        # r = min((7)*self.step_size, 3 + self.gamma * ((math.log(n_nodes) / n_nodes) ** (1 / self.dim)))
        N_near = self.operation.current_mode.subtree.search_nn_dist(n_new.data, self.r, dist = self.SearchHeuristic) 
        if not self.env.informed:
            return N_near
        else:
            if self.operation.task_sequence != [] and self.operation.task_sequence[0] == self.operation.current_mode.constraint.label:
                return N_near
            #N_near needs to be preselected
            return self.env.in_informed_subset(N_near, self.operation)        
    
    def FindParent(self, N_near, n_new, n_nearest):
        c_min = n_nearest.data.cost + self.Cost(n_nearest, n_new)
        n_min = n_nearest.data
        for n_near in N_near:
            c_new = n_near.cost + self.Cost(n_near, n_new.data)
            if c_new < c_min and self.env.is_collision_free(n_near, n_new):
                c_min = c_new
                n_min = n_near
        n_new.data.cost = c_min
        n_new.data.parent = kdtree.KDNode(data=n_min, dimensions=len(n_min.coords))
        n_min.children.append(n_new.data) #Set child
        n_new.data.agent_cost = dict(Counter(n_new.data.parent.data.agent_cost) + Counter(self.AgentCost(n_new.data.parent.data, n_new.data)))
        self.tree.add(n_new.data)
        self.operation.current_mode.subtree.add(n_new.data)
        self.tree_size+=1
        self.operation.current_mode.subtree_size +=1

    def FreeSpace(self):
        #TODO adapt it individually for each agent
        X_free = (self.env.grid_size)**2
        for obstacle in self.env.obstacles:
            X_free -= obstacle.geometry.area
        return X_free

    def UnitBallVolume(self):
        return math.pi ** (self.dim / 2) / math.gamma((self.dim / 2) + 1)

    def Rewire(self, N_near, n_new, costs_before):
        rewired = False
        for n_near in N_near:
            costs_before.append(n_near.cost)
            if n_near != n_new.data.parent.data and n_near != n_new.data:         
                c_pot = n_new.data.cost + self.Cost(n_new.data, n_near)
                c_agent = dict(Counter(n_new.data.agent_cost) + Counter(self.AgentCost(n_new.data, n_near)))
                if c_pot < n_near.cost:
                    if self.env.is_collision_free(n_near, n_new):
                        #reset children
                        n_near.parent.data.children.remove(n_near)
                        #set parent
                        n_near.parent = n_new
                        if n_new.data != n_near:
                            n_new.data.children.append(n_near)
                        n_near.cost = c_pot
                        n_near.agent_cost = c_agent
                        rewired = True
        return rewired
    
    def GeneratePath(self, node):
        path = {i: [] for i in range(len(self.env.agents))}
        path_nodes = []
        while node:
            path_nodes.append(node)
            for agent in self.env.agents:
                path[agent.name].append(node.data.state[agent.name])
            node = node.data.parent
        self.operation.path = {agent: positions[::-1] for agent, positions in path.items()}     
        self.operation.path_nodes = path_nodes[::-1]
        self.operation.cost = self.operation.path_nodes[-1].data.cost

    def UpdateCost(self, n):
        stack = [n]
        while stack:
            current_node = stack.pop()
            n_agent_cost = Counter(current_node.agent_cost)
            for child in current_node.children:
                child.cost = current_node.cost + self.Cost(current_node, child)
                child.agent_cost = dict(n_agent_cost + Counter(self.AgentCost(current_node, child)))
                stack.append(child)
   
    def SetOptimizationModeIdx(self):
        if self.operation.task_sequence == []:
            self.optimization_mode_idx  = -1       
        elif len(self.operation.modes) == 1:
            self.optimization_mode_idx = 0
        else:
            self.optimization_mode_idx = -2

    def SetPathOptimalTransitionNode(self, iteration):
        self.SetOptimizationModeIdx()
        transition_nodes = self.operation.modes[self.optimization_mode_idx].transition_nodes
        lowest_cost = np.inf
        lowest_cost_idx = None
        for idx, node in enumerate(transition_nodes):
            if node.data.cost < lowest_cost and node.data.cost < self.operation.cost:
                lowest_cost = node.data.cost
                lowest_cost_idx = idx
        if lowest_cost_idx is not None: 
            self.GeneratePath(transition_nodes[lowest_cost_idx]) 
            print(f"iter  {iteration}: Changed cost to ", self.operation.cost, " Mode ", self.operation.modes.index(self.operation.current_mode))
            if (self.operation.ptc_cost - self.operation.cost) > self.ptc_threshold:
                self.operation.ptc_cost = self.operation.cost
                self.operation.ptc_iter = iteration

            end_iteration = time.time()
            passed_time = end_iteration - self.start
            self.analysis.SavePklFIle(self.tree, self.operation, self.env.colors, passed_time, self.num_DOF) 

    def AddTransitionNode(self, n):
            idx = self.operation.modes.index(self.operation.current_mode)
            if idx != len(self.operation.modes) - 1:
                self.operation.modes[idx + 1].subtree.add(n.data)
                self.operation.modes[idx+1].subtree_size +=1
    
    def ManageTransition(self, n_new, constrained_agent, iteration):
        cost_constrained = self.SearchHeuristic(self.operation.current_mode.constraint.transition, n_new.data, True, constrained_agent)
        if  cost_constrained < self.goal_radius:
            self.operation.current_mode.transition_nodes.append(n_new)
            # Check if initial transition node of current mode is found
            if self.operation.task_sequence != [] and self.operation.current_mode.constraint.label == self.operation.task_sequence[0]:
                print(f"iter  {iteration}: A{self.env.agents[constrained_agent].name} found T{self.operation.task_sequence[0][constrained_agent]}: Cost: ", n_new.data.cost)
                self.operation.task_sequence.pop(0)
                init_path = True
                amount_modes = len(self.operation.modes)
                if self.operation.task_sequence != []:
                    init_path = False
                    self.operation.modes.append(self.operation.mode_sequence[amount_modes])
                else:
                    self.operation.ptc_iter = iteration
                    self.operation.ptc_cost = n_new.data.cost
                self.GeneratePath(n_new)
                end_iteration = time.time()
                passed_time = end_iteration - self.start
                self.analysis.SavePklFIle(self.tree, self.operation, self.env.colors, passed_time, self.num_DOF, init_path = init_path) 
            self.AddTransitionNode(n_new)
        self.SetPathOptimalTransitionNode(iteration)

    def SetModePorbability(self):
        num_modes = len(self.operation.modes)
        if num_modes == 1:
            return
        
        if self.operation.task_sequence == [] and self.probability_new_mode != 0:
                probability = [1/num_modes] * num_modes
        
        elif self.probability_new_mode == 'None':
            # equally
            probability = [1 / (num_modes)] * (num_modes)

        elif self.probability_new_mode == 1:
            # greedy (only latest mode is selected until all initial paths are found)
            probability = [0] * (num_modes)
            probability[-1] = 1

        elif self.probability_new_mode == 0:
            # Uniformly
            total_transition_nodes = sum(len(mode.transition_nodes) for mode in self.operation.modes)
            total_nodes = self.tree_size + total_transition_nodes
            # Calculate probabilities inversely proportional to node counts
            inverse_probabilities = [
                1 - (mode.subtree_size / total_nodes)
                for mode in self.operation.modes
            ]
            # Normalize the probabilities to sum to 1
            total_inverse = sum(inverse_probabilities)
            probability = [
                inv_prob / total_inverse for inv_prob in inverse_probabilities
            ]

        else:
            # manually set
            total_transition_nodes = sum(len(mode.transition_nodes) for mode in self.operation.modes)
            total_nodes = self.tree_size + total_transition_nodes
            # Calculate probabilities inversely proportional to node counts
            inverse_probabilities = [
                1 - (mode.subtree_size / total_nodes)
                for mode in self.operation.modes[:-1]  # Exclude the last mode
            ]

            # Normalize the probabilities of all modes except the last one
            remaining_probability = 1-self.probability_new_mode  
            total_inverse = sum(inverse_probabilities)
            probability = [
                (inv_prob / total_inverse) * remaining_probability
                for inv_prob in inverse_probabilities
            ] + [self.probability_new_mode]

        self.operation.mode_probability = probability

    def Initialization(self):
        """ Initialize values"""
        if self.env.informed:
            for mode in self.operation.mode_sequence:
                agent_name = next(iter(mode.constraint.label))
                task_nr = mode.constraint.label[agent_name]
                cmin_diff = self.goal_radius
                if task_nr == 0:
                    state_start = {agent_name: self.env.agents[agent_name].start}
                else:
                    state_start = {agent_name: self.env.agents[agent_name].tasks[task_nr-1]}
                    cmin_diff *= 2
                mode.constraint.start = Node(state_start)
                cmin, mode.constraint.C = self.env.rotation_to_world_frame(mode.constraint.start, mode.constraint.transition ,agent_name)
                mode.constraint.cmin = cmin-cmin_diff
                mode.constraint.state_centre = (mode.constraint.start.state[agent_name] + mode.constraint.transition.state[agent_name])/2
        for agent in self.env.agents:
            agent.cs = self.env.configuration_space(agent)

    def Plan(self):

        self.logger.info('Step size: %s', json.dumps(self.step_size, indent=2)) 
        self.logger.info('Goal radius: %s', json.dumps(self.goal_radius, indent=2)) 
        self.logger.info('Max iteration: %s', json.dumps(self.max_iter, indent=2))
        self.logger.info('Probability of newly added mode: %s', json.dumps(self.probability_new_mode, indent=2))
        self.logger.info('PTC threshold: %s', json.dumps(self.ptc_threshold, indent=2))
        i = 0
        self.Initialization()

        while True:
            # Mode selection
            self.SetModePorbability()
            self.operation.current_mode = (np.random.choice(self.operation.modes, p = self.operation.mode_probability))
            # Initialization of selected mode
            Node.set_defaults(self.operation.current_mode, self.env.agents)
            constrained_agent = next(iter(self.operation.current_mode.constraint.label))
            if self.tree.data is None:
                self.tree.add(Node(self.operation.start_node))
                self.operation.current_mode.subtree.add(Node(self.operation.start_node))
                self.tree_size+=1
                self.operation.current_mode.subtree_size +=1

            # RRT* core
            n_rand, n_rand_label = self.env.sample_manifold(self.operation)
            n_nearest = self.Nearest(n_rand)       
            n_new = self.Steer(n_nearest, n_rand)
            if self.env.is_collision_free(n_nearest, n_new):
                N_near = self.Near(n_new)
                self.FindParent(N_near, n_new, n_nearest)
                costs_before = []
                self.analysis.LogData(i, self.operation.current_mode.subtree, n_rand, n_nearest, n_new, N_near, costs_before, self.r, self.operation.current_mode.label)

                if self.Rewire(N_near, n_new, costs_before):
                    self.UpdateCost(n_new.data)
                self.analysis.LogData(i, self.operation.current_mode.subtree, n_rand, n_nearest, n_new, N_near, costs_before, self.r, self.operation.current_mode.label, False)
                     
                self.ManageTransition(n_new, constrained_agent, i)
                if i%200 == 0:
                    print("iter ",i)
                    end_iteration = time.time()
                    passed_time = end_iteration - self.start
                    self.analysis.SavePklFIle(self.tree, self.operation, self.env.colors, passed_time, self.num_DOF, n_new = n_new.data, N_near = N_near, r =self.r, n_rand = n_rand.data.state, n_rand_label=n_rand_label)
            if self.operation.task_sequence == [] and i != self.operation.ptc_iter and (i- self.operation.ptc_iter)%self.max_iter == 0:
                diff = self.operation.ptc_cost - self.operation.cost
                self.operation.ptc_cost = self.operation.cost
                if diff < self.ptc_threshold:
                    break
            i += 1

                 
        end_iteration = time.time()
        passed_time = end_iteration - self.start
        self.analysis.SavePklFIle(self.tree, self.operation, self.env.colors, passed_time, self.num_DOF, n_new = n_new.data, N_near = N_near, r =self.r, n_rand = n_rand.data.state, n_rand_label=n_rand_label)
             
class Operation:
    def __init__(self, mode_sequence, task_sequence, agents):
        self.modes = [mode_sequence[0]]
        self.mode_sequence= mode_sequence
        self.task_sequence = task_sequence
        self.current_mode = None
        self.agents = agents
        self.path = {i: None for i in range(len(agents))}
        self.path_nodes = None
        self.cost = 0
        self.ptc_cost = 0
        self.ptc_iter = None
        self.mode_probability = [1]

    @property
    def start_node(self):
        start = {}
        for agent in self.agents: 
                start[agent.name] = agent.start
        return start
     
class Mode:
    """Mode description"""
    def __init__(self, mode, dim):
        self.label = mode
        self.constraint = Constraint()
        self.transition_nodes = []
        self.subtree = kdtree.create(dimensions=dim)
        self.subtree_size = 0

class Constraint:
    """Contains the constraints description for the agent that needs to achieve its transition in this mode """
    def __init__(self):
        self.label = None 
        self.transition = None 
        self.C = None
        self.L = None
        self.cmin = None
        self.state_centre = None
        self.start = None

    def cmax(self, transition_nodes, path_nodes, goal_radius):
        if transition_nodes == []:
            return None
        else:
            agent_name = next(iter(self.label))
            for node in transition_nodes:
                if node in path_nodes:
                    c = node.data.agent_cost[agent_name]
                    break
            return c-self.get_start_node(path_nodes, goal_radius, agent_name)

    def get_start_node(self, path_nodes, goal_radius, agent_name):
        for node in path_nodes:
            if np.array_equal(node.data.state[agent_name], self.start.state[agent_name]) or Metric.EuclideanDistance1Agent(node, self.start, agent_name)[1] < goal_radius:
                return node.data.agent_cost[agent_name]


