from abc import ABC, abstractmethod

# abstract class for skills.
class BaseSkill(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def step(self, q, env):
    pass

class EEPoseGoalReaching(BaseSkill):
  def __init__(self, goal, ee_name):
    self.goal = goal
    self.ee_name = ee_name

  def step(self, q, env):
    # get jacobian
    jac = 0
    
    # compute pid law
    current_ee_pose = 0
    error = self.goal - current_ee_pose
    q_dot = jac @ error

    # integrate to get next pos
    q_new = q + dt * q_dot
    return q_new

# simple pid controller
class EEPositionGoalReaching(BaseSkill):
  def __init__(self, goal, ee_name):
    self.goal = goal
    self.ee_name = ee_name

  def step(self, q, env):
    # get jacobian
    jac = 0
    
    # compute pid law
    current_ee_pos = 0
    error = self.goal - current_ee_pos
    q_dot = jac @ error

    # integrate to get next pos
    q_new = q + dt * q_dot
    return q_new

# abstract class for skills.
class BaseTimedSkill(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def step(self, q, t, env):
    raise NotImplementedError


# question: can the mode be changed in a skill?
# or does it need to be two skills?
class VacuumGrasping(BaseTimedSkill):
  def __init__(self, box_pos):
    pass

  def step(self, t, q, env):
    raise NotImplementedError

class EndEffectorPoseFollowing(BaseTimedSkill):
  def __init__(self, line_start_pos, line_goal_pos, ee_name):
    self.line_start_pos = line_start_pos
    self.line_goal_pos = line_goal_pos

    self.ee_name = ee_name

  def step(self, t, q, env):
    # look up where we are on the trajctory
    desired_next_pos = 0
    current_ee_pos = 0
    jac = 0

    # return pt
    pos_error = 0
    rot_error = log_map_rot_error()
    err = [pos_error, rot_error]
    q_dot = gain * jac @ err

    q_new = q + q_dot * dt

    raise NotImplementedError


class EndEffectorPositionFollowing(BaseTimedSkill):
  def __init__(self, line_start_pos, line_goal_pos, ee_name):
    self.line_start_pos = line_start_pos
    self.line_goal_pos = line_goal_pos

    self.ee_name = ee_name

  def step(self, t, q, env):
    # look up where we are on the trajctory and get next position
    # compute control input -> pose is free/might be constrained
    # integrate
    # return pt
    raise NotImplementedError

# cool because it includes multiple robots.
class DualRobotGrasping(BaseTimedSkill):
  def __init__(self):
    self.obj_path = 0
    self.ee_names = []

    self.obj_name = 0

  def step(self, t, q, env):
    # get desired position of obj at time
    # get ee-pos
    # get jacobians
    # do ik to compute the positions of the end effectors
    raise NotImplementedError


# Note: might be a cooler demo if we also have skills that are 'env aware'
# might also be more interesting planning wise.
