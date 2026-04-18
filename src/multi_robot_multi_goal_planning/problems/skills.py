import numpy as np

from abc import ABC, abstractmethod
import robotic

from dataclasses import dataclass
from typing import Optional, List
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d

import time

##########
# Note: might be a cooler demo if we also have skills that are 'env aware'
# might also be more interesting planning wise.
##########

# TODO:
# - enable choosing only subset of joints to plan for -> enables planning for e.g. grippers/dex hands
# - unify interface -> merge (t,q) into 'state' or somethign like that to make life easier

@dataclass
class SkillRolloutResult:
  trajectory: np.ndarray
  times: np.ndarray
  is_deterministic: bool = True
  distributions: Optional[List] = None # Later with stochastic skills?
  # ...

# abstract class for skills. 
class DeterministicBaseSkill(ABC):
  def __init__(self, joints, dt=0.1):
    self.joints = joints # Store joint names when passed by planner
    self.dt = dt

  @abstractmethod
  def step(self, q, env):
    pass

  # TODO: move to step itself? two return values?
  @abstractmethod
  def done(self, q, env):
    pass

  def rollout(self, q_init, task, all_joints, env, t0, max_steps=1000):
    """
    Rollout deterministic untimed skill till convergence
    """
    env.C.selectJoints(task.skill.joints) # Restrict to subspace
    q = q_init.copy()
    trajectory = [q]
    times = [t0]
    
    for _ in range(max_steps):
        q = self.step(q, env)
        times.append(times[-1] + self.dt)
        trajectory.append(q)
        
        if self.done(q, env):
            break
        
    env.C.selectJoints(all_joints) # Restore full space
    return SkillRolloutResult(
        trajectory=np.array(trajectory),
        times=np.array(times),
    )

# abstract class for stochastic skills.
class StochasticBaseSkill(ABC):
  def __init__(self, joints, dt=0.1):
    self.joints = joints
    self.dt = dt

  @abstractmethod
  def step(self, q, env):
    pass

  @abstractmethod
  def done(self, q, env):
    pass

# abstract class for deterministic timed skills.
class BaseDeterministicTimedSkill(ABC):
  def __init__(self, joints, dt=0.01):
    self.joints = joints
    self.dt = dt

  # TODO: should likely simply merge q and t to 'state'
  @abstractmethod
  def step(self, t, q, env):
    raise NotImplementedError

  @abstractmethod
  def done(self, t, q, env):
    pass
  
  def rollout(self, q_init, task, all_joints, env, t0):
    """
    Rollout deterministic timed skill for fixed duration
    """
    env.C.selectJoints(task.skill.joints) # Restrict to subspace
    n_steps = max(1, round(self.duration / self.dt))
    q = q_init.copy()
    trajectory = [q]
    times = [t0]

    for i in range(n_steps):
        t_norm = (i + 1) / n_steps
        q = self.step(t_norm, q, env)
        times.append(times[-1] + self.dt)
        trajectory.append(q)
        
        if self.done(t_norm, q, env):
            break
    
    env.C.selectJoints(all_joints) # Restore full space
    return SkillRolloutResult(
        trajectory=np.array(trajectory),
        times=np.array(times),
    )

# abstract class for stochastic timed skills.
class BaseStochasticTimedSkill(ABC):
  def __init__(self, joints):
    self.joints = joints

  @abstractmethod
  def step(self, q, t, env):
    raise NotImplementedError

  @abstractmethod
  def done(self, q, t, env):
    pass

class EEPositionGoalReaching(DeterministicBaseSkill):
  def __init__(self, joints, goal, ee_name, dt=0.1, ik_gain=1.0):
    super().__init__(joints, dt=dt)

    self.goal_position = goal
    self.ee_name = ee_name
    self.ik_gain = ik_gain

    self.qdot_clip = 0.2

  def step(self, q, env):
    # get jacobian
    env.C.setJointState(q, self.joints)
    [err, jac] = env.C.eval(robotic.FS.position, [self.ee_name], 1, self.goal_position)
    
    # compute pid law
    q_dot = np.linalg.pinv(jac) @ err

    # integrate to get next pos
    q_new = q - self.ik_gain * q_dot
    return q_new

  def done(self, q, env):
    env.C.setJointState(q, self.joints)
    [err, jac] = env.C.eval(robotic.FS.position, [self.ee_name], 1, self.goal_position)
    
    if np.linalg.norm(err) < 1e-3:
      return True

    return False

# simple pid controller
class EEPoseGoalReaching(DeterministicBaseSkill):
  def __init__(self, joints, goal, ee_name, dt=0.1, ik_gain=1.0):
    super().__init__(joints, dt=dt)
    
    self.goal_pose = goal
    self.ee_name = ee_name
    self.ik_gain = ik_gain

    self.scale_stepsize = False

  def step(self, q, env):
    # get jacobian
    env.C.setJointState(q, self.joints)

    ee_pose = env.C.getFrame(self.ee_name).getPose()

    mod_goal_pose = self.goal_pose * 1.
    if np.dot(ee_pose[3:], self.goal_pose[3:]) < 0:
      mod_goal_pose[3:] = -mod_goal_pose[3:]

    [err, jac] = env.C.eval(robotic.FS.pose, [self.ee_name], 1, mod_goal_pose)

    # compute pid law
    q_dot = np.linalg.pinv(jac) @ err

    if self.scale_stepsize:
      max_step = 0.5
      current_speed = np.linalg.norm(q_dot)
      if current_speed > max_step:
          q_dot = (q_dot / current_speed) * max_step

    # integrate to get next pos
    q_new = q - self.ik_gain * q_dot

    # print(env.C.getFrame(self.ee_name).getPose())
    # print(self.goal_pose)

    # print(err)

    # env.C.setJointState(q_new, self.joints)
    # env.C.view(True)

    return q_new

  def done(self, q, env):
    # get jacobian
    env.C.setJointState(q, self.joints)

    ee_pose = env.C.getFrame(self.ee_name).getPose()
    mod_goal_pose = self.goal_pose * 1.
    if np.dot(ee_pose[3:], self.goal_pose[3:]) < 0:
      mod_goal_pose[3:] = -mod_goal_pose[3:]

    [err, jac] = env.C.eval(robotic.FS.pose, [self.ee_name], 1, mod_goal_pose)

    if np.linalg.norm(err) < 1e-3:
      return True

    return False

# Should probably do this for all the other pose reaching things as well
# This can for example be used for a handover
class RelativePoseReaching(DeterministicBaseSkill):
  def __init__(self, joints, frame_1_name, frame_2_name, transformation, dt=0.1):
    super().__init__(joints, dt=dt)
    
    self.frame_1_name = frame_1_name
    self.frame_2_name = frame_2_name
    self.relative_transformation = transformation

    # To deal with the double covering of the quaternions
    self.mod_rel_transformation = 1. * self.relative_transformation
    self.mod_rel_transformation[3:] = -self.mod_rel_transformation[3:]

  def step(self, q, env):
    env.C.setJointState(q, self.joints)

    # TODO: Pretty sure this could be done better??
    [err_1, jac_1] = env.C.eval(robotic.FS.poseRel, [self.frame_1_name, self.frame_2_name], 1, self.relative_transformation)
    [err_2, jac_2] = env.C.eval(robotic.FS.poseRel, [self.frame_1_name, self.frame_2_name], 1, self.mod_rel_transformation)

    if np.linalg.norm(err_1) < np.linalg.norm(err_2):
      q_dot = np.linalg.pinv(jac_1) @ err_1
    else:
      q_dot = np.linalg.pinv(jac_2) @ err_2

    # integrate to get next pos
    q_new = q - self.dt * q_dot
    
    return q_new

  def done(self, q, env):
    env.C.setJointState(q, self.joints)

    [err_1, jac_1] = env.C.eval(robotic.FS.poseRel, [self.frame_1_name, self.frame_2_name], 1, self.relative_transformation)
    [err_2, jac_2] = env.C.eval(robotic.FS.poseRel, [self.frame_1_name, self.frame_2_name], 1, self.mod_rel_transformation)

    if np.linalg.norm(err_1) < 1e-3 or np.linalg.norm(err_2) < 1e-3:
      return True

    return False

# question: can the mode be changed in a skill?
# or does it need to be two skills?
class VacuumGrasping(BaseDeterministicTimedSkill):
  def __init__(self, joints, box_pos):
    super().__init__(joints)


  def step(self, t, q, env):
    raise NotImplementedError

  def done(self, t, q, env):
    raise NotImplementedError

class EndEffectorPoseFollowing(BaseDeterministicTimedSkill):
  def __init__(self, joints, ee_name, poses, times=None, dt=0.01, ik_gain=0.1):
    super().__init__(joints, dt=dt)
    
    self.line_start_pos = line_start_pos
    self.line_goal_pos = line_goal_pos

    self.duration = 1
    self.ik_gain = ik_gain

    self.ee_name = ee_name

    self.poses = np.array(poses)
    num_poses = len(self.poses)
    
    if times is None:
        self.times = np.linspace(0,1,num_poses)
    else:
        self.times = np.array(times)
    
    self.pos_interp = interp1d(self.times, self.poses[:, :3], axis=0, kind='linear')

    self.key_rots = R.from_quat(self.poses[:, 3:], scalar_first=True)
    self.slerp = Slerp(self.times, self.key_rots)

  def _get_desired_obj_pose_at_time(self, t):
    t = np.clip(t, self.times[0], self.times[-1])
    
    # Interpolate position
    p_new = self.pos_interp(t)
    
    # Interpolate rotation
    R_t = self.slerp([t])[0]
    q = R_t.as_quat(scalar_first=True)

    return np.concatenate([p_new, q])    
    
  def step(self, t, q, env):
    # look up where we are on the trajctory
    desired_next_pos = self._get_desired_pose_at_time(t)

    q_new = 1. * q

    for _ in range(100):
      env.C.setJointState(q_new, self.joints)
      [err, jac] = env.C.eval(robotic.FS.pose, [self.ee_name], 1, desired_next_pos)

      # compute pid law
      q_dot = np.linalg.pinv(jac) @ err

      # integrate to get next pos
      q_new = q_new - self.ik_gain * q_dot
    
    return q_new

  def done(self, t, q, env):
    desired_next_pos = self._get_desired_pose_at_time(self.duration)

    env.C.setJointState(q, self.joints)
    [err, jac] = env.C.eval(robotic.FS.pose, [self.ee_name], 1, desired_next_pos)
    
    if np.linalg.norm(err) < 1e-3:
      return True

    return False

class EndEffectorPositionFollowing(BaseDeterministicTimedSkill):
  def __init__(self, joints, ee_name, positions, times=None, dt=0.01, ik_gain=1.0):
    super().__init__(joints, dt=dt)

    self.duration = 1
    self.ik_gain = ik_gain

    self.ee_name = ee_name

    self.positions = np.array(positions)
    num_poses = len(self.positions)
    
    if times is None:
        self.times = np.linspace(0,1,num_poses)
    else:
        self.times = np.array(times)
    
    self.pos_interp = interp1d(self.times, self.positions[:, :3], axis=0, kind='linear')

  def _get_desired_position_at_time(self, t):
    t = np.clip(t, self.times[0], self.times[-1])
    
    # Interpolate position
    p_new = self.pos_interp(t)

    return p_new

  def step(self, t, q, env):
    # look up where we are on the trajctory and get next position
    desired_position = self._get_desired_position_at_time(t)

    q_new = 1. * q
    for _ in range(100):
      env.C.setJointState(q_new, self.joints)
      [err, jac] = env.C.eval(robotic.FS.position, [self.ee_name], 1, desired_position)

      if np.linalg.norm(err) < 1e-3:
        break

      # compute pid law
      q_dot = np.linalg.pinv(jac) @ err

      # integrate to get next pos
      q_new = q_new - self.ik_gain * q_dot

    # env.C.view(False)
    # time.sleep(0.1)

    return q_new

  def done(self, t, q, env):
    desired_position = self._get_desired_position_at_time(self.duration)

    env.C.setJointState(q, self.joints)
    [err, jac] = env.C.eval(robotic.FS.position, [self.ee_name], 1, desired_position)

    if np.linalg.norm(err) < 1e-3:
      return True

    return False

def compute_end_effector_pose(obj_pose, transform):
  p_o = obj_pose[:3]
  q_o = obj_pose[3:]

  p_eo = transform[:3]
  q_eo = transform[3:]

  R_wo = R.from_quat(q_o, scalar_first=True)
  R_eo = R.from_quat(q_eo, scalar_first=True)

  # Invert EE→object
  R_oe = R_eo.inv()
  p_oe = -R_oe.apply(p_eo)

  # Compose: ^wT_e = ^wT_o * ^oT_e
  R_we = R_wo * R_oe
  p_we = p_o + R_wo.apply(p_oe)

  return np.concatenate([p_we, R_we.as_quat(scalar_first=True)])

# cool because it includes multiple robots.
class DualRobotGrasping(BaseDeterministicTimedSkill):
  """Skill for a given object trajectory, where the robots end effectors keep a constant 
  transformation to the object.
  """
  def __init__(self, joints, ee_names, transformations, poses, times=None, dt=0.01, ik_gain=1.0):
    super().__init__(joints, dt=dt)
    
    self.duration = 1
    self.max_num_ik_iters = 100
    self.ik_gain = ik_gain

    self.ee_names = ee_names

    # we assume that ee_pose + transformation == obj_pose
    self.transformation = transformations
    
    self.poses = np.array(poses)
    num_poses = len(self.poses)
    
    if times is None:
        self.times = np.linspace(0,1,num_poses)
    else:
        self.times = np.array(times)
    
    self.pos_interp = interp1d(self.times, self.poses[:, :3], axis=0, kind='linear')

    self.key_rots = R.from_quat(self.poses[:, 3:], scalar_first=True)
    self.slerp = Slerp(self.times, self.key_rots)

  def _get_desired_obj_pose_at_time(self, t):
    t = np.clip(t, self.times[0], self.times[-1])
    
    # Interpolate position
    p_new = self.pos_interp(t)
    
    # Interpolate rotation
    R_t = self.slerp([t])[0]
    q = R_t.as_quat(scalar_first=True)

    return np.concatenate([p_new, q])    
    
  def step(self, t, q, env):
    env.C.setJointState(q, self.joints)
    desired_pose = self._get_desired_obj_pose_at_time(t)
    q_new = q.copy()

    # This implementation is somewhat inefficient/computationally expensive as is
    for i in range(len(self.ee_names)):
      desired_ee_pose = compute_end_effector_pose(desired_pose, self.transformation[i])
      
      for j in range(self.max_num_ik_iters):
        env.C.setJointState(q_new, self.joints)
        [err, jac] = env.C.eval(robotic.FS.pose, [self.ee_names[i]], 1, desired_ee_pose)

        if np.linalg.norm(err) < 1e-3:
          break

        q_dot = np.linalg.pinv(jac) @ err
        q_new = q_new - self.ik_gain * q_dot # dt for rollout (traj discretization), not IK convergence

    # env.C.view(False)
    # time.sleep(0.1)

    return q_new

  def done(self, t, q, env):
    if t > 1.0:
      return True

    return False

# basically the same thing as pose reaching, but with obstacle avoidance
class ModelBasedInsertion(DeterministicBaseSkill):
  def __init__(self, joints, goal, ee_name, dt=0.1):
    super().__init__(joints, dt=dt)
    
    self.goal_pose = goal
    self.ee_name = ee_name

    self.qdot_clip = 0.2

  def step(self, q, env):
    # get jacobian
    env.C.setJointState(q, self.joints)

    ee_pose = env.C.getFrame(self.ee_name).getPose()

    mod_goal_pose = self.goal_pose * 1.
    if np.dot(ee_pose[3:], self.goal_pose[3:]) < 0:
      mod_goal_pose[3:] = -mod_goal_pose[3:]

    [pose_err, pose_jac] = env.C.eval(robotic.FS.pose, [self.ee_name], 1, mod_goal_pose)

    # # TODO (Liam) double check -> idea: use nullspace (avoids fighting between pid and "collision"-corrector) and no need for 100 iterations..
    # # Check for collisions at current state
    # env.C.computeCollisions()
    # [coll_err, coll_jac] = env.C.eval(robotic.FS.accumulatedCollisions, [])
    
    # # Primary task: collision avoidance (want coll_err -> 0)
    # coll_pinv = np.linalg.pinv(coll_jac)
    # coll_q_dot = coll_pinv @ coll_err

    # # Secondary task: pose reaching (in null-space of collision avoidance task)
    # coll_null = np.eye(len(q)) - coll_pinv @ coll_jac
    # pose_jac_projected = pose_jac @ coll_null

    # # Compute pose velocity in nullspace (collision correction already applide)
    # pose_q_dot_null = np.linalg.pinv(pose_jac_projected) @ (pose_err - pose_jac @ coll_q_dot)

    # total_q_dot = coll_q_dot + pose_q_dot_null

    # # integrate to get next pos
    # q_new = q - self.dt * total_q_dot

    # return q_new

    # compute pid law
    pose_q_dot = np.linalg.pinv(pose_jac) @ pose_err

    # integrate to get next pos
    q_new = q - self.dt * pose_q_dot

    # correct position such that we are not in collision
    for _ in range(100):
      env.C.setJointState(q_new, self.joints)
      env.C.computeCollisions()
      [coll_err, coll_jac] = env.C.eval(robotic.FS.accumulatedCollisions, [])

      if np.linalg.norm(coll_err) < 1e-6:
        break
  
      coll_q_dot = np.linalg.pinv(coll_jac) @ coll_err
      q_new = q_new - coll_q_dot

    # env.C.setJointState(q_new, self.joints)
    # env.C.view(True)

    return q_new

  def done(self, q, env):
    # get jacobian
    env.C.setJointState(q, self.joints)
    
    ee_pose = env.C.getFrame(self.ee_name).getPose()
    mod_goal_pose = self.goal_pose * 1.
    if np.dot(ee_pose[3:], self.goal_pose[3:]) < 0:
      mod_goal_pose[3:] = -mod_goal_pose[3:]

    [err, jac] = env.C.eval(robotic.FS.pose, [self.ee_name], 1, mod_goal_pose)

    if np.linalg.norm(err) < 1e-3:
      return True

    return False

class Insertion(StochasticBaseSkill):
  def __init__(self, joints, dt=0.1):
    super().__init__(joints, dt=dt)

  def step(self, q, env):
    # query the policy
    # onnx?
    # decide noise level ourselves?
    pass

  def done(self, q, env):
    raise NotImplementedError

class DexterousGrasping(StochasticBaseSkill):
  def __init__(self, joints, dt=0.1):
    super().__init__(joints, dt=dt)

  def step(self, q, env):
    pass

  def done(self, q, env):
    raise NotImplementedError

class Handover(DeterministicBaseSkill):
  def __init__(self, joints, dt=0.1):
    super().__init__(joints, dt=dt)

  def step(self, q, env):
    pass

  def done(self, q, env):
    raise NotImplementedError

class JogJoint(BaseDeterministicTimedSkill):
  """Skill for simple jogging (=moving a single joint in config space) of a joint
  at a given speed.
  """
  def __init__(self, joints, speed, idx, duration, dt=0.01):
    super().__init__(joints, dt=dt)
    
    self.speed = speed
    self.idx = idx
    self.duration = duration

  def step(self, t, q, env):
    qn = q.copy()
    qn[self.idx] += self.speed * self.dt
    return qn

  def done(self, t, q, env):
    #if t > self.duration:
    #print(t%10)
    if t > 1.0:
      return True

    return False

# Scrwing should actually also go down compared to just joint jogging
# Technically based on sensor/force feedback
class Screw(DeterministicBaseSkill):
  def __init__(self, joints, speed, ee_name, dt=0.1):
    super().__init__(joints, dt=dt)
    
    self.speed = speed
    self.ee_name = ee_name

  def step(self, q, env):
    pass

  def done(self, q, env):
    raise NotImplementedError

# It might be more efficient to precompute/rollout a distribution compared to rolling it out
# in the planning loop.
class PrecomputedSkillDistribution(StochasticBaseSkill):
  """Stoachstic skill with precomputed end-distributions/precomputed trajectory distributions.
  Enables not requiring a learned/scripted function for the rollout.
  """
  def __init__(self, joints, dt=0.1):
    super().__init__(joints, dt=dt)
    
  def step(self, q, env):
    pass

  def done(self, q, env):
    raise NotImplementedError

# this can model bin picking form a bin where we do not care which item we take
# could e.g. be a bin of all the same objects, and we do not care
class StochasticBinPick(StochasticBaseSkill):
  def __init__(self, joints, dt=0.1):
    super().__init__(joints, dt=dt)

  def step(self, q, env):
    pass

  def done(self, q, env):
    raise NotImplementedError
