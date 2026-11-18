"""Registered cuArms-inspired packing and unpacking problems."""
from multi_robot_multi_goal_planning.problems.core.registry import register
from multi_robot_multi_goal_planning.problems.mujoco.envs.roboballet.env import make_roboballet_environment

register([('cuarms.pack', {'kind': 'pack'}), ('cuarms.unpack', {'kind': 'unpack'}), ('cuarms.pack_sequence', {'kind': 'pack', 'task_order': 'random-sequence'}), ('cuarms.unpack_sequence', {'kind': 'unpack', 'task_order': 'random-sequence'})])(make_roboballet_environment)
