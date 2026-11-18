"""Registered RoboBallet welding problems."""
from multi_robot_multi_goal_planning.problems.core.registry import register
from multi_robot_multi_goal_planning.problems.mujoco.envs.roboballet.env import make_roboballet_environment

register([('roboballet.welding', {'kind': 'welding', 'drop_uncovered': True}), ('roboballet.welding_4', {'kind': 'welding', 'robots': 4}), ('roboballet.welding_4_sequence', {'kind': 'welding', 'robots': 4, 'task_order': 'random-sequence'}), ('roboballet.welding_sequence', {'kind': 'welding', 'task_order': 'random-sequence', 'drop_uncovered': True})])(make_roboballet_environment)
