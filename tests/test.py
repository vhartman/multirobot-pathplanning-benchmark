import pytest

from multi_robot_multi_goal_planning.problems.util import generate_binary_search_indices


@pytest.mark.parametrize("n, expected", [
    (1, (0,)),
    (2, (0, 1)),
    (3, (1, 0, 2)),
    (4, (1, 0, 2, 3)),
    (5, (2, 0, 3, 1, 4)),
])
def test_binary_indices(n, expected):
    assert generate_binary_search_indices(n) == expected