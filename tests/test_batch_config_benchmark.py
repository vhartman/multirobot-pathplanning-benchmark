import numpy as np
from multi_robot_multi_goal_planning.problems.configuration import (
    NpConfiguration,
    batch_config_cost,
)
from typing import List

import pytest


def generate_test_data(dims: List, num_pts=1000):
    cumulative_dimension = sum(dims)
    slices = [(sum(dims[:i]), sum(dims[: i + 1])) for i in range(len(dims))]
    pt = np.random.rand(cumulative_dimension)
    single_config = NpConfiguration(pt, slices)

    pts = np.random.rand(
        num_pts, cumulative_dimension
    )  # adjust size based on typical usage
    return single_config, pts


@pytest.mark.parametrize("reduction", ["max", "sum"])
@pytest.mark.parametrize("dims", [[2, 2], [7, 7], [3, 3, 3], [14]])
@pytest.mark.parametrize("num_points", [10, 100, 1000, 5000])
def test_batch_config_cost_benchmark(benchmark, dims, num_points, reduction):
    config, pts = generate_test_data(dims, num_pts=num_points)

    def fn(pt, pts):
        return batch_config_cost(
            pt, pts, metric="euclidean", reduction=reduction
        )
    benchmark(fn, config, pts)
