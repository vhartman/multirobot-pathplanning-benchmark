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
@pytest.mark.parametrize("dims", [[2, 2], [7, 7], [3, 3, 3], [2, 5], [14]])
@pytest.mark.parametrize("num_points", [10, 100, 1000, 5000])
def test_batch_config_cost_zero_dist(reduction, dims, num_points):
    cumulative_dimension = sum(dims)
    slices = [(sum(dims[:i]), sum(dims[: i + 1])) for i in range(len(dims))]
    pt = np.zeros(cumulative_dimension)
    single_config = NpConfiguration(pt, slices)

    _, pts = generate_test_data(dims, num_pts=num_points)
    pts = pts * 0

    dists = batch_config_cost(single_config, pts, reduction=reduction)

    assert all(dists == 0)


@pytest.mark.parametrize("reduction", ["max", "sum"])
@pytest.mark.parametrize("dims", [[2, 2], [7, 7], [3, 3, 3], [2, 5], [14]])
@pytest.mark.parametrize("num_points", [10, 100, 1000, 5000])
def test_batch_config_cost_equal_pts(reduction, dims, num_points):
    cumulative_dimension = sum(dims)
    slices = [(sum(dims[:i]), sum(dims[: i + 1])) for i in range(len(dims))]
    pt = np.random.rand(cumulative_dimension)
    single_config = NpConfiguration(pt, slices)

    _, pts = generate_test_data(dims, num_pts=num_points)
    pts[:, :] = pt

    dists = batch_config_cost(single_config, pts, reduction=reduction)

    assert all(dists == 0)


@pytest.mark.parametrize("reduction", ["max", "sum"])
@pytest.mark.parametrize("dims", [[2, 2], [7, 7], [3, 3, 3], [2, 5], [14]])
@pytest.mark.parametrize("num_points", [10, 100, 1000, 5000])
def test_batch_config_cost_translation(reduction, dims, num_points):
    single_config, pts = generate_test_data(dims, num_pts=num_points)

    offset = 1
    pts_offset = pts + offset
    single_config_offset = NpConfiguration(
        single_config.state() + offset, single_config.array_slice
    )

    dists = batch_config_cost(single_config, pts, reduction=reduction)
    dists_translated = batch_config_cost(
        single_config_offset, pts_offset, reduction=reduction
    )

    assert np.allclose(dists, dists_translated)


@pytest.mark.parametrize("dims", [[2], [6], [14]])
@pytest.mark.parametrize("num_points", [10, 100, 1000, 5000])
def test_batch_config_cost_single_agent(dims, num_points):
    single_config, pts = generate_test_data(dims, num_pts=num_points)

    dists_max = batch_config_cost(single_config, pts, reduction="max", w=0.0)
    dists_sum = batch_config_cost(single_config, pts, reduction="sum")

    assert np.allclose(dists_sum, dists_max)


@pytest.mark.parametrize("dims", [[2, 2], [7, 7], [3, 3, 3], [2, 5], [14]])
def test_batch_config_cost_euclidean_manual_comparison(dims):
    single_config, pts = generate_test_data(dims, num_pts=1)
    dist_from_function = batch_config_cost(
        single_config, pts, reduction="sum", metric="euclidean"
    )

    manual_dist = 0
    slice = single_config.array_slice
    for i in range(len(dims)):
        single_agent_dist = np.linalg.norm(
            pts[0][slice[i][0] : slice[i][1]]
            - single_config.state()[slice[i][0] : slice[i][1]]
        )
        manual_dist += single_agent_dist

    assert dist_from_function[0] == manual_dist


@pytest.mark.parametrize("dims", [[2, 2], [7, 7], [3, 3, 3], [2, 5], [14]])
def test_batch_config_cost_max_manual_comparison(dims):
    single_config, pts = generate_test_data(dims, num_pts=1)
    dist_from_function = batch_config_cost(
        single_config, pts, reduction="sum", metric="max"
    )

    manual_dist = 0
    slice = single_config.array_slice
    for i in range(len(dims)):
        single_agent_dist = np.max(
            np.abs(
                pts[0][slice[i][0] : slice[i][1]]
                - single_config.state()[slice[i][0] : slice[i][1]]
            )
        )
        manual_dist += single_agent_dist

    assert dist_from_function[0] == manual_dist
