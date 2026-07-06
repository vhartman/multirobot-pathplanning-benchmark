import numpy as np
from multi_robot_multi_goal_planning.problems.core.configuration import (
    NpConfiguration,
    batch_config_cost,
    batch_config_dist,
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
@pytest.mark.parametrize("num_points", [1, 10, 100, 1000, 5000])
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
@pytest.mark.parametrize("num_points", [1, 10, 100, 1000, 5000])
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
@pytest.mark.parametrize("num_points", [1, 10, 100, 1000, 5000])
def test_batch_config_cost_translation(reduction, dims, num_points):
    single_config, pts = generate_test_data(dims, num_pts=num_points)

    offset = 1
    pts_offset = pts + offset
    single_config_offset = single_config.from_flat(
        single_config.state() + offset
    )

    dists = batch_config_cost(single_config, pts, reduction=reduction)
    dists_translated = batch_config_cost(
        single_config_offset, pts_offset, reduction=reduction
    )

    assert np.allclose(dists, dists_translated)


@pytest.mark.parametrize("dims", [[2], [6], [14]])
@pytest.mark.parametrize("num_points", [1, 10, 100, 1000, 5000])
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
    slice = single_config._array_slice
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
    slice = single_config._array_slice
    for i in range(len(dims)):
        single_agent_dist = np.max(
            np.abs(
                pts[0][slice[i][0] : slice[i][1]]
                - single_config.state()[slice[i][0] : slice[i][1]]
            )
        )
        manual_dist += single_agent_dist

    assert dist_from_function[0] == manual_dist


@pytest.mark.parametrize(
    "metric", ["max", "euclidean", "sum_euclidean", "max_euclidean"]
)
@pytest.mark.parametrize("dims", [[2, 2], [7, 7], [3, 3, 3], [2, 5], [14]])
@pytest.mark.parametrize("num_points", [1, 10, 100, 1000, 5000])
def test_batch_config_dist_zero_dist(metric, dims, num_points):
    cumulative_dimension = sum(dims)
    slices = [(sum(dims[:i]), sum(dims[: i + 1])) for i in range(len(dims))]
    pt = np.zeros(cumulative_dimension)
    single_config = NpConfiguration(pt, slices)

    _, pts = generate_test_data(dims, num_pts=num_points)
    pts = pts * 0

    dists = batch_config_dist(single_config, pts, metric=metric)

    assert all(dists == 0)


@pytest.mark.parametrize(
    "metric", ["max", "euclidean", "sum_euclidean", "max_euclidean"]
)
@pytest.mark.parametrize("dims", [[2, 2], [7, 7], [3, 3, 3], [2, 5], [14]])
@pytest.mark.parametrize("num_points", [1, 10, 100, 1000, 5000])
def test_batch_config_dist_equal_pts(metric, dims, num_points):
    cumulative_dimension = sum(dims)
    slices = [(sum(dims[:i]), sum(dims[: i + 1])) for i in range(len(dims))]
    pt = np.random.rand(cumulative_dimension)
    single_config = NpConfiguration(pt, slices)

    _, pts = generate_test_data(dims, num_pts=num_points)
    pts[:, :] = pt

    dists = batch_config_dist(single_config, pts, metric=metric)

    assert all(dists == 0)

@pytest.mark.parametrize(
    "metric", ["max", "euclidean", "sum_euclidean", "max_euclidean"]
)
@pytest.mark.parametrize("dims", [[2, 2], [7, 7], [3, 3, 3], [2, 5], [14]])
@pytest.mark.parametrize("num_points", [1, 10, 100, 1000, 5000])
def test_batch_config_dist_translation(metric, dims, num_points):
    single_config, pts = generate_test_data(dims, num_pts=num_points)

    offset = 1
    pts_offset = pts + offset
    single_config_offset = single_config.from_flat(
        single_config.state() + offset
    )

    dists = batch_config_dist(single_config, pts, metric=metric)
    dists_translated = batch_config_dist(
        single_config_offset, pts_offset, metric=metric
    )

    assert np.allclose(dists, dists_translated)


# ---------------------------------------------------------------------------
# from_flat sharing / immutability invariants
#
# from_flat deliberately SHARES the parent's slice metadata (it describes the
# env's fixed robot layout) and, for float64 input, ALIASES the caller's array
# instead of copying ("hand over ownership"). These tests pin the invariants
# that make that safe; if one of them fails after a refactor, the sharing has
# become a correctness bug and from_flat must go back to copying.
# ---------------------------------------------------------------------------


def _two_robot_config():
    return NpConfiguration.from_list(
        [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
    )


def test_from_flat_shares_slice_metadata():
    c = _two_robot_config()
    child = c.from_flat(np.arange(5, dtype=np.float64))
    grandchild = child.from_flat(np.arange(5, dtype=np.float64) + 1)
    # sharing is intentional: one layout object per lineage
    assert child._array_slice is c._array_slice
    assert grandchild._array_slice is c._array_slice
    assert child._slices is c._slices


def test_shared_slice_metadata_is_immutable():
    c = _two_robot_config()
    child = c.from_flat(np.arange(5, dtype=np.float64))
    # a write through ANY alias must raise, not silently corrupt the lineage
    with pytest.raises(ValueError):
        child._array_slice[0, 0] = 99
    with pytest.raises(ValueError):
        c._array_slice[0, 0] = 99
    with pytest.raises(TypeError):
        child._slices[0] = (0, 99)


def test_from_flat_q_ownership():
    c = _two_robot_config()
    q64 = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    child = c.from_flat(q64)
    # float64 input is aliased, not copied (the documented ownership hand-over) --
    # if this ever changes, from_flat got slower; if callers rely on mutating the
    # array after the call, they are broken NOW rather than subtly later
    assert child.q is q64
    # non-float64 input must be converted (and therefore copied)
    q_int = np.array([9, 8, 7, 6, 5])
    child_int = c.from_flat(q_int)
    assert child_int.q is not q_int
    assert child_int.q.dtype == np.float64


def test_from_flat_child_values_independent_of_parent():
    c = _two_robot_config()
    child = c.from_flat(np.array([9.0, 8.0, 7.0, 6.0, 5.0]))
    # distinct q buffers: writing one never leaks into the other
    child[0] = np.array([-1.0, -2.0, -3.0])
    assert np.array_equal(c.robot_state(0), [1.0, 2.0, 3.0])
    c[1] = np.array([40.0, 50.0])
    assert np.array_equal(child.robot_state(1), [6.0, 5.0])
    # robot_state slicing correct on the child
    assert np.array_equal(child.robot_state(0), [-1.0, -2.0, -3.0])


def test_root_configs_do_not_share_metadata():
    a = _two_robot_config()
    b = _two_robot_config()
    # sharing only propagates along from_flat lineages, never across roots
    assert a._array_slice is not b._array_slice
