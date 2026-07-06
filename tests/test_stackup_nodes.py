"""Equivalence tests for IterationData._stackup_nodes.

The optimized sort + stack implementation must reproduce the original
O(n^2) containment-scan behaviour exactly — including its handling of
nodes with identical (start, end) intervals, which end up mutually
parented and therefore excluded from the returned roots. The original
algorithm is kept here as the reference oracle.
"""
import random

import pytest

from perf_estimator.profiler.analyser import IterationData
from perf_estimator.profiler.node import OperatorNode


def make_node(start, dur):
    return OperatorNode(
        {
            "ph": "X",
            "cat": "cpu_op",
            "name": "aten::test",
            "pid": 0,
            "tid": 0,
            "ts": start,
            "dur": dur,
            "args": {},
        }
    )


def oracle_stackup(nodes):
    """The original O(n^2) implementation, kept as the reference."""
    root_stacks = []
    for node in nodes:
        matched = list(
            filter(
                lambda x: x.start_time <= node.start_time
                and x.end_time >= node.end_time,
                nodes,
            )
        )
        matched.remove(node)
        if len(matched) == 0:
            root_stacks.append(node)
        else:
            matched.sort(key=lambda x: (x.start_time, -x.end_time))
            matched[-1].add_child(node)
    return root_stacks


def snapshot(nodes, roots):
    parents = {
        id(n): (id(n.parent) if n.parent is not None else None) for n in nodes
    }
    return parents, {id(r) for r in roots}


def assert_equivalent(intervals):
    iteration = IterationData.__new__(IterationData)  # no init needed

    nodes_a = [make_node(s, d) for s, d in intervals]
    oracle_snap = snapshot(nodes_a, oracle_stackup(nodes_a))

    nodes_b = [make_node(s, d) for s, d in intervals]
    new_snap = snapshot(nodes_b, IterationData._stackup_nodes(iteration, nodes_b))

    # compare shapes via index-mapped ids since the node objects differ
    remap = dict(zip(map(id, nodes_a), map(id, nodes_b)))
    remap[None] = None
    assert {remap[k]: remap[v] for k, v in oracle_snap[0].items()} == new_snap[0]
    assert {remap[r] for r in oracle_snap[1]} == new_snap[1]


def test_empty():
    iteration = IterationData.__new__(IterationData)
    assert IterationData._stackup_nodes(iteration, []) == []


def test_simple_nesting():
    assert_equivalent([(0, 100), (10, 20), (12, 5), (50, 30)])


def test_siblings_and_roots():
    assert_equivalent([(0, 10), (20, 10), (40, 10)])


def test_shared_start_and_end():
    # same start, different ends / same end, different starts
    assert_equivalent([(0, 100), (0, 50), (60, 40), (0, 100 - 1)])


def test_identical_interval_twins_leave_roots():
    # Twins mutually contain each other; the original drops them (and
    # everything nested strictly inside) from the returned roots.
    intervals = [(0, 10), (0, 10), (2, 3)]
    iteration = IterationData.__new__(IterationData)
    nodes = [make_node(s, d) for s, d in intervals]
    roots = IterationData._stackup_nodes(iteration, nodes)
    assert roots == []
    assert_equivalent(intervals)


def test_zero_duration_ops():
    assert_equivalent([(0, 50), (10, 0), (10, 0), (30, 0)])


@pytest.mark.parametrize("seed", range(20))
def test_random_intervals_match_oracle(seed):
    rng = random.Random(seed)
    intervals = []
    for _ in range(rng.randint(1, 120)):
        start = rng.randint(0, 500)
        dur = rng.randint(0, 60)
        intervals.append((start, dur))
    # inject a few exact duplicates to exercise the twin path
    for _ in range(rng.randint(0, 5)):
        intervals.append(rng.choice(intervals))
    rng.shuffle(intervals)
    assert_equivalent(intervals)
