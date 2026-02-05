import pytest

from aoa.model.activity import Activity
from aoa.model.network import AllocationException, Network, NonUniqueIdException
from aoa.model.node import Node


def test_one_activity():
    activity = Activity(
        id=1,
    )

    network = Network([activity])
    nodes: list[Node] = network.get_node_list_sorted_by_depth()
    assert len(nodes) == 2  # start and end nodes
    assert nodes[0].outbound_activities == [activity]
    assert nodes[1].inbound_activities == [activity]


def test_two_sequential_activities():
    activities = [
        Activity(
            id=1,
        ),
        Activity(
            id=2,
            predecessors=set([1]),
        ),
    ]

    network = Network(activities)
    nodes: list[Node] = network.get_node_list_sorted_by_depth()
    assert len(nodes) == 3  # start and end nodes
    assert nodes[0].outbound_activities == [activities[0]]
    assert nodes[1].inbound_activities == [activities[0]]
    assert nodes[1].outbound_activities == [activities[1]]
    assert nodes[2].inbound_activities == [activities[1]]


def test_two_parallel_activities():
    activities = [
        Activity(
            id=1,
        ),
        Activity(
            id=2,
        ),
    ]

    network = Network(activities)
    nodes: list[Node] = network.get_node_list_sorted_by_depth()
    assert len(nodes) == 3  # start and end nodes
    assert nodes[0].outbound_activities == [activities[0], activities[1]]

    activity_node_1_id = [activity.id for activity in nodes[1].inbound_activities if activity.id > 0][0]
    activity_node_2_id = [activity.id for activity in nodes[2].inbound_activities if activity.id > 0][0]
    assert activity_node_1_id == 3 - activity_node_2_id

    if nodes[1].outbound_activities:
        node_2_inbound_dummy_ids = [activity.id for activity in nodes[2].inbound_activities if activity.id < 0]
        assert nodes[1].outbound_activities[0].id in node_2_inbound_dummy_ids

    if nodes[2].outbound_activities:
        node_1_inbound_dummy_ids = [activity.id for activity in nodes[1].inbound_activities if activity.id < 0]
        assert nodes[2].outbound_activities[0].id in node_1_inbound_dummy_ids


def test_three_parallel_activities():
    """Test a network with three parallel activities."""
    activities = [
        Activity(
            id=1,
        ),
        Activity(
            id=2,
        ),
        Activity(
            id=3,
        ),
    ]

    network = Network(activities)
    nodes: list[Node] = network.get_node_list_sorted_by_depth()
    assert len(nodes) == 4  # start and end nodes
    assert nodes[0].outbound_activities == [activities[0], activities[1], activities[2]]
    assert nodes[1].inbound_activities[0].id == 2
    assert nodes[1].outbound_activities[0].id == -1
    assert nodes[2].inbound_activities[0].id == 3
    assert nodes[2].outbound_activities[0].id == -2
    assert nodes[3].inbound_activities[0].id == 1
    assert nodes[3].inbound_activities[1].id == -1
    assert nodes[3].inbound_activities[2].id == -2


def test_complex_network():
    """Test a complex network with multiple dependencies."""
    activities = [
        Activity(id=1),
        Activity(id=2),
        Activity(id=3),
        Activity(id=4),
        Activity(id=5),
        Activity(id=6, predecessors={1, 2}),
        Activity(id=7, predecessors={2, 3}),
        Activity(id=8, predecessors={3, 4}),
        Activity(id=9, predecessors={4, 5}),
        Activity(id=10, predecessors={1, 2, 3}),
        Activity(id=11, predecessors={1, 2, 3, 4, 5}),
    ]

    network = Network(activities)
    nodes: list[Node] = network.get_node_list_sorted_by_depth()
    assert len(nodes) == 11
    assert nodes[0].outbound_activities == [activities[0], activities[1], activities[2], activities[3], activities[4]]

    node_start_dependencies = [str(node) for node in nodes]
    assert "start" in node_start_dependencies
    assert "2" in node_start_dependencies
    assert "3" in node_start_dependencies
    assert "4" in node_start_dependencies
    assert "1-2" in node_start_dependencies
    assert "2-3" in node_start_dependencies
    assert "3-4" in node_start_dependencies
    assert "4-5" in node_start_dependencies
    assert "1-2-3" in node_start_dependencies
    assert "1-2-3-4-5" in node_start_dependencies
    assert "end" in node_start_dependencies


def test_multiple_end_nodes() -> None:

    activities = [
        Activity(
            id=1,
            effort=10,
        ),
        Activity(
            id=2,
            effort=1,
        ),
        Activity(
            id=3,
            effort=1,
            predecessors={2},
        ),
        Activity(
            id=4,
            effort=1,
            predecessors={3},
        ),
        Activity(
            id=5,
            effort=1,
            predecessors={2},
        ),
    ]

    network = Network(activities)
    nodes: list[Node] = network.get_node_list_sorted_by_depth()
    assert len(nodes) == 4

    def stdp_2_str(node: Node) -> str:
        if not node.start_dependencies:
            return "start"
        return "-".join(str(dep) for dep in sorted(node.start_dependencies))

    node_start_dependencies = [stdp_2_str(node) for node in nodes]
    assert "start" in node_start_dependencies
    assert "2" in node_start_dependencies
    assert "3" in node_start_dependencies
    assert "1-4-5" in node_start_dependencies


def test_overconstraining() -> None:

    activities = [
        Activity(
            id=1,
            effort=10,
        ),
        Activity(
            id=2,
            effort=1,
        ),
        Activity(
            id=3,
            effort=1,
            predecessors={2},
        ),
        Activity(
            id=4,
            effort=1,
            predecessors={3},
        ),
        Activity(
            id=5,
            effort=1,
            predecessors={2, 3},
        ),
    ]

    with pytest.raises(AllocationException):
        _ = Network(activities)


def test_validate_unique_activity_ids() -> None:
    activities = [
        Activity(id=1),
        Activity(id=1),
    ]

    with pytest.raises(NonUniqueIdException, match=r"Duplicate activity ID\[1] found"):
        _ = Network(activities)
