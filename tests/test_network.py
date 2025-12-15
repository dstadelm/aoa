from aoa.activity import Activity
from aoa.network import Network
from aoa.node import Node


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
    assert nodes[1].inbound_activities[0].id == 2
    assert nodes[1].outbound_activities[0].id == -1
    assert nodes[2].inbound_activities[0].id == 1
    assert nodes[2].inbound_activities[1].id == -1


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
