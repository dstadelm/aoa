from aoa.model.activity import Activity, ActivityCollection
from aoa.model.network import create_network
from aoa.model.node import Node
from aoa.model.node_dict import NodeDict


def test_one_activity():
    activity = Activity(
        id=1,
    )

    network = create_network(ActivityCollection([activity]))
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

    network = create_network(ActivityCollection(activities))
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

    network = create_network(ActivityCollection(activities))

    # either the node is a start node, then  it should have two activities as outbound activities
    # or it is the end node and has one activity as inbound activity and one dummy activity as inbound activities
    # or it has one activity as inbound activity and no outbound activities
    assert network.node_dict.start_node
    inbound_activity_lengths = [len(node.inbound_activities) for node in network.node_dict.values()]
    assert 0 in inbound_activity_lengths
    assert 1 in inbound_activity_lengths
    assert 2 in inbound_activity_lengths

    for node in network.node_dict.values():
        if not node.inbound_activities:
            assert node.outbound_activities == activities
        elif len(node.inbound_activities) == 1:
            assert node.inbound_activities[0] in activities
            assert len(node.outbound_activities) == 1
            assert node.outbound_activities[0].id < 0
        else:
            assert len(node.inbound_activities) == 2
            assert len([activity for activity in node.inbound_activities if activity.id < 0]) == 1
            assert len([activity for activity in node.inbound_activities if activity.id > 0]) == 1
            assert not node.outbound_activities


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

    network = create_network(ActivityCollection(activities))
    # either the node is a start node, then  it should have all three activities as outbount activities
    # or it is the end node and has one activity as inbound activity and two dummy activities as inbound activities
    # or it has one activity as inbound activity and one dummy activity as outbound activity
    assert network.node_dict.start_node
    inbound_activity_lengths = [len(node.inbound_activities) for node in network.node_dict.values()]
    assert len(inbound_activity_lengths) == 4
    assert 0 in inbound_activity_lengths
    assert inbound_activity_lengths.count(1) == 2
    assert 3 in inbound_activity_lengths

    for node in network.node_dict.values():
        if not node.inbound_activities:
            assert node.outbound_activities == activities
        elif len(node.inbound_activities) == 1:
            assert node.inbound_activities[0] in activities
            assert len(node.outbound_activities) == 1
            assert node.outbound_activities[0].id < 0
        else:
            assert len(node.inbound_activities) == 3
            assert len([activity for activity in node.inbound_activities if activity.id < 0]) == 2
            assert len([activity for activity in node.inbound_activities if activity.id > 0]) == 1
            assert not node.outbound_activities


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

    network = create_network(ActivityCollection(activities))
    nodes: NodeDict = network.node_dict
    assert len(nodes) == 11
    assert nodes[set()].outbound_activities == [
        activities[0],
        activities[1],
        activities[2],
        activities[3],
        activities[4],
    ]

    node_start_dependencies = [str(node) for node in nodes.values()]
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

    network = create_network(ActivityCollection(activities))
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
