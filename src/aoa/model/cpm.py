from dataclasses import dataclass, field

from aoa.model.network import Network


def calculate_cpm(network: Network) -> None:
    _calculate_free_float(network)


def _calculate_earliest_start(network: Network) -> None:
    """Iterate over all nodes and determines the earliest possible start.

    The nodes are updated with the according earliest possible start value.

    Attributes:
        network(Network): The network to calculate the earliest start for

    """
    for node in network.get_node_list_sorted_by_depth():
        earliest_starts: list[float] = [
            activity.earliest_start + activity.duration for activity in node.inbound_activities
        ]
        for activity in node.outbound_activities:
            activity.earliest_start = max(earliest_starts) if earliest_starts else 0


def _calculate_latest_finish(network: Network) -> None:
    """Iterate over all nodes and determines the latest possible start.

    The nodes are updated with the according latest possible finish value.

    Attributes:
        network(Network): The network to calculate the latest finish for


    """
    _calculate_earliest_start(network)
    nodes_sorted_by_depth = network.get_node_list_sorted_by_depth()
    reversed_nodes = [nodes_sorted_by_depth[i] for i in range(len(nodes_sorted_by_depth) - 1, -1, -1)]

    end_node = reversed_nodes[0]

    for node in reversed_nodes:
        if node.is_end:
            latest_finish = max(
                [activity.earliest_start + activity.duration for activity in end_node.inbound_activities], default=0
            )
        else:
            latest_finish = min([activity.latest_finish - activity.duration for activity in node.outbound_activities])

        for activity in node.inbound_activities:
            activity.latest_finish = latest_finish


def _calculate_free_float(network: Network) -> None:
    _calculate_latest_finish(network)
    for activity in network.activities.values():
        end_node = network.get_activity_nodes(activity).end_node
        if end_node:
            end_node_earliest_start: float = end_node.earliest_start
            activity.free_float = end_node_earliest_start - activity.earliest_finish
