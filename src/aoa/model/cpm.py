from aoa.model.duration import duration as compute_duration
from aoa.model.network import Network
from aoa.model.resources import ResourceCollection


def calculate_cpm(network: Network, resources: ResourceCollection | None = None) -> None:
    if resources is None:
        resources = ResourceCollection([])
    _calculate_free_float(network, resources)


def _calculate_earliest_start(network: Network, resources: ResourceCollection | None = None) -> None:
    """Iterate over all nodes and determines the earliest possible start.

    Nodes are processed in topological order; each activity's ``earliest_start``
    and ``earliest_finish`` fields are populated.
    """
    if resources is None:
        resources = ResourceCollection([])
    for node in network.get_node_list_sorted_by_depth():
        earliest_starts: list[float] = [activity.earliest_finish for activity in node.inbound_activities]
        for activity in node.outbound_activities:
            activity.earliest_start = max(earliest_starts) if earliest_starts else 0
            activity.earliest_finish = activity.earliest_start + compute_duration(activity, resources)


def _calculate_latest_finish(network: Network, resources: ResourceCollection | None = None) -> None:
    """Iterate over all nodes and determines the latest possible finish.

    Populates each activity's ``latest_finish`` and ``latest_start`` fields.
    """
    if resources is None:
        resources = ResourceCollection([])
    _calculate_earliest_start(network, resources)
    nodes_sorted_by_depth = network.get_node_list_sorted_by_depth()
    reversed_nodes = [nodes_sorted_by_depth[i] for i in range(len(nodes_sorted_by_depth) - 1, -1, -1)]

    end_node = reversed_nodes[0]

    for node in reversed_nodes:
        if node.is_end:
            latest_finish = max(
                [activity.earliest_finish for activity in end_node.inbound_activities], default=0
            )
        else:
            latest_finish = min([activity.latest_start for activity in node.outbound_activities])

        for activity in node.inbound_activities:
            activity.latest_finish = latest_finish
            activity.latest_start = latest_finish - compute_duration(activity, resources)


def _calculate_free_float(network: Network, resources: ResourceCollection | None = None) -> None:
    _calculate_latest_finish(network, resources)
    for activity in network.activities.values():
        end_node = network.get_activity_nodes(activity).end_node
        if end_node:
            end_node_earliest_start: float = end_node.earliest_start
            activity.free_float = end_node_earliest_start - activity.earliest_finish
