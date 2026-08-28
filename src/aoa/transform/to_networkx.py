import logging

import networkx as nx

from aoa.model.network import Network

logger = logging.getLogger(__name__)


def to_networkx(network: Network) -> nx.DiGraph:
    G: nx.DiGraph = nx.DiGraph()

    for node in network.get_node_list_sorted_by_depth():
        logger.debug("Adding node %s to NetworkX graph", node.id)
        G.add_node(node.id, data=node)
    for activity in network.activities.values():
        start_node = network.get_activity_start_node(activity)
        end_node = network.get_activity_end_node(activity)
        logger.debug(
            "Adding edge from node %s to node %s for activity %s", start_node.id, end_node.id, activity.activity
        )
        G.add_edge(start_node.id, end_node.id, activity=activity)
    return G
