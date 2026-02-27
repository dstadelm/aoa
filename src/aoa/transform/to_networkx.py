import networkx as nx

from aoa.model.network import Network


def to_networkx(network: Network) -> nx.DiGraph:
    G: nx.DiGraph = nx.DiGraph()
    # print("Creating NetworkX graph from AOA network...")

    for node in network.get_node_list_sorted_by_depth():
        # print(f"Adding node {node.id} to NetworkX graph...")
        G.add_node(node.id, data=node)
    for activity in network.activities:
        start_node = network.get_activity_start_node(activity)
        end_node = network.get_activity_end_node(activity)
        # print(f"Adding edge from node {start_node.id} to node {end_node.id} for activity {activity.activity}...")
        G.add_edge(start_node.id, end_node.id, activity=activity)
    return G
