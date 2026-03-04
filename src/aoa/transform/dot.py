# typing: ignore
import networkx as nx
import pygraphviz as pgvz

from aoa.model.activity import Activity
from aoa.model.node import Node

from .coloring_strategy import ColoringStrategyProtocol


def create_dot(graph: nx.DiGraph, coloring_strategy: ColoringStrategyProtocol) -> pgvz.AGraph:
    set_dot_attributes(graph, coloring_strategy)
    gvz: pgvz.AGraph = nx.nx_agraph.to_agraph(graph)
    # gvz.graph_attr["splines"] = "curved"
    # gvz.graph_attr["splines"] = "polyline"
    # gvz.graph_attr["splines"] = "ortho"
    # gvz.graph_attr["splines"] = "spline"
    # gvz.graph_attr["splines"] = "line"
    gvz = rank_dot_nodes(graph, gvz)
    gvz.graph_attr["rankdir"] = "TB"
    gvz.layout(prog="dot", args="-Nshape=Mrecord")
    print(gvz)
    return gvz


def rank_dot_nodes(graph: nx.DiGraph, gvz: pgvz.AGraph) -> pgvz.AGraph:
    rank_dict: dict[int, list[int]] = dict()
    for node_name in graph.nodes:
        print(f"Processing node {node_name} for ranking...")
        node: Node = graph.nodes[node_name]["data"]
        rank_dict.setdefault(node.max_depth, []).append(node_name)

    for key, value in rank_dict.items():
        subgraph = gvz.add_subgraph(value, name=str(key))
        subgraph.graph_attr["rank"] = "same"

    return gvz


def set_dot_attributes(graph: nx.DiGraph, coloring_strategy: ColoringStrategyProtocol):
    set_edge_attributes(graph)
    coloring_strategy(graph)


def set_edge_attributes(graph: nx.DiGraph) -> None:
    # nx.set_edge_attributes(graph, values={}, name="label")
    # nx.set_edge_attributes(graph, values={}, name="weight")
    for e in graph.edges:
        edge = graph.edges[e]
        activity: Activity = edge["activity"]
        if activity.is_dummy:
            edge["style"] = "dashed"
        else:
            edge["label"] = f"""[{activity.id}] {activity.activity}"""
            edge[
                "edgetooltip"
            ] = f"""
ES:{activity.earliest_start}
D:{activity.duration}
EF:{activity.earliest_finish}
LS:{activity.latest_start}
TF:{activity.total_float}
LF:{activity.latest_finish}
                """

            edge["decorate"] = "false"

        if activity.critical:
            edge["penwidth"] = "5.0"
        else:
            edge["penwidth"] = "2.0"
