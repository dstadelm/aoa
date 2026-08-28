# typing: ignore
from dataclasses import dataclass, field

import networkx as nx
import pygraphviz as pgvz

from aoa.model.activity import Activity
from aoa.model.node import Node

from .coloring_strategy import ColoringStrategyProtocol
from .theme import DEFAULT_THEME_NAME, THEMES, Theme


@dataclass
class DotFormating:
    critical: dict[str, str] = field(default_factory=lambda: {"color": "red", "penwidth": "5.0"})
    medium_float: dict[str, str] = field(default_factory=lambda: {"color": "orange", "penwidth": "2.0"})
    high_float: dict[str, str] = field(default_factory=lambda: {"color": "green", "penwidth": "2.0"})
    dummy: dict[str, str] = field(default_factory=lambda: {"style": "dashed", "color": "black", "penwidth": "2.0"})


def create_dot(
    graph: nx.DiGraph,
    coloring_strategy: ColoringStrategyProtocol,
    theme: Theme = THEMES[DEFAULT_THEME_NAME],
    rankdir: str = "TB",
) -> pgvz.AGraph:
    set_dot_attributes(graph, coloring_strategy, theme)
    gvz: pgvz.AGraph = nx.nx_agraph.to_agraph(graph)
    gvz = rank_dot_nodes(graph, gvz)
    gvz.graph_attr["rankdir"] = rankdir
    gvz.graph_attr["bgcolor"] = "transparent"
    gvz.node_attr["style"] = "filled"
    gvz.node_attr["shape"] = "circle"
    gvz.node_attr["label"] = ""
    gvz.node_attr["fillcolor"] = theme.node_fill
    gvz.node_attr["color"] = theme.node_stroke
    gvz.node_attr["fontcolor"] = theme.text
    gvz.edge_attr["color"] = theme.edge
    gvz.edge_attr["fontcolor"] = theme.text_muted
    gvz.layout(prog="dot")
    return gvz


def rank_dot_nodes(graph: nx.DiGraph, gvz: pgvz.AGraph) -> pgvz.AGraph:
    rank_dict: dict[int, list[int]] = dict()
    for node_name in graph.nodes:
        node: Node = graph.nodes[node_name]["data"]
        rank_dict.setdefault(node.max_depth, []).append(node_name)

    for key, value in rank_dict.items():
        subgraph = gvz.add_subgraph(value, name=str(key))
        subgraph.graph_attr["rank"] = "same"

    return gvz


def set_dot_attributes(
    graph: nx.DiGraph,
    coloring_strategy: ColoringStrategyProtocol,
    theme: Theme,
):
    coloring_strategy(graph)
    set_edge_attributes(graph, theme)


def _token_to_color(token: str, theme: Theme) -> str:
    return {
        "critical": theme.critical,
        "red": theme.red,
        "orange": theme.orange,
        "green": theme.green,
        "edge": theme.edge,
    }.get(token, theme.edge)


def set_edge_attributes(graph: nx.DiGraph, theme: Theme) -> None:
    for e in graph.edges:
        edge = graph.edges[e]
        activity: Activity = edge["activity"]
        if activity.is_dummy:
            edge["style"] = "dashed"
            edge["color"] = theme.edge
        else:
            edge["label"] = activity.activity
            edge["edgetooltip"] = f"""[{activity.id}] {activity.activity}
ES:{activity.earliest_start}
D:{activity.planned_effort}
EF:{activity.earliest_finish}
LS:{activity.latest_start}
TF:{activity.total_float}
LF:{activity.latest_finish}
                """

            edge["decorate"] = "false"

            token = edge.get("color_token")
            if token:
                edge["color"] = _token_to_color(token, theme)

        if activity.critical:
            edge["penwidth"] = "5.0"
            edge["color"] = theme.critical
        else:
            edge["penwidth"] = "2.0"

        penwidth = float(edge["penwidth"])
        edge["arrowsize"] = f"{2.0 / penwidth:.3f}"
