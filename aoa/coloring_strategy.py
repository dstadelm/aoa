from typing import Protocol

import networkx as nx


class ColoringStrategyProtocol(Protocol):
    def __call__(self, graph: nx.DiGraph): ...


class ColoringStrategies:

    @classmethod
    def relative(cls, graph: nx.DiGraph):
        max_float = find_max_float(graph)
        low_float_threshold = round(max_float / 3)
        medium_float_threshold = round(2 * max_float / 3)
        color_graph(graph, low_float_threshold, medium_float_threshold)

    @classmethod
    def exponential(cls, graph: nx.DiGraph):
        max_float = find_max_float(graph)
        low_float_threshold = round(max_float / 9)
        medium_float_threshold = round(max_float / 3)
        color_graph(graph, low_float_threshold, medium_float_threshold)


def find_max_float(graph: nx.DiGraph):
    max_float = 0
    for _, _, data in graph.out_edges(data=True):
        local_float = data["activity"].total_float
        if local_float > max_float:
            max_float = local_float

    return max_float


def color_graph(graph: nx.DiGraph, low_float_threshold: int, medium_float_threshold: int):
    for _, _, data in graph.out_edges(data=True):
        activity = data["activity"]
        if not activity.critical:
            if activity.total_float < low_float_threshold:
                data["color"] = "red"
            elif activity.total_float < medium_float_threshold:
                data["color"] = "orange"
            else:
                data["color"] = "green"
