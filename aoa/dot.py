import networkx as nx
from aoa.coloring_strategy import ColoringStrategyProtocol


def set_dot_attributes(graph: nx.DiGraph, coloring_strategy: ColoringStrategyProtocol):
    set_edge_attributes(graph)
    coloring_strategy(graph)

    # critical_path = find_critical_path(graph)


# def set_node_attributes(graph: nx.DiGraph) -> None:
#     nx.set_node_attributes(graph, values={}, name="label")
#     for node in graph.nodes:
#         node_data = graph.nodes[node]["data"]
#         if has_output_activity(graph, node):
#             graph.nodes[node]["label"] = (
#                 "{"
#                 + str(node)
#                 + "|{ES | "
#                 + str(node_data.earliest_start)
#                 + "}|{LS | "
#                 + str(node_data.latest_start)
#                 + "}}"
#             )


def set_edge_attributes(graph: nx.DiGraph) -> None:
    # nx.set_edge_attributes(graph, values={}, name="label")
    # nx.set_edge_attributes(graph, values={}, name="weight")
    for e in graph.edges:
        edge = graph.edges[e]
        activity = edge["activity"]
        if activity.duration == 0 and activity.effort == 0:
            edge["style"] = "dashed"
        else:
            # {activity.description}
            # edge["label"] = (
            #     f"""<<table border="0" align="left">
            #     <tr>
            #         <td align="left"  >ES:{activity.earliest_start}</td>
            #         <td align="center">D:{activity.duration}</td>
            #         <td align="right" >EF:{activity.earliest_finish}</td>
            #     </tr>
            #     <tr>
            #         <td colspan="3">[{activity.id}] {activity.description}</td>
            #     </tr>
            #     <tr>
            #         <td align="left"  >LS:{activity.latest_start}</td>
            #         <td align="center">TF:{activity.total_float}</td>
            #         <td align="right" >LF:{activity.latest_finish}</td>
            #     </tr>
            #     </table>>"""
            # )
            edge["label"] = (
                f"""[{activity.id}] {activity.description}
                    ES:{activity.earliest_start} / EF:{activity.earliest_finish} / LS:{activity.latest_start} / LF:{activity.latest_finish} / D:{activity.duration} / TF:{activity.total_float}"""
            )
            # edge["labeltooltip"] = (
            #     f"""<<table border=0><tr><td>Earliest Start</td><td>{activity.earliest_start}</td></tr><tr><td>Earliest Finish</td><td>{activity.earliest_finish}</td></tr><tr><td>Latest Sart</td><td>{activity.latest_start}</td></tr><tr><td>Latest Finish</td><td>{activity.latest_finish}</td></tr><tr><td>Duration</td><td>{activity.duration}</td></tr><tr><td>Total Float</td><td>{activity.total_float}</td></tr></table>>"""
            # )
            edge["edgetooltip"] = (
                f"""<<table border="0" align="left">
                <tr>
                    <td align="left"  >ES:{activity.earliest_start}</td>
                    <td align="center">D:{activity.duration}</td>
                    <td align="right" >EF:{activity.earliest_finish}</td>
                </tr>
                <tr>
                    <td colspan="3">[{activity.id}] {activity.description}</td>
                </tr>
                <tr>
                    <td align="left"  >LS:{activity.latest_start}</td>
                    <td align="center">TF:{activity.total_float}</td>
                    <td align="right" >LF:{activity.latest_finish}</td>
                </tr>
                </table>>"""
            )

            # edge["decorate"] = "true"

        if activity.critical:
            # edge["style"] = "bold"
            # edge["color"] = "red"
            edge["penwidth"] = "4.0"


def has_output_activity(graph: nx.DiGraph, node: str) -> bool:
    for _, _, data in graph.out_edges(node, data=True):
        activity = data["activity"]
        if activity.effort > 0:
            return True
    return False
