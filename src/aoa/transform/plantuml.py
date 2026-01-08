from pathlib import Path

from aoa.model.activity import Activity, DummyActivity
from aoa.model.network import Network
from aoa.model.node import Node


class PlantUml:
    def __init__(self, network: Network):
        self.plantuml: str = ""
        self.sorted_nodes: list[Node] = network.get_node_list_sorted_by_depth()
        self.network: Network = network

    def get_txt(self) -> str:
        return self._get_header() + self._get_map() + "\n" + self._get_network() + self._get_trailer()

    def write_txt(self, file: Path) -> None:
        with open(file, "w") as f:
            _ = f.write(self.get_txt())

    def _get_header(self) -> str:
        return """@startuml PERT
top to bottom direction
' Horizontal lines: -->, <--, <-->
' Vertical lines: ->, <-, <->
title Pert: Project Design

"""

    def _get_trailer(self) -> str:
        return "\n@enduml"

    def _get_map(self) -> str:
        map_list = [
            f"""map {node.id} {{
    earliest start => {node.earliest_start}
    latest start => {node.latest_start}
}}"""
            for node in self.sorted_nodes
        ]
        return "\n".join(map_list)

    def _get_network(self) -> str:
        network = [
            (
                f"{self.network.get_activity_start_node(activity).id} -{self._line_fmt(activity)}-> {self.network.get_activity_end_node(activity).id} : {activity.activity} (Id={activity.id}, D={activity.duration}, TF={activity.total_float}, FF={activity.free_float})"
                if type(activity) is Activity
                else f"{self.network.get_activity_start_node(activity).id} -{self._line_fmt(activity)}-> {self.network.get_activity_end_node(activity).id}"
            )
            for node in self.sorted_nodes
            for activity in node.outbound_activities
        ]

        return "\n".join(network)

    def _line_fmt(self, activity: Activity | DummyActivity) -> str:
        if type(activity) is Activity:
            if activity.total_float == 0:
                return "[thickness=4]"
            else:
                return ""
        if type(activity) is DummyActivity:
            if (
                self.network.get_activity_start_node(activity).earliest_start
                == self.network.get_activity_end_node(activity).latest_start
            ):
                return "[dashed,thickness=4]"
            else:
                return "[dashed]"
        return ""
