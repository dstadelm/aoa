from pathlib import Path
from typing import Union

from activity import Activity, DummyActivity
from network import Network


class PlantUml:
    def __init__(self, network: Network):
        self.plantuml: str = ""
        self.sorted_nodes = network.get_node_list_sorted_by_depth()
        self.activity_node_lut = network._activities

    def get_txt(self) -> str:
        return self._get_header() + self._get_map() + "\n" + self._get_network() + self._get_trailer()

    def write_txt(self, file: Path) -> None:
        with open(file, "w") as f:
            f.write(self.get_txt())

    def _get_header(self) -> str:
        return """``` plantuml
@startuml PERT
top to bottom direction
' Horizontal lines: -->, <--, <-->
' Vertical lines: ->, <-, <->
title Pert: Project Design

"""

    def _get_trailer(self) -> str:
        return "\n@enduml\n```"

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
            f"{self.activity_node_lut[activity.id].start_node.id} -{self._line_fmt(activity)}-> {self.activity_node_lut[activity.id].end_node.id} : {activity.description} (Id={activity.id}, D={activity.duration}, TF={activity.total_float}, FF={activity.free_float})"
            if type(activity) == Activity
            else f"{self.activity_node_lut[activity.id].start_node.id} -{self._line_fmt(activity)}-> {self.activity_node_lut[activity.id].end_node.id}"
            for node in self.sorted_nodes
            for activity in node.outbound_activities
        ]

        return "\n".join(network)

    def _line_fmt(self, activity: Union[Activity, DummyActivity]) -> str:
        if type(activity) == Activity:
            if activity.total_float == 0:
                return "[thickness=4]"
            else:
                return ""
        if type(activity) == DummyActivity:
            if (
                self.activity_node_lut[activity.id].start_node.earliest_start
                == self.activity_node_lut[activity.id].end_node.latest_start
            ):
                return "[dashed,thickness=4]"
            else:
                return "[dashed]"
        return ""
