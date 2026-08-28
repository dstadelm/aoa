from pathlib import Path
from typing import Optional

from aoa.model.activity import Activity
from aoa.model.duration import duration as compute_duration
from aoa.model.network import Network
from aoa.model.node import Node
from aoa.model.resources import ResourceCollection

from .coloring_strategy import ColoringStrategyProtocol
from .theme import Theme


class PlantUml:
    def __init__(
        self,
        network: Network,
        resources: Optional[ResourceCollection] = None,
        theme: Optional[Theme] = None,
        coloring_strategy: Optional[ColoringStrategyProtocol] = None,
    ):
        self.plantuml: str = ""
        self.sorted_nodes: list[Node] = network.get_node_list_sorted_by_depth()
        self.network: Network = network
        self.resources: ResourceCollection = resources if resources is not None else ResourceCollection([])
        self.theme: Optional[Theme] = theme
        self._activity_color_tokens: dict[int, str] = {}
        if coloring_strategy is not None:
            self._activity_color_tokens = self._compute_color_tokens(network, coloring_strategy)

    @staticmethod
    def _compute_color_tokens(
        network: Network, coloring_strategy: ColoringStrategyProtocol
    ) -> dict[int, str]:
        """Run the strategy over a networkx view and collect tokens per activity."""
        # Local import to avoid a hard dependency for callers that don't need coloring.
        from .to_networkx import to_networkx

        graph = to_networkx(network)
        coloring_strategy(graph)
        tokens: dict[int, str] = {}
        for _, _, data in graph.edges(data=True):
            activity: Activity = data["activity"]
            token = data.get("color_token")
            if token is not None:
                tokens[activity.id] = token
        return tokens

    def get_txt(self) -> str:
        return self._get_header() + self._get_map() + "\n" + self._get_network() + self._get_trailer()

    def write_txt(self, file: Path) -> None:
        with open(file, "w") as f:
            _ = f.write(self.get_txt())

    def _get_header(self) -> str:
        theme_block = self._get_theme_skinparams()
        return f"""@startuml PERT
top to bottom direction
' Horizontal lines: -->, <--, <-->
' Vertical lines: ->, <-, <->
title Pert: Project Design
{theme_block}
"""

    def _get_theme_skinparams(self) -> str:
        if self.theme is None:
            return ""
        t = self.theme
        return (
            f"skinparam backgroundColor {t.bgcolor}\n"
            f"skinparam defaultFontColor {t.text}\n"
            f"skinparam ArrowColor {t.edge}\n"
            f"skinparam ArrowFontColor {t.text_muted}\n"
            f"skinparam map {{\n"
            f"    BackgroundColor {t.node_fill}\n"
            f"    BorderColor {t.node_stroke}\n"
            f"    FontColor {t.text}\n"
            f"}}\n"
        )

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
                f"{self.network.get_activity_nodes(activity).start_node.id} -{self._line_fmt(activity)}-> {self.network.get_activity_nodes(activity).end_node.id}"
                if activity.is_dummy
                else f"{self.network.get_activity_nodes(activity).start_node.id} -{self._line_fmt(activity)}-> {self.network.get_activity_nodes(activity).end_node.id} : {activity.activity} (Id={activity.id}, D={compute_duration(activity, self.resources)}, TF={activity.total_float}, FF={activity.free_float})"
            )
            for node in self.sorted_nodes
            for activity in node.outbound_activities
        ]

        return "\n".join(network)

    def _line_fmt(self, activity: Activity) -> str:
        if not activity.is_dummy:
            if activity.total_float == 0:
                return self._format_style(["thickness=4"], critical=True)
            token = self._activity_color_tokens.get(activity.id)
            if token and token != "critical":
                return self._format_style([], token=token)
            return self._format_style([])
        else:
            tight = (
                self.network.get_activity_start_node(activity).earliest_start
                == self.network.get_activity_end_node(activity).latest_start
            )
            if tight:
                return self._format_style(["dashed", "thickness=4"], dummy=True)
            return self._format_style(["dashed"], dummy=True)

    def _format_style(
        self,
        parts: list[str],
        critical: bool = False,
        dummy: bool = False,
        token: Optional[str] = None,
    ) -> str:
        if self.theme is not None:
            if critical:
                parts = [f"#{self._strip(self.theme.critical)}"] + parts
            elif dummy:
                parts = parts + [f"#{self._strip(self.theme.edge)}"]
            elif token is not None:
                color = self._theme_color_for_token(token)
                if color:
                    parts = parts + [f"#{self._strip(color)}"]
        if not parts:
            return ""
        return f"[{','.join(parts)}]"

    def _theme_color_for_token(self, token: str) -> Optional[str]:
        if self.theme is None:
            return None
        return {
            "critical": self.theme.critical,
            "red": self.theme.red,
            "orange": self.theme.orange,
            "green": self.theme.green,
            "edge": self.theme.edge,
        }.get(token)

    @staticmethod
    def _strip(color: str) -> str:
        return color.lstrip("#")
