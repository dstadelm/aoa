from pathlib import Path

from pyfakefs.fake_filesystem import FakeFilesystem

from aoa.model.network import Network
from aoa.model.node import Node
from aoa.model.project import load_yaml_project


def test_yaml_network(fs: FakeFilesystem) -> None:
    _ = fs.create_file(  # pyright: ignore [reportUnknownMemberType]
        "network.yaml",
        contents="""
activities: 
    - {id: 1,  activity: "activity 1"}
    - {id: 2,  activity: "activity 2"}
    - {id: 3,  activity: "activity 3"}
    - {id: 4,  activity: "activity 4"}
    - {id: 5,  activity: "activity 5"}
    - {id: 6,  activity: "activity 6", predecessors: [1,2]}
    - {id: 7,  activity: "activity 7", predecessors: [2,3]}
    - {id: 8,  activity: "activity 8", predecessors: [3,4]}
    - {id: 9,  activity: "activity 9", predecessors: [4,5]}
    - {id: 10, activity: "activity 10", predecessors: [1,2,3]}
    - {id: 11, activity: "activity 11", predecessors: [1,2,3,4,5]}
        """,
    )
    project = load_yaml_project(Path("network.yaml"))

    network = Network(project.activities)
    nodes: list[Node] = network.get_node_list_sorted_by_depth()
    assert len(nodes) == 11
    assert nodes[0].outbound_activities == [
        project.activities[0],
        project.activities[1],
        project.activities[2],
        project.activities[3],
        project.activities[4],
    ]

    def set_to_str(value: set[int]) -> str:
        if not value:
            return "start"
        return "-".join(str(v) for v in sorted(value))

    node_start_dependencies = [set_to_str(node.start_dependencies) for node in nodes]
    assert "start" in node_start_dependencies
    assert "2" in node_start_dependencies
    assert "3" in node_start_dependencies
    assert "4" in node_start_dependencies
    assert "1-2" in node_start_dependencies
    assert "2-3" in node_start_dependencies
    assert "3-4" in node_start_dependencies
    assert "4-5" in node_start_dependencies
    assert "1-2-3" in node_start_dependencies
    assert "1-2-3-4-5" in node_start_dependencies
    assert "1-2-3-4-5-6-7-8-9-10-11" in node_start_dependencies
    assert len(project.activities) == 11
