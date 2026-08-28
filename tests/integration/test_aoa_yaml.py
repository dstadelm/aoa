from pathlib import Path

from aoa.model.cpm import calculate_cpm
from aoa.model.network import create_network
from aoa.model.project import load_yaml_project

AOA_YAML = Path(__file__).parent.parent / "artefacts" / "AoA.yaml"


def test_aoa_yaml_loads_and_builds_network() -> None:
    """Regression test: AoA.yaml is a real-world diamond-heavy graph that previously
    triggered a false-positive cycle detection. Ensures it loads, validates, builds
    a network, and produces a stable CPM result.
    """
    project = load_yaml_project(AOA_YAML)
    activities = project.get_activities()

    assert len(activities) == 15

    network = create_network(activities)
    nodes = network.get_node_list_sorted_by_depth()
    assert len(nodes) == 13

    calculate_cpm(network, project.get_resources())

    critical_ids = sorted(a.id for a in activities.values() if not a.is_dummy and a.critical)
    assert critical_ids == [3, 5, 10, 13, 15]

    project_duration = max(a.earliest_finish for a in activities.values())
    # Critical-path effort = 80 days at 70% workload → 80 / 0.7 ≈ 114.2857
    assert project_duration == 80.0 / 0.7
