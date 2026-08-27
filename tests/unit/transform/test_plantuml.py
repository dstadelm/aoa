from datetime import date

from aoa.model.activity import Activity, ActivityCollection
from aoa.model.cpm import calculate_cpm
from aoa.model.network import create_network
from aoa.transform.gantt import create_gantt
from aoa.transform.plantuml import PlantUml
from aoa.transform.coloring_strategy import ColoringStrategies
from aoa.transform.theme import THEMES


def test_simple_diamond_pert() -> None:
    activities = [
        Activity(
            id=1,
            planned_effort=5,
        ),
        Activity(
            id=2,
            planned_effort=3,
            predecessors=set([1]),
        ),
        Activity(
            id=3,
            planned_effort=5,
        ),
        Activity(
            id=4,
            planned_effort=5,
            predecessors=set([3]),
        ),
    ]

    network = create_network(ActivityCollection(activities))
    calculate_cpm(network)
    plantuml = PlantUml(network)
    result = plantuml.get_txt()
    assert (
        result
        == """@startuml PERT
top to bottom direction
' Horizontal lines: -->, <--, <-->
' Vertical lines: ->, <-, <->
title Pert: Project Design

map 0 {
    earliest start => 0
    latest start => 0
}
map 1 {
    earliest start => 5
    latest start => 7
}
map 2 {
    earliest start => 5
    latest start => 5
}
map 3 {
    earliest start => 10
    latest start => 10
}
0 --> 1 :  (Id=1, D=5, TF=2, FF=0)
0 -[thickness=4]-> 2 :  (Id=3, D=5, TF=0, FF=0)
1 --> 3 :  (Id=2, D=3, TF=2, FF=2)
2 -[thickness=4]-> 3 :  (Id=4, D=5, TF=0, FF=0)
@enduml"""
    )


def test_simple_gantt() -> None:
    activities = [
        Activity(
            id=1,
            activity="Start Project",
            planned_effort=5,
        ),
        Activity(
            id=2,
            activity="Design Phase",
            planned_effort=3,
            predecessors=set([1]),
        ),
        Activity(
            id=3,
            activity="Implementation Phase",
            planned_effort=5,
            predecessors=set([2]),
        ),
        Activity(
            id=4,
            activity="Testing Phase",
            planned_effort=5,
            predecessors=set([3]),
        ),
    ]

    network = create_network(ActivityCollection(activities))
    calculate_cpm(network)
    result = create_gantt(activities, milestones=[], start_date=date.fromisoformat("2024-01-01"))
    assert (
        result
        == """@startgantt
project starts at 2024-01-01
sunday are closed
saturday are closed
printscale weekly

[Start Project] as [TASK_1] requires 5 days
[Design Phase] as [TASK_2] requires 3 days
[Implementation Phase] as [TASK_3] requires 5 days
[Testing Phase] as [TASK_4] requires 5 days
[TASK_1]->[TASK_2]
[TASK_2]->[TASK_3]
[TASK_3]->[TASK_4]


@endgantt"""
    )


def _diamond_network():
    activities = [
        Activity(id=1, planned_effort=5),
        Activity(id=2, planned_effort=3, predecessors=set([1])),
        Activity(id=3, planned_effort=5),
        Activity(id=4, planned_effort=5, predecessors=set([3])),
    ]
    network = create_network(ActivityCollection(activities))
    calculate_cpm(network)
    return network


def test_themed_pert_contains_skinparams() -> None:
    theme = THEMES["tokyonight"]
    network = _diamond_network()
    result = PlantUml(network, theme=theme).get_txt()
    assert "skinparam backgroundColor transparent" in result
    assert f"skinparam ArrowColor {theme.edge}" in result
    assert f"skinparam defaultFontColor {theme.text}" in result
    assert f"BackgroundColor {theme.node_fill}" in result


def test_themed_pert_colors_critical_arrows() -> None:
    theme = THEMES["tokyonight"]
    network = _diamond_network()
    result = PlantUml(network, theme=theme).get_txt()
    critical_hex = theme.critical.lstrip("#")
    # Critical activities on the CP should be styled with the theme's critical color.
    assert f"[#{critical_hex},thickness=4]" in result


def test_themed_pert_colors_non_critical_arrows_via_strategy() -> None:
    theme = THEMES["tokyonight"]
    network = _diamond_network()
    result = PlantUml(
        network,
        theme=theme,
        coloring_strategy=ColoringStrategies.exponential,
    ).get_txt()
    # At least one non-critical activity must carry a themed color that is
    # NOT the critical color (activities 1 and 2 have positive float).
    for color in (theme.red, theme.orange, theme.green):
        if f"[#{color.lstrip('#')}]" in result:
            return
    raise AssertionError("expected at least one themed non-critical arrow color")


def test_untinted_pert_matches_legacy_output() -> None:
    """Without a theme, output must be identical to the legacy format."""
    network = _diamond_network()
    result = PlantUml(network).get_txt()
    assert "skinparam" not in result
    assert "0 -[thickness=4]-> 2" in result
