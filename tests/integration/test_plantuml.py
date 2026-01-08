from datetime import date

from aoa.model.activity import Activity
from aoa.model.cpm import calculate_cpm
from aoa.model.network import Network
from aoa.transform.plantuml import PlantUml
from aoa.transformation.gantt import create_gantt


def test_simple_diamond_pert() -> None:
    activities = [
        Activity(
            id=1,
            effort=5,
        ),
        Activity(
            id=2,
            effort=3,
            predecessors=set([1]),
        ),
        Activity(
            id=3,
            effort=5,
        ),
        Activity(
            id=4,
            effort=5,
            predecessors=set([3]),
        ),
    ]

    network = Network(activities)
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
            effort=5,
        ),
        Activity(
            id=2,
            activity="Design Phase",
            effort=3,
            predecessors=set([1]),
        ),
        Activity(
            id=3,
            activity="Implementation Phase",
            effort=5,
            predecessors=set([2]),
        ),
        Activity(
            id=4,
            activity="Testing Phase",
            effort=5,
            predecessors=set([3]),
        ),
    ]

    network = Network(activities)
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
