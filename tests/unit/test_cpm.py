from aoa.model.activity import Activity
from aoa.model.cpm import calculate_earliest_start, calculate_latest_finish
from aoa.model.network import Network


def test_earliest_start_calculation() -> None:
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
    ]

    network = Network(activities)
    calculate_earliest_start(network)
    assert activities[0].earliest_start == 0
    assert activities[1].earliest_start == activities[0].effort


def test_latest_finish_calculation() -> None:
    activities = [
        Activity(
            id=1,
            effort=10,
        ),
        Activity(
            id=2,
            effort=1,
        ),
        Activity(
            id=3,
            effort=1,
            predecessors={2},
        ),
        Activity(
            id=4,
            effort=1,
            predecessors={3},
        ),
        Activity(
            id=5,
            effort=1,
            predecessors={2},
        ),
    ]

    network = Network(activities)
    calculate_earliest_start(network)
    calculate_latest_finish(network)
    assert activities[0].latest_finish == 10
    assert activities[1].latest_finish == 10
    assert activities[2].latest_finish == 10
    assert activities[3].latest_finish == 10
    assert activities[4].latest_finish == 10
