from aoa.model.activity import Activity, ActivityProtocol, DummyActivity
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
            effort=2,
            predecessors={3},
        ),
        Activity(
            id=5,
            effort=4,
            predecessors={2},
        ),
    ]

    network = Network(activities)
    calculate_latest_finish(network)
    activity_id_lut: dict[int, Activity | DummyActivity] = {}
    for activity in activities:
        activity_id_lut[activity.id] = activity
    assert activity_id_lut[1].latest_finish == 10
    assert activity_id_lut[2].latest_finish == 6
    assert activity_id_lut[3].latest_finish == 8
    assert activity_id_lut[4].latest_finish == 10
    assert activity_id_lut[5].latest_finish == 10
