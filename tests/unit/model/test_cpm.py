from aoa.model.activity import Activity, ActivityCollection
from aoa.model.cpm import _calculate_earliest_start  # pyright: ignore [reportPrivateUsage]
from aoa.model.cpm import _calculate_free_float  # pyright: ignore [reportPrivateUsage]
from aoa.model.cpm import _calculate_latest_finish  # pyright: ignore [reportPrivateUsage]
from aoa.model.network import Network, create_network


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

    network = create_network(ActivityCollection(activities))
    _calculate_earliest_start(network)
    annotated_activities = network.activities
    assert annotated_activities[1].earliest_start == 0
    assert annotated_activities[2].earliest_start == activities[0].effort


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

    network = create_network(ActivityCollection(activities))
    _calculate_latest_finish(network)
    assert network.activities[1].latest_finish == 10
    assert network.activities[2].latest_finish == 6
    assert network.activities[3].latest_finish == 8
    assert network.activities[4].latest_finish == 10
    assert network.activities[5].latest_finish == 10


def test_total_float() -> None:
    activities = [
        Activity(
            id=1,
            effort=10,
        ),
        Activity(
            id=2,
            effort=2,
        ),
        Activity(
            id=3,
            effort=1,
            predecessors={2},
        ),
    ]
    collection = ActivityCollection(activities)
    network = create_network(collection)
    _calculate_latest_finish(network)

    assert collection[2].total_float == 7
    assert collection[3].total_float == 7
    assert collection[1].total_float == 0


def test_free_float() -> None:
    activities = [
        Activity(
            id=1,
            effort=10,
        ),
        Activity(
            id=2,
            effort=2,
        ),
        Activity(
            id=3,
            effort=1,
            predecessors={2},
        ),
    ]

    collection = ActivityCollection(activities)
    network = create_network(collection)
    _calculate_free_float(network)

    assert collection[2].free_float == 0
    assert collection[3].free_float == 7
    assert collection[1].free_float == 0
