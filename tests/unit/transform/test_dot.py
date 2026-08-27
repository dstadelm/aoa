from aoa.model.activity import Activity, ActivityCollection
from aoa.model.cpm import calculate_cpm
from aoa.model.network import create_network
from aoa.transform.coloring_strategy import ColoringStrategies
from aoa.transform.dot import create_dot
from aoa.transform.theme import THEMES
from aoa.transform.to_networkx import to_networkx


def _build_network():
    activities = [
        Activity(id=1, effort=5),
        Activity(id=2, effort=3, predecessors=set([1])),
        Activity(id=3, effort=5),
        Activity(id=4, effort=5, predecessors=set([3])),
    ]
    network = create_network(ActivityCollection(activities))
    calculate_cpm(network)
    return network


def test_create_dot_applies_theme_defaults() -> None:
    theme = THEMES["tokyonight"]
    network = _build_network()
    gvz = create_dot(to_networkx(network), ColoringStrategies.exponential, theme=theme)

    assert gvz.graph_attr["bgcolor"] == theme.bgcolor
    assert gvz.node_attr["fillcolor"] == theme.node_fill
    assert gvz.node_attr["color"] == theme.node_stroke
    assert gvz.node_attr["fontcolor"] == theme.text
    assert gvz.edge_attr["color"] == theme.edge
    assert gvz.edge_attr["fontcolor"] == theme.text_muted


def test_create_dot_uses_theme_critical_color_for_critical_edges() -> None:
    theme = THEMES["catppuccin-mocha"]
    network = _build_network()
    gvz = create_dot(to_networkx(network), ColoringStrategies.exponential, theme=theme)

    dot_source = gvz.to_string()
    # Critical color must appear at least once (activities 3 & 4 are on critical path).
    assert theme.critical.lower() in dot_source.lower()


def test_create_dot_default_theme_produces_material_light_bgcolor() -> None:
    network = _build_network()
    gvz = create_dot(to_networkx(network), ColoringStrategies.exponential)
    assert gvz.graph_attr["bgcolor"] == THEMES["material-light"].bgcolor
    assert gvz.node_attr["fillcolor"] == THEMES["material-light"].node_fill
