"""Flask REST API for the AoA frontend.

Endpoints:
    GET  /api/project?file=<path>  — Load a YAML project file and return JSON.
    POST /api/project              — Save project JSON back to a YAML file.
    POST /api/network              — Compute the AoA network + CPM from activities JSON.
"""

import os
from datetime import date
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from aoa.model.activity import Activity, ActivityCollection
from aoa.model.cpm import calculate_cpm
from aoa.model.milestones import Milestone
from aoa.model.network import Network, create_network
from aoa.model.project import Project, load_yaml_project, save_yaml_project
from aoa.model.resources import Resource
from aoa.model.state import State
from aoa.transform.coloring_strategy import ColoringStrategies
from aoa.transform.dot import create_dot
from aoa.transform.gantt_data import build_gantt_data
from aoa.transform.theme import resolve_theme
from aoa.transform.to_networkx import to_networkx

app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _activity_to_dict(a: Activity) -> dict:
    return {
        "id": a.id,
        "activity": a.activity,
        "predecessors": sorted(a.predecessors) if a.predecessors else [],
        "planned_effort": a.planned_effort,
        "actual_effort": a.actual_effort,
        "owner": a.owner,
        "resource": a.resource,
        "state": a.state.name,
    }


def _resource_to_dict(r: Resource) -> dict:
    return {
        "id": r.id,
        "name": r.name,
        "workload": r.workload,
        "weekdays": r.weekdays,
        "holidays": r.holidays,
    }


def _milestone_to_dict(m: Milestone) -> dict:
    return {
        "id": m.id,
        "description": m.description,
        "owner": m.owner,
        "due_date": m.due_date.isoformat(),
        "state": m.state.name,
    }


def _project_to_json(project: Project) -> dict:
    return {
        "start": project.start.isoformat(),
        "activities": [_activity_to_dict(a) for a in project.activities],
        "resources": [_resource_to_dict(r) for r in project.resources],
        "milestones": [_milestone_to_dict(m) for m in project.milestones],
    }


def _activities_from_json(data: list[dict]) -> list[Activity]:
    activities = []
    for d in data:
        activities.append(
            Activity(
                id=int(d["id"]),
                activity=d.get("activity", ""),
                predecessors=set(d.get("predecessors", [])),
                planned_effort=float(d.get("planned_effort", 0)),
                actual_effort=float(d.get("actual_effort", 0)),
                owner=d.get("owner", ""),
                resource=d.get("resource", ""),
                state=State[d["state"].upper()] if d.get("state") else State.OPEN,
            )
        )
    return activities


def _resources_from_json(data: list[dict]) -> list[Resource]:
    return [
        Resource(
            id=d.get("id", ""),
            name=d.get("name", ""),
            workload=d.get("workload", ""),
            weekdays=d.get("weekdays", "1111100"),
            holidays=d.get("holidays", []),
        )
        for d in data
    ]


def _milestones_from_json(data: list[dict]) -> list[Milestone]:
    return [
        Milestone(
            id=d.get("id", ""),
            description=d.get("description", ""),
            owner=d.get("owner", ""),
            due_date=date.fromisoformat(d["due_date"]) if d.get("due_date") else date.today(),
            state=State[d["state"].upper()] if d.get("state") else State.OPEN,
        )
        for d in data
    ]


# ---------------------------------------------------------------------------
# Network serialization (for d3-dagre)
# ---------------------------------------------------------------------------


def _network_to_graph_json(network: Network) -> dict:
    """Convert a CPM-annotated Network to a JSON-serializable graph for d3-dagre."""
    nodes = []
    for node in network.get_node_list_sorted_by_depth():
        nodes.append(
            {
                "id": node.id,
                "maxDepth": node.max_depth,
                "earliestStart": node.earliest_start,
                "latestStart": node.latest_start,
            }
        )

    edges = []
    # Compute coloring thresholds (exponential strategy)
    max_float = 0.0
    for activity in network.activities.values():
        if activity.total_float > max_float:
            max_float = activity.total_float
    low_threshold = round(max_float / 9)
    medium_threshold = round(max_float / 3)

    for activity in network.activities.values():
        activity_nodes = network.get_activity_nodes(activity)
        start_node = activity_nodes.start_node
        end_node = activity_nodes.end_node
        if not start_node or not end_node:
            continue

        # Determine color
        color = ""
        if activity.critical:
            color = "critical"
        elif activity.total_float < low_threshold:
            color = "red"
        elif activity.total_float < medium_threshold:
            color = "orange"
        else:
            color = "green"

        edges.append(
            {
                "source": start_node.id,
                "target": end_node.id,
                "activityId": activity.id,
                "label": activity.activity if not activity.is_dummy else "",
                "isDummy": activity.is_dummy,
                "critical": activity.critical,
                "color": color,
                "planned_effort": activity.planned_effort,
                "earliestStart": activity.earliest_start,
                "earliestFinish": activity.earliest_finish,
                "latestStart": activity.latest_start,
                "latestFinish": activity.latest_finish,
                "totalFloat": activity.total_float,
                "freeFloat": activity.free_float,
            }
        )

    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.route("/api/project", methods=["GET"])
def get_project():
    """Load a YAML project file and return its contents as JSON."""
    file_path = request.args.get("file", "")
    if not file_path:
        return jsonify({"error": "Missing 'file' query parameter"}), 400

    path = Path(file_path)
    if not path.is_absolute():
        path = Path(os.getcwd()) / path

    if not path.exists():
        return jsonify({"error": f"File not found: {path}"}), 404

    try:
        project = load_yaml_project(path)
        return jsonify({"file": str(path), "project": _project_to_json(project)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/project", methods=["POST"])
def post_project():
    """Save project JSON to a YAML file."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    file_path = data.get("file", "")
    project_data = data.get("project", {})

    if not file_path:
        return jsonify({"error": "Missing 'file' field"}), 400

    path = Path(file_path)
    if not path.is_absolute():
        path = Path(os.getcwd()) / path

    try:
        start = date.fromisoformat(project_data.get("start", date.today().isoformat()))
        activities = _activities_from_json(project_data.get("activities", []))
        resources = _resources_from_json(project_data.get("resources", []))
        milestones = _milestones_from_json(project_data.get("milestones", []))

        project = Project(start=start, activities=activities, resources=resources, milestones=milestones)
        save_yaml_project(project, path)
        return jsonify({"status": "ok", "file": str(path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/network", methods=["POST"])
def post_network():
    """Compute the AoA network and CPM from activities JSON."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    activities_data = data.get("activities", [])
    if not activities_data:
        return jsonify({"error": "No activities provided"}), 400

    try:
        activities = _activities_from_json(activities_data)
        activity_collection = ActivityCollection(activities=activities)
        network = create_network(activity_collection)
        calculate_cpm(network)
        graph = _network_to_graph_json(network)
        return jsonify(graph)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/network/dot", methods=["POST"])
def post_network_dot():
    """Compute the AoA network + CPM and return Graphviz-rendered SVG."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    activities_data = data.get("activities", [])
    if not activities_data:
        return jsonify({"error": "No activities provided"}), 400

    try:
        activities = _activities_from_json(activities_data)
        activity_collection = ActivityCollection(activities=activities)
        network = create_network(activity_collection)
        calculate_cpm(network)
        nx_graph = to_networkx(network)
        theme = resolve_theme(data.get("theme"))
        rankdir = data.get("rankdir", "TB")
        if rankdir not in ("TB", "BT", "LR", "RL"):
            rankdir = "TB"
        agraph = create_dot(nx_graph, ColoringStrategies.exponential, theme=theme, rankdir=rankdir)
        svg_bytes = agraph.draw(format="svg", prog="dot")
        return app.response_class(svg_bytes, mimetype="image/svg+xml")
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def main():
    app.run(debug=True, port=5000)


@app.route("/api/network/gantt", methods=["POST"])
def post_network_gantt():
    """Compute CPM and return a Mermaid gantt diagram text."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    activities_data = data.get("activities", [])
    if not activities_data:
        return jsonify({"error": "No activities provided"}), 400

    try:
        activities = _activities_from_json(activities_data)
        activity_collection = ActivityCollection(activities=activities)
        network = create_network(activity_collection)
        calculate_cpm(network)

        milestones = _milestones_from_json(data.get("milestones", []))
        start_str = data.get("start")
        start_date = date.fromisoformat(start_str) if start_str else date.today()

        # Only real (non-dummy) activities, sorted by id for stable output
        real_activities = sorted(
            (a for a in network.activities.values() if not a.is_dummy),
            key=lambda a: a.id,
        )
        text = build_gantt_data(real_activities, milestones, start_date)
        return jsonify(text)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    main()
