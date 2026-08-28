"""Build JSON gantt data (dates, dependencies, milestones) for the frontend d3 gantt renderer.

Dates are computed by adding working days (Mon–Fri) to the project start date.
"""

from datetime import date, timedelta
from math import ceil

from aoa.model.activity import Activity
from aoa.model.milestones import Milestone


def add_working_days(start: date, days: int) -> date:
    """Return `start` shifted forward by `days` working days (Mon–Fri)."""
    if days <= 0:
        return _next_working_day(start)
    current = _next_working_day(start)
    remaining = days
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def _next_working_day(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def build_gantt_data(
    activities: list[Activity],
    milestones: list[Milestone],
    start_date: date,
) -> dict:
    real_ids = {a.id for a in activities if not a.is_dummy}
    tasks = []
    for a in activities:
        if a.is_dummy:
            continue
        start_offset = int(a.earliest_start)
        duration = max(1, ceil(a.duration))
        task_start = add_working_days(start_date, start_offset)
        task_end = add_working_days(task_start, duration)
        preds = sorted(f"T{p}" for p in a.predecessors if p in real_ids)
        tasks.append(
            {
                "id": f"T{a.id}",
                "label": a.activity or f"Activity {a.id}",
                "start": task_start.isoformat(),
                "end": task_end.isoformat(),
                "duration": duration,
                "critical": a.critical,
                "predecessors": preds,
            }
        )

    ms = [
        {
            "id": m.id,
            "label": m.description or m.id,
            "date": m.due_date.isoformat(),
        }
        for m in milestones
    ]

    return {"start": start_date.isoformat(), "tasks": tasks, "milestones": ms}
