from datetime import date

from aoa.model.activity import Activity
from aoa.model.milestones import Milestone


def create_gantt(activities: list[Activity], milestones: list[Milestone], start_date: date) -> str:
    ret = create_header(start_date) + "\n"
    ret += "\n".join(create_tasks(activities)) + "\n"
    ret += "\n".join(create_dependencies(activities)) + "\n"
    ret += "\n".join(create_milestones(milestones)) + "\n"
    # ret += "\n".join(minimize_milestones(milestones)) + "\n"
    ret += "\n".join(create_legend(milestones)) + "\n"
    ret += create_footer()
    return ret


def create_header(start_date: date) -> str:
    return f"""@startgantt
project starts at {start_date.isoformat()}
sunday are closed
saturday are closed
printscale weekly
"""


def create_footer() -> str:
    return "@endgantt"


def create_tasks(activities: list[Activity]) -> list[str]:
    return [
        f"[{activity.activity}] as [TASK_{activity.id}] requires {activity.planned_effort} days" for activity in activities
    ]


def create_dependencies(activities: list[Activity]) -> list[str]:
    return [
        f"[TASK_{predecessor}]->[TASK_{activity.id}]"
        for activity in activities
        for predecessor in activity.predecessors
    ]


def create_legend(milestones: list[Milestone]) -> list[str]:
    if not milestones:
        return []
    ret = ["legend"]
    ret.extend(milestone_legend(milestones))
    ret.append("end legend")
    return ret


def create_milestones(milestones: list[Milestone]) -> list[str]:
    return [f"[{milestone.id}] happens at {milestone.due_date.isoformat()}" for milestone in milestones]


def milestone_legend(milestones: list[Milestone]) -> list[str]:
    return [f"[{milestone.id}] {milestone.description}" for milestone in milestones]


def minimize_milestones(milestones: list[Milestone]) -> list[str]:
    due_date_dict: dict[str, list[str]] = {}
    for milestone in milestones:
        due_date_dict.setdefault(milestone.due_date.isoformat(), []).append(milestone.id)

    def recurse_dates(due_date: dict[str, list[str]], result: list[str]) -> list[str]:
        if not due_date:
            return result

        new_dict: dict[str, list[str]] = dict()

        same_row_milestones: list[str] = []
        for key, value in due_date.items():
            same_row_milestones.append(value[0])
            if len(value) > 1:
                new_dict[key] = value[1:]

        if len(same_row_milestones) > 1:
            result.extend(
                [f"[{m}] displays on same row as [{same_row_milestones[0]}]" for m in same_row_milestones[1:]]
            )

        return recurse_dates(new_dict, result)

    return recurse_dates(due_date_dict, [])
