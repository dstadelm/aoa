from aoa.model.activity import Activity
from aoa.model.resources import ResourceCollection


def duration(activity: Activity, ressources: ResourceCollection) -> float:
    """Calculate the duration of an activity based on its effort and the resource's workload.

    Args:
        activity (Activity): The activity to calculate the duration for
        ressources (ResourceCollection): The resource collection to look up the resource's workload
    Returns:
        float: The duration of the activity
    """
    if activity.is_dummy:
        return 0.0

    if not activity.resource:
        return activity.effort

    resource = ressources.get(activity.resource, None)
    if not resource:
        raise ValueError(
            f"Activity {activity.id} has resource {activity.resource} which is not in the resource collection"
        )
    workload = float(resource.workload) / 100
    if workload <= 0:
        raise ValueError(f"Resource {resource.id} has invalid workload {resource.workload}")

    return activity.effort / workload
