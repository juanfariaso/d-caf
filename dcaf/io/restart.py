"""Helpers for restoring a saved D-CAF stellar snapshot.

Resuming is supported only after star formation has completed.
The check compares the snapshot model time with the final formation time
reported by the supplied star-formation framework.
"""

from dcaf.io.output import load_latest_snapshot, load_snapshot_by_index
from dcaf.framework import StarFormationFramework
from amuse.units.quantities import Quantity


def formation_finished_at_time(
    model_time: Quantity,
    framework: StarFormationFramework,
) -> bool:
    """Determine whether star formation has finished at a model time.

    Args:
        model_time: Snapshot or simulation time to test.
        framework: Framework providing ``get_last_formation_time()``.

    Returns:
        (bool): ``True`` when the framework reports no scheduled final formation
            time, or when ``model_time`` is at or after it.
    """
    last_time = framework.get_last_formation_time()

    if last_time is None:
        return True

    return model_time >= last_time


def validate_resume_after_formation(
    model_time: Quantity,
    framework: StarFormationFramework,
) -> None:
    """Raise an error when a snapshot predates the end of star formation.

    Args:
        model_time: Saved snapshot time to validate.
        framework: Framework providing the final formation time.

    Raises:
        ValueError:  If the snapshot time is earlier than the framework's final
            scheduled formation time.
    """
    if not formation_finished_at_time(model_time, framework):
        last_time = framework.get_last_formation_time()
        raise ValueError(
            "Resume only supported after star formation is finished. "
            f"Snapshot time is {model_time.in_(last_time.unit)}, "
            f"but last formation time is {last_time.in_(last_time.unit)}."
        )


def get_resume_state(
    snapshot_index: int = None,
    framework: StarFormationFramework = None,
    source_folder: str = "dcaf_output",
    snapshot_basename: str = "stars_",
) -> dict:
    """Load a snapshot state suitable for D-CAF resumption.

    Args:
        snapshot_index: Snapshot number to load. The latest
            valid snapshot is used when omitted.
        framework: Optional framework used to confirm formation
            has finished before restoration.
        source_folder: Directory containing the source snapshots.
        snapshot_basename: Filename prefix that identifies snapshots.

    Returns:
        (dict): Loaded snapshot state containing ``path``, ``snapshot_index``,
            ``stars``, ``model_time``, and ``source_folder``.

    Raises:
        ValueError: If a framework is supplied and the snapshot predates the end
            of star formation.
    """

    if snapshot_index is None:
        state = load_latest_snapshot(
            source_folder = source_folder,
            snapshot_basename=snapshot_basename,
        )
    else:
        state = load_snapshot_by_index(
            snapshot_index,
            source_folder=source_folder,
            snapshot_basename=snapshot_basename,
        )

    state["source_folder"] = source_folder

    if framework is not None:
        validate_resume_after_formation(state["model_time"], framework)

    return state
