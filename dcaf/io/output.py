"""Utilities for locating and loading D-CAF stellar snapshots.

D-CAF writes stellar snapshots as AMUSE files named
``stars_###.amuse``. These helpers identify snapshots across continuous output
segments and return the stellar particles together with their saved model time.
"""

import glob
import os
import re

from amuse.io import read_set_from_file


def snapshot_filename(
    snapshot_index: int,
    source_folder: str = "dcaf_output",
    snapshot_basename: str = "stars_",
) -> str:
    """Return the path for a numbered D-CAF snapshot.

    Args:
        snapshot_index: Non-negative snapshot number.
        source_folder: Directory containing the snapshot.
        snapshot_basename: Filename prefix before the index.

    Returns:
        (str): Path in the form
            ``<source_folder>/<snapshot_basename>###.amuse``.
    """
    filename = f"{snapshot_basename}{snapshot_index:03d}.amuse"
    return os.path.join(source_folder, filename)


def snapshot_index_from_path(path: str, snapshot_basename: str = "stars_") -> int:
    """Extract a snapshot index from an AMUSE snapshot path.

    Args:
        path: Path whose basename follows the snapshot pattern.
        snapshot_basename: Expected filename prefix.

    Returns:
        (int): Integer encoded between the prefix and
            ``.amuse`` suffix.

    Raises:
        ValueError: If the basename does not match
            ``<snapshot_basename>###.amuse``.
    """
    name = os.path.basename(path)
    m = re.match(rf"{re.escape(snapshot_basename)}(\d+)\.amuse$", name)
    if m is None:
        raise ValueError(
            f"File '{path}' does not match snapshot pattern "
            f"'{snapshot_basename}###.amuse'"
        )
    return int(m.group(1))


def find_snapshot_files(
    source_folder: str = "dcaf_output",
    snapshot_basename: str = "stars_",
) -> list[tuple[int, str]]:
    """Find valid snapshot files in a folder, ordered by snapshot index.

    Args:
        source_folder: Directory to search.
        snapshot_basename: Filename prefix that identifies
            snapshots.

    Returns:
        (list[tuple[int, str]]): ``(snapshot_index, path)``
            pairs in ascending index order. Files that match the glob but not
            the complete snapshot pattern are ignored.
    """
    pattern = os.path.join(source_folder, f"{snapshot_basename}*.amuse")
    files = glob.glob(pattern)

    snapshots = []
    for path in files:
        try:
            idx = snapshot_index_from_path(path, snapshot_basename=snapshot_basename)
            snapshots.append((idx, path))
        except ValueError:
            pass

    snapshots.sort(key=lambda x: x[0])
    return snapshots


def find_latest_snapshot(
    source_folder: str = "dcaf_output",
    snapshot_basename: str = "stars_",
) -> tuple[int, str]:
    """Return the highest-indexed valid snapshot in a folder.

    Args:
        source_folder: Directory to search.
        snapshot_basename: Filename prefix that identifies
            snapshots.

    Returns:
        (tuple[int, str]): Index and path of the latest snapshot.

    Raises:
        FileNotFoundError: If the folder contains no valid snapshots with the
            requested basename.
    """
    snapshots = find_snapshot_files(
        source_folder=source_folder,
        snapshot_basename=snapshot_basename,
    )

    if len(snapshots) == 0:
        raise FileNotFoundError(
            f"No snapshots found in '{source_folder}' with basename '{snapshot_basename}'"
        )

    return snapshots[-1]

def get_output_folders(base_output_folder: str = "dcaf_output") -> list[str]:
    """Return continuous D-CAF output segments for a base folder.

    Args:
        base_output_folder: Base output directory. Segments are
            named ``<base>``, ``<base>_1``, ``<base>_2``, and so on.

    Returns:
        (list[str]): Existing segment paths in segment order.

    Raises:
        FileNotFoundError: If no matching directory exists or the segment
            numbering has a gap.
    """
    base = base_output_folder.rstrip("/")
    parent = os.path.dirname(base)
    if parent == "":
        parent = "."
    stem = os.path.basename(base)

    folders = []
    pattern = os.path.join(parent, stem + "*")

    for path in glob.glob(pattern):
        if not os.path.isdir(path):
            continue

        name = os.path.basename(path.rstrip("/"))

        if name == stem:
            seg = 0
        else:
            m = re.fullmatch(rf"{re.escape(stem)}_(\d+)", name)
            if m is None:
                continue
            seg = int(m.group(1))

        folders.append((seg, path))

    folders.sort(key=lambda x: x[0])

    if len(folders) == 0:
        raise FileNotFoundError(
            f"No output folders found matching '{base_output_folder}'"
        )

    for expected_seg, (seg, path) in enumerate(folders):
        if seg != expected_seg:
            raise FileNotFoundError(
                "Non-continuous output folders found for "
                f"'{base_output_folder}': expected segment {expected_seg}, "
                f"found '{path}'."
            )

    return [path for seg, path in folders]


def load_snapshot(path: str) -> dict:
    """Load an AMUSE snapshot and its saved model time.

    Args:
        path: Path to an AMUSE-format stellar snapshot.

    Returns:
        (dict): State with ``path``, ``snapshot_index``
            (initially ``None``), ``stars`` (AMUSE ``Particles``), and
            ``model_time`` (an AMUSE ``Quantity``).

    Raises:
        ValueError: If the snapshot has no saved ``collection_attributes.model_time``.
    """
    stars = read_set_from_file(path, format="amuse")

    if not hasattr(stars.collection_attributes, "model_time"):
        raise ValueError(
            f"Snapshot '{path}' does not contain collection_attributes.model_time"
        )

    model_time = stars.collection_attributes.model_time

    return {
        "path": path,
        "snapshot_index": None,
        "stars": stars,
        "model_time": model_time,
    }


def load_snapshot_by_index(
    snapshot_index: int,
    source_folder: str = "dcaf_output",
    snapshot_basename: str = "stars_",
) -> dict:
    """Load the snapshot with a specified index.

    Args:
        snapshot_index: Snapshot number to load.
        source_folder: Directory containing the snapshot.
        snapshot_basename: Filename prefix before the index.

    Returns:
        (dict): Loaded snapshot state, including its requested ``snapshot_index``.
    """
    path = snapshot_filename(
        snapshot_index,
        source_folder=source_folder,
        snapshot_basename=snapshot_basename,
    )
    state = load_snapshot(path)
    state["snapshot_index"] = snapshot_index
    return state


def load_latest_snapshot(
    source_folder: str = "dcaf_output",
    snapshot_basename: str = "stars_",
) -> dict:
    """Load the highest-indexed valid snapshot in a folder.

    Args:
        source_folder: Directory to search.
        snapshot_basename: Filename prefix that identifies
            snapshots.

    Returns:
        (dict): Loaded state of the latest snapshot, including its
            ``snapshot_index``.
    """
    snapshot_index, path = find_latest_snapshot(
        source_folder=source_folder,
        snapshot_basename=snapshot_basename,
    )

    state = load_snapshot(path)
    state["snapshot_index"] = snapshot_index
    return state
