"""Shared read-modify-write boundaries for computed collection results.

IPD and Delta-MTP both update ``processed.pkl`` and then regenerate every trajectory JSON.
They must therefore use the same per-collection lock: feature-private locks would allow both
to load one old dataframe and whichever persisted last would silently erase the other result.

This module owns no science and no HTTP. It only locates stored frames/rows and invokes the
one complete collection serializer after a successful mutation.
"""

from __future__ import annotations

import threading

from backend.services import system_processing, system_serialization, trajectory_service

NotFound = trajectory_service.NotFound

_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


def lock_for(upload_id):
    """One shared read-modify-write lock per stored collection."""
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(upload_id), threading.Lock())


def frame_entry(upload_id, slug, frame_index):
    """Return a system payload and its frame identified by animation ``frame_index``."""
    system = trajectory_service.get_trajectory(upload_id, slug)
    for frame in system["frames"]:
        if int(frame["frame_index"]) == int(frame_index):
            return system, frame
    raise NotFound(
        f"No frame {frame_index} in system {slug!r} of upload {upload_id!r}; "
        f"it has {len(system['frames'])} frames."
    )


def row_label(df, frame):
    """Map a serialized frame's stable row position to the dataframe label."""
    position = int(frame["row_index"])
    if not 0 <= position < len(df):
        raise NotFound(
            f"Frame {frame['frame_index']} refers to row {position}, which is outside the "
            f"stored dataframe's {len(df)} rows."
        )
    return df.index[position]


def persist(upload_id, df):
    """Regenerate the complete collection from a modified processed dataframe.

    ``save_collection`` writes each artifact through a same-directory temporary file and
    publishes the manifest last. Derived geometry fields are intentionally not recomputed.
    """
    manifest = system_serialization.load_manifest(upload_id)
    return system_serialization.save_collection(
        upload_id,
        df,
        system_processing.group_trajectories(df),
        source_filename=manifest["source_filename"],
        display_name=manifest.get("display_name"),
        validation=manifest.get("validation"),
    )
