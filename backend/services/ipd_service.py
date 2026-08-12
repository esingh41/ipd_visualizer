"""Serving and computing IPD for a stored frame.

The disk and workflow layer, parallel to ``trajectory_service``: that one answers from the
JSON written at upload time, this one reaches back into ``processed.pkl`` because induced-dipole
histories are too large to have been written there.

Three entry points, all addressed the way the frontend addresses everything else -- by upload
id, system slug and *frame index*::

    get_history      one frame, one mode, the full SCF trajectory
    compute          run one frame and one mode, persist, hand back the result
    compute_system   the same across a whole trajectory, persisted once

A frame index is a position in the animation, not a position in the dataframe: frames are
sorted by separation while rows arrive in build order. The mapping between them was decided at
upload and lives in the stored system JSON, so it is read from there rather than re-derived.

Writes take a per-collection lock. ``system_serialization._write_atomic`` makes each individual
file replacement atomic, but a compute is a read-modify-write spanning the pickle *and* every
system JSON, and the SCF in the middle of it takes real time. Two concurrent computes on one
collection without this would both start from the same dataframe and the second would erase the
first's result -- with no error, and no way to tell afterwards.
"""

from __future__ import annotations

import threading

import numpy as np

from backend.services import (
    capability,
    dataframe_schema,
    ipd_results,
    system_processing,
    system_serialization,
    thole_damping,
    trajectory_service,
)
from backend.services.errors import IpdError

NotFound = trajectory_service.NotFound

_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(upload_id):
    """One lock per collection, created on demand.

    Per collection rather than one global lock so a long run on one upload does not block
    reads or computes on another. The guard exists only to make the dict insertion itself
    thread-safe.
    """
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(upload_id), threading.Lock())


# --- locating a frame -------------------------------------------------------


def _frame_entry(upload_id, slug, frame_index):
    """The stored frame payload at this animation position.

    Searched by ``frame_index`` rather than indexed by it. They agree today -- grouping assigns
    0..n-1 in order -- but the list position and the field carrying that number are two
    different claims, and only one of them is the identifier the URL names.
    """
    system = trajectory_service.get_trajectory(upload_id, slug)
    for frame in system["frames"]:
        if int(frame["frame_index"]) == int(frame_index):
            return system, frame
    raise NotFound(
        f"No frame {frame_index} in system {slug!r} of upload {upload_id!r}; "
        f"it has {len(system['frames'])} frames."
    )


def _row_label(df, frame):
    """The dataframe label for a stored frame's ``row_index`` position."""
    position = int(frame["row_index"])
    if not 0 <= position < len(df):
        raise NotFound(
            f"Frame {frame['frame_index']} refers to row {position}, which is outside the "
            f"stored dataframe's {len(df)} rows."
        )
    return df.index[position]


# --- reading ----------------------------------------------------------------


def get_history(upload_id, slug, frame_index, mode_id):
    """One frame's stored induced-dipole history for one damping mode."""
    ipd_results.mode_parameterization(mode_id)  # reject a bad mode before touching disk
    _, frame = _frame_entry(upload_id, slug, frame_index)
    df = system_serialization.load_processed(upload_id)
    return ipd_results.frame_ipd_history(df, _row_label(df, frame), mode_id)


# --- computing --------------------------------------------------------------


def _require_inputs(df, label, mode_id):
    """Raise unless this row carries finite values in every MBIS input the kernel reads.

    Checked here rather than left to the kernel so a missing column or a bare-ion ``[NaN]``
    volume ratio is a user-fixable input error naming the field, not an opaque 500 after a
    200-step all-NaN SCF run.
    """
    required = dataframe_schema.FEATURE_REQUIREMENTS["ipd_computable"]
    missing = sorted(column for column in required if column not in df.columns)
    if missing:
        raise IpdError(
            "ipd_inputs_missing",
            "This dataset cannot be used for an IPD calculation: it is missing "
            f"{', '.join(repr(column) for column in missing)}.",
            details={"missing_columns": missing, "mode": mode_id},
        )

    invalid = []
    for column in sorted(required):
        try:
            value = np.asarray(df.at[label, column], dtype=float)
        except (TypeError, ValueError):
            invalid.append(column)
            continue
        if value.size == 0 or not np.isfinite(value).all():
            invalid.append(column)
    if invalid:
        raise IpdError(
            "ipd_inputs_invalid",
            f"Row {label} cannot be used for an IPD calculation: required input(s) "
            f"{', '.join(repr(column) for column in invalid)} contain missing or "
            "non-finite values.",
            status=422,
            details={"invalid_columns": invalid, "mode": mode_id, "row_index": int(label)},
        )


def _compute_into(df, label, mode_id):
    """Run one row and one mode, writing the result into ``df`` in place."""
    parameterization = ipd_results.mode_parameterization(mode_id)
    _require_inputs(df, label, mode_id)
    try:
        result = thole_damping.compute_ipd_row(df.loc[label], **parameterization)
    except IpdError:
        raise
    except Exception as exc:
        # The SCF reaches into apnet_pt and numpy; a bad row surfaces as anything from a
        # ValueError on non-finite radii to a torch shape error. Reported as one code with the
        # original text kept, rather than a 500 that says nothing about which row failed.
        raise IpdError(
            "ipd_computation_failed",
            f"The IPD calculation failed for row {label}. {type(exc).__name__}: {exc}",
            status=500,
            details={"mode": mode_id, "row_index": int(label)},
            user_fixable=False,
        )
    thole_damping.write_ipd_row(df, label, result, **parameterization)


def _persist(upload_id, df):
    """Rewrite the collection from a modified dataframe, returning the fresh manifest.

    Re-runs ``save_collection`` rather than patching the affected frame's JSON in place. That
    costs a full rewrite of the collection's files, but it means there is exactly one code path
    producing a stored collection -- a second, partial one could disagree with it and the
    disagreement would be invisible.

    ``add_derived_columns`` is deliberately *not* re-run: ``processed.pkl`` already carries its
    output, and re-running it is the one way a contact distance could be recomputed from a
    geometry that has since been normalized.
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


def compute(upload_id, slug, frame_index, mode_id):
    """Run one frame and one mode, persist it, and return it ready to display.

    The response carries both the refreshed availability block and the new history, so the
    caller can go straight from "not computed" to showing arrows without a second request.
    """
    capability.require_capability()
    ipd_results.mode_parameterization(mode_id)
    _, frame = _frame_entry(upload_id, slug, frame_index)

    with _lock_for(upload_id):
        df = system_serialization.load_processed(upload_id)
        label = _row_label(df, frame)
        _compute_into(df, label, mode_id)
        _persist(upload_id, df)

        _, refreshed = _frame_entry(upload_id, slug, frame_index)
        return {
            "frame_index": int(frame_index),
            "ipd": refreshed["ipd"],
            "energies": refreshed.get("energies", {}),
            "history": ipd_results.frame_ipd_history(df, label, mode_id),
        }


def compute_system(upload_id, slug, mode_id):
    """Run one mode across every frame of a trajectory, persisting once at the end.

    The unit that matches how these are actually looked at: a scan is walked frame by frame,
    and computing them one request at a time would pay the collection rewrite once per frame.

    Every frame is rerun, including one that already carries this mode: this action is the
    explicit reprocessing path. A frame that fails is recorded and the rest still run. Because
    `_compute_into` writes only after a successful kernel call, a failed rerun leaves that
    frame's previous stored result intact.
    """
    capability.require_capability()
    ipd_results.mode_parameterization(mode_id)
    system = trajectory_service.get_trajectory(upload_id, slug)

    with _lock_for(upload_id):
        df = system_serialization.load_processed(upload_id)

        computed, failed = [], []
        for frame in system["frames"]:
            index = int(frame["frame_index"])
            label = _row_label(df, frame)
            try:
                _compute_into(df, label, mode_id)
            except IpdError as exc:
                failed.append({"frame_index": index, "code": exc.code, "error": exc.message})
                continue
            computed.append(index)

        if computed:
            _persist(upload_id, df)

        refreshed = trajectory_service.get_trajectory(upload_id, slug)
        return {
            "mode": mode_id,
            "label": ipd_results.mode_label(mode_id),
            "computed": computed,
            # Kept as an empty compatibility field for callers of the previous fill-missing
            # behavior; this endpoint no longer skips stored frames.
            "skipped": [],
            "failed": failed,
            # No histories: this is the refreshed summary that lights the timeline and updates
            # Plotly from the same in-memory trajectory. Histories remain lazy per frame/mode.
            "frames": [
                {
                    "frame_index": int(frame["frame_index"]),
                    "ipd": frame["ipd"],
                    "energies": frame.get("energies", {}),
                }
                for frame in refreshed["frames"]
            ],
        }
