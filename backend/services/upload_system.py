"""The upload workflow, top level.

The orchestration layer, and the only place that knows the order of the pipeline::

    raw bytes
      -> collection_id_for          content hash, so the same file re-opens the same collection
      -> load_uploaded_dataframe    unpickled exactly once
      -> normalize_columns          aliases to canonical names
      -> validate                   structured report
      -> drop invalid rows
      -> add_derived_columns        contact distance, eq_ratio
      -> group_trajectories         grouping, ordering, frame index
      -> save_collection            processed.pkl, systems/*.json, manifest.json
      -> manifest

"""

from __future__ import annotations

import hashlib
import io

import pandas as pd

from backend.services import (
    dataframe_schema,
    dataframe_validation,
    system_processing,
    system_serialization,
)


class UploadError(Exception):
    """An upload that cannot be stored, carrying the report that says why.

    ``report`` has the same shape ``dataframe_validation.validate`` returns, so a caller
    renders one thing whether the failure was an unreadable file or a missing column.
    """

    def __init__(self, report):
        super().__init__(report["message"])
        self.report = report


def failure_report(code, message):
    return {
        "ok": False,
        "code": code,
        "message": message,
        "missing_columns": [],
        "total_rows": 0,
        "valid_rows": 0,
        "invalid_rows": [],
    }


def collection_id_for(raw):
    """Content-derived id, so re-uploading an identical file re-opens the same collection."""
    return "c-" + hashlib.sha256(raw).hexdigest()[:12]


def load_uploaded_dataframe(raw):
    """``pd.read_pickle`` over the uploaded bytes, with the failure modes made actionable."""
    try:
        df = pd.read_pickle(io.BytesIO(raw))
    except ModuleNotFoundError as exc:
        # A pickled qcelemental.Molecule names the exact module its class came from, so a
        # dataframe written against one qcelemental build cannot be read by a server running
        # another -- including two builds reporting the same version string. Worth its own
        # message: the raw ModuleNotFoundError sends people looking for a missing package
        # rather than a version mismatch.
        hint = ""
        if "qcelemental" in str(exc):
            hint = (
                " This dataframe was written with a different qcelemental build than the "
                "server has. Upload it from an environment whose qcelemental matches the "
                "one that created it, or re-save the dataframe there."
            )
        raise UploadError(failure_report("unreadable_dataset", f"Could not read the file: {exc}.{hint}"))
    except Exception as exc:
        raise UploadError(
            failure_report(
                "unreadable_dataset",
                f"Could not read the file as a pandas pickle. {type(exc).__name__}: {exc}",
            )
        )

    if not isinstance(df, pd.DataFrame):
        raise UploadError(
            failure_report(
                "unreadable_dataset",
                f"The pickle contains a {type(df).__name__}, not a pandas DataFrame.",
            )
        )
    if df.empty:
        raise UploadError(failure_report("unreadable_dataset", "The dataframe contains no rows."))
    return df


def process_upload(raw, filename):
    """Register an uploaded dataframe and return its manifest.

    Idempotent by content: the same bytes give the same ``collection_id``, and an already
    stored collection is returned untouched rather than rewritten. A stored collection whose
    ``schema_version`` no longer matches is reprocessed, which is what makes a schema bump
    take effect.

    Raises :class:`UploadError` when there is nothing worth storing -- an unreadable file, no
    geometry column, or no row carrying a usable geometry.
    """
    collection_id = collection_id_for(raw)
    try:
        stored = system_serialization.load_manifest(collection_id)
        if stored.get("schema_version") == dataframe_schema.SCHEMA_VERSION:
            return stored
    except (FileNotFoundError, ValueError):
        pass

    df = load_uploaded_dataframe(raw)
    # One canonical ordering, fixed here. Every row_index downstream is a position in this
    # frame, and nothing reorders it afterwards.
    df = dataframe_validation.normalize_columns(df).reset_index(drop=True)

    report = dataframe_validation.validate(df)
    if not report["ok"]:
        raise UploadError(report)

    # Dropped rather than carried: validate's message already promises these will not be
    # shown, and a frame with no readable geometry has nothing to contribute to a trajectory.
    # The indices are recorded in the manifest so the loss is visible.
    dropped = [entry["row_index"] for entry in report["invalid_rows"]]
    if dropped:
        df = df.drop(index=dropped).reset_index(drop=True)

    system_processing.add_derived_columns(df)
    trajectories = system_processing.group_trajectories(df)

    return system_serialization.save_collection(
        collection_id,
        df,
        trajectories,
        source_filename=filename,
        validation={
            "total_rows": report["total_rows"],
            "valid_rows": report["valid_rows"],
            "dropped_rows": dropped,
        },
    )
