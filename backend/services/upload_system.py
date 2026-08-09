"""The upload workflow, top level.

The orchestration layer, and the only place that knows the order of the pipeline::

    raw bytes
      -> collection_id_for          content hash, so the same file re-opens the same collection
      -> (stored at this schema?    return it untouched)
      -> (stored at an older one?   _reserialize from processed.pkl, keeping computed results)
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


def _reserialize(collection_id, stored):
    """Rewrite a stale-schema collection from ``processed.pkl``, or None if that is not there.

    The reason a schema bump does not simply fall through to reprocessing the upload: results
    computed *since* registration live in ``processed.pkl`` and in no uploaded file. Re-reading
    the bytes the user just handed over would silently discard every IPD run they have paid
    for -- with no error, and nothing afterwards to say it happened.

    ``add_derived_columns`` is deliberately not re-run, for the reason ``ipd_service._persist``
    documents: the stored frame already carries its output, and re-running it is the one way a
    contact distance could be recomputed from a geometry that has since been normalized.

    Returns None rather than raising when the pickle is missing or unreadable, leaving the
    caller to fall back to a full reprocess -- a stale collection is worth rescuing, but not at
    the cost of refusing an upload that would otherwise work.
    """
    try:
        df = system_serialization.load_processed(collection_id)
    except (OSError, ValueError, EOFError, AttributeError, ImportError):
        return None

    return system_serialization.save_collection(
        collection_id,
        df,
        system_processing.group_trajectories(df),
        source_filename=stored["source_filename"],
        display_name=stored.get("display_name"),
        validation=stored.get("validation"),
    )


def process_upload(raw, filename):
    """Register an uploaded dataframe and return its manifest.

    Idempotent by content: the same bytes give the same ``collection_id``, and an already
    stored collection is returned untouched rather than rewritten.

    Three outcomes, not two. A stored collection at the current ``schema_version`` is returned
    as it stands; one at an older version is **re-serialized from its own ``processed.pkl``**,
    which is what makes a schema bump take effect without throwing away computed results; and
    anything not stored at all is processed from the uploaded bytes.

    Raises :class:`UploadError` when there is nothing worth storing -- an unreadable file, no
    geometry column, or no row carrying a usable geometry.
    """
    collection_id = collection_id_for(raw)
    try:
        stored = system_serialization.load_manifest(collection_id)
        if stored.get("schema_version") == dataframe_schema.SCHEMA_VERSION:
            return stored
        migrated = _reserialize(collection_id, stored)
        if migrated is not None:
            return migrated
    except (FileNotFoundError, ValueError, KeyError):
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
