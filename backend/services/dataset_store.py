"""Server-managed storage for user-supplied dimer dataframes.

Purely mechanical: identity, files, locking, atomic writes. It knows nothing about IPD,
MBIS columns or induced dipoles -- ``ipd_compute`` layers that on top. Keeping the split
means the write path can be tested without apnet_pt installed.

Layout, one directory per dataset::

    data/uploads/<dataset_id>/original.pkl   exactly as uploaded, never modified
    data/uploads/<dataset_id>/working.pkl    the primary record of computed IPD results
    data/uploads/<dataset_id>/meta.json      row map, fingerprints, compatibility report

**This module unpickles user-supplied files, which executes arbitrary code from them.**
That is a deliberate trade for a single-user application on localhost -- the dimer
dataframes carry ``qcelemental.Molecule`` objects and nested NumPy arrays that no safe
format round-trips without a conversion step. It is also why ``app.py`` binds to
127.0.0.1 rather than 0.0.0.0. The bundled-catalog path in ``dipole_data`` is unaffected
and still reads its ``.npz`` files with ``allow_pickle=False``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd

from backend.services.errors import IpdError

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

ORIGINAL_NAME = "original.pkl"
WORKING_NAME = "working.pkl"
META_NAME = "meta.json"

# Per-dataset write locks. Serializing on the dataset rather than globally lets two
# datasets be worked on at once while still making "load, modify, save" atomic within
# one. _LOCKS_GUARD only protects the dict itself, never a calculation.
_LOCKS: Dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()

# The most recently loaded working dataframe per dataset, keyed by the file's (mtime, size)
# so an edit made outside this process is still noticed. One request can otherwise re-read
# the same multi-megabyte pickle three or four times: the summary, the catalog and the
# calculation each want the frame.
#
# The cached object is the same frame a writer mutates in place, so it must only ever be
# handed to a mutating caller that holds `dataset_lock` and finishes with `save_working`.
_WORKING_CACHE: Dict[str, Tuple[Tuple[float, int], "pd.DataFrame"]] = {}


class DatasetNotFound(LookupError):
    """Raised for an unknown ``dataset_id``.

    A distinct type, mirroring ``dipole_data.SystemNotFound``, so the Flask layer can
    answer 404 without this module importing Flask.
    """


# --- identity and paths -----------------------------------------------------


def dataset_id_for(raw: bytes) -> str:
    """Content-derived id, so re-uploading an identical file re-opens the same dataset.

    That makes registration idempotent: the user who uploads the same pickle twice gets
    back the working copy with whatever IPD results they already computed, rather than a
    second directory that silently starts empty.
    """
    return "d-" + hashlib.sha256(raw).hexdigest()[:12]


def dataset_dir(dataset_id: str) -> Path:
    """Resolve a dataset directory, refusing anything that escapes ``data/uploads``.

    ``dataset_id`` reaches here straight from a URL path segment, so this is the boundary
    that stops ``../`` from reading or overwriting elsewhere on disk.
    """
    path = (UPLOAD_DIR / dataset_id).resolve()
    if path.parent != UPLOAD_DIR.resolve():
        raise DatasetNotFound(f"Invalid dataset id: {dataset_id!r}")
    return path


def _require_dir(dataset_id: str) -> Path:
    path = dataset_dir(dataset_id)
    if not (path / META_NAME).is_file():
        raise DatasetNotFound(f"Unknown dataset: {dataset_id}")
    return path


def working_path(dataset_id: str) -> Path:
    return _require_dir(dataset_id) / WORKING_NAME


def original_path(dataset_id: str) -> Path:
    return _require_dir(dataset_id) / ORIGINAL_NAME


def exists(dataset_id: str) -> bool:
    try:
        _require_dir(dataset_id)
    except DatasetNotFound:
        return False
    return True


# --- locking ----------------------------------------------------------------


@contextmanager
def dataset_lock(dataset_id: str) -> Iterator[None]:
    """Serialize writes to one dataset for the lifetime of the block."""
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(dataset_id, threading.Lock())
    with lock:
        yield


# --- reading and writing dataframes -----------------------------------------


def read_dataframe(path: Path) -> pd.DataFrame:
    """``pd.read_pickle`` with the failure modes turned into actionable errors."""
    try:
        df = pd.read_pickle(path)
    except ModuleNotFoundError as exc:
        # A pickled qcelemental.Molecule names the exact module its class came from, so a
        # dataframe written against one qcelemental build cannot be read by a server
        # running another -- including two builds that report the same version string.
        # Observed for real between two 0.29.0 installs, one with models/v1 and one
        # without. Worth its own message: the raw ModuleNotFoundError sends people
        # looking for a missing package rather than a version mismatch.
        missing = str(exc)
        hint = ""
        if "qcelemental" in missing:
            hint = (
                " This dataframe was written with a different qcelemental build than the "
                "server has. Register it from an environment whose qcelemental matches "
                "the one that created it, or re-save the dataframe there."
            )
        raise IpdError(
            "unreadable_dataset",
            f"Could not read the dataframe: {missing}.{hint}",
            details={"missing_module": missing},
        )
    except Exception as exc:
        raise IpdError(
            "unreadable_dataset",
            f"Could not read the file as a pandas pickle. {type(exc).__name__}: {exc}",
        )

    if not isinstance(df, pd.DataFrame):
        raise IpdError(
            "unreadable_dataset",
            f"The pickle contains a {type(df).__name__}, not a pandas DataFrame.",
        )
    if df.empty:
        raise IpdError("unreadable_dataset", "The dataframe contains no rows.")
    return df


def _stamp(path: Path) -> Tuple[float, int]:
    stat = path.stat()
    return (stat.st_mtime, stat.st_size)


def working_stamp(dataset_id: str) -> Tuple[float, int]:
    """Cheap change token for the working file, for callers that cache derived data."""
    return _stamp(working_path(dataset_id))


def load_working(dataset_id: str) -> pd.DataFrame:
    """The current working dataframe, the primary record of computed IPD results."""
    path = working_path(dataset_id)
    stamp = _stamp(path)

    cached = _WORKING_CACHE.get(dataset_id)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    df = read_dataframe(path)
    _WORKING_CACHE[dataset_id] = (stamp, df)
    return df


def invalidate_cache(dataset_id: str) -> None:
    """Forget the cached working frame, forcing the next load to re-read from disk."""
    _WORKING_CACHE.pop(dataset_id, None)


def save_working(dataset_id: str, df: pd.DataFrame) -> None:
    """Persist the working dataframe, atomically.

    Serialize to a temporary file *in the same directory* -- so the final ``os.replace``
    is a same-filesystem rename and therefore atomic -- and only then swap it in. A
    failure part-way through pickling leaves the previous good working.pkl untouched
    rather than a truncated file that would fail every later read.
    """
    target = working_path(dataset_id)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        df.to_pickle(tmp)
        os.replace(tmp, target)
        # Re-key the freshly written frame rather than dropping it: this is the object the
        # caller just mutated, so it is already the correct contents for the new file.
        _WORKING_CACHE[dataset_id] = (_stamp(target), df)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        invalidate_cache(dataset_id)
        raise IpdError(
            "persist_failed",
            f"Could not save the updated dataframe. {type(exc).__name__}: {exc}",
            status=500,
            retryable=True,
            user_fixable=False,
        )


# --- row identity -----------------------------------------------------------


def row_fingerprint(row) -> str:
    """A stable digest of the row's identity and geometry.

    Compared before every write so a request built against an older view of the dataset
    cannot silently land on a different geometry. Covers the two things that decide
    *which* calculation this is -- the label and the coordinates -- and deliberately not
    the IPD result columns, which are expected to change.
    """
    digest = hashlib.sha256()
    digest.update(str(row.get("system_id", "")).encode("utf-8"))

    molecule = row.get("qcel_molecule")
    geometry = getattr(molecule, "geometry", None)
    if geometry is not None:
        coords = np.ascontiguousarray(np.round(np.asarray(geometry, dtype=float), 6))
        digest.update(str(coords.shape).encode("utf-8"))
        digest.update(coords.tobytes())
    return digest.hexdigest()[:16]


def row_id_for(position: int) -> str:
    """Server-assigned row identifier.

    The positional index of the row in the working dataframe, zero-padded so ids sort
    lexicographically. Stable because the server owns working.pkl and never reorders or
    filters it -- and any doubt is settled by the fingerprint check, not by this string.
    """
    return f"{position:06d}"


# --- metadata ---------------------------------------------------------------


def read_meta(dataset_id: str) -> Dict[str, Any]:
    path = _require_dir(dataset_id) / META_NAME
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise IpdError(
            "unreadable_dataset",
            f"Dataset metadata is corrupt: {exc}",
            status=500,
            user_fixable=False,
        )


def write_meta(dataset_id: str, meta: Dict[str, Any]) -> None:
    """Write meta.json atomically, for the same reason save_working does."""
    target = dataset_dir(dataset_id) / META_NAME
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n")
    os.replace(tmp, target)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- registration and listing -----------------------------------------------


def store_upload(raw: bytes, filename: str) -> Tuple[str, Path, bool]:
    """Write the upload to its own directory. Returns ``(dataset_id, dir, is_new)``.

    Does not read the dataframe or build metadata -- ``ipd_compute.register`` does that,
    so this module stays free of any IPD knowledge.
    """
    if not raw:
        raise IpdError("unreadable_dataset", "The uploaded file is empty.")

    dataset_id = dataset_id_for(raw)
    directory = dataset_dir(dataset_id)
    is_new = not (directory / META_NAME).is_file()

    directory.mkdir(parents=True, exist_ok=True)
    original = directory / ORIGINAL_NAME
    if not original.is_file():
        original.write_bytes(raw)
    return dataset_id, directory, is_new


def restore_original(dataset_id: str) -> None:
    """Discard every computed result by reinstating the uploaded dataframe.

    The file-level primitive only. It restores ``original.pkl`` verbatim, which is *not* what
    a working copy looks like -- that one is canonically indexed and carries derived metadata
    columns. ``ipd_compute.restore`` is the operation the API exposes; it re-applies both.
    """
    directory = _require_dir(dataset_id)
    tmp = directory / (WORKING_NAME + ".tmp")
    shutil.copyfile(directory / ORIGINAL_NAME, tmp)
    os.replace(tmp, directory / WORKING_NAME)
    # The file was replaced behind the cache's back rather than through save_working.
    invalidate_cache(dataset_id)


def list_datasets() -> List[Dict[str, Any]]:
    """Summary of every registered dataset, newest first."""
    if not UPLOAD_DIR.is_dir():
        return []

    summaries: List[Dict[str, Any]] = []
    for directory in UPLOAD_DIR.iterdir():
        if not (directory / META_NAME).is_file():
            continue
        try:
            meta = json.loads((directory / META_NAME).read_text())
        except json.JSONDecodeError:
            continue
        summaries.append(
            {
                "dataset_id": meta.get("dataset_id", directory.name),
                "name": meta.get("name", directory.name),
                "registered_at": meta.get("registered_at"),
                "n_rows": meta.get("n_rows", 0),
                "n_systems": meta.get("n_systems", 0),
                "compatibility": meta.get("compatibility", {}),
            }
        )
    summaries.sort(key=lambda entry: entry.get("registered_at") or "", reverse=True)
    return summaries
