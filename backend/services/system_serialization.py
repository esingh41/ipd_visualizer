"""What gets stored, and what gets sent to the frontend.

One directory per collection::

    data/systems/<collection-id>/
        manifest.json           collection metadata and feature availability
        processed.pkl           the normalized dataframe, for backend use
        systems/<slug>.json     one file per trajectory, never one combined file

**The backend artifact is a pickle, not Parquet.** Arrow cannot represent this data: a
``qcelemental.Molecule`` cell fails outright, and so does any cell holding a 2-D or 3-D array
-- which covers ``mu hf/adz A``, ``theta hf/adz A`` and every ``mu_ind hist`` result, 41 of
``na_water_dimers.pkl``'s 181 columns. Only 1-D arrays survive. A pickle preserves dtypes,
array shapes and missing values exactly, and the upload it came from was already a pickle, so
this adds no trust boundary that did not exist.

The collection id is supplied by the caller: identity belongs to whoever registers an upload,
not to the code that writes its files.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backend.services import dataframe_schema, system_processing

COLLECTIONS_DIR = Path(__file__).resolve().parents[2] / "data" / "systems"

MANIFEST_NAME = "manifest.json"
PROCESSED_NAME = "processed.pkl"
SYSTEMS_DIRNAME = "systems"

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def collection_dir(collection_id):
    """Where one collection's files live, refusing anything that escapes COLLECTIONS_DIR.

    A collection id reaches here straight from a URL path segment, so this is the boundary
    that stops ``../`` from reading or overwriting elsewhere on disk.
    """
    path = (COLLECTIONS_DIR / str(collection_id)).resolve()
    if path.parent != COLLECTIONS_DIR.resolve():
        raise ValueError(f"Invalid collection id: {collection_id!r}")
    return path


# --- conversion -------------------------------------------------------------


def _jsonable(value):
    """A JSON-safe version of a cell: numpy out, None for anything non-finite.

    ``json.dump`` emits a bare ``NaN`` by default, which no strict JSON parser accepts.
    ``eq_ratio`` is genuinely NaN wherever no ratio was recoverable, so non-finite floats
    become null here and the writer passes ``allow_nan=False`` to make a miss loud rather
    than silently producing a file the frontend cannot read.
    """
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, bool)):
        return value
    return str(value)


def slugify(name):
    """A filesystem-safe filename stem for a trajectory name.

    Trajectory names come from user data -- "Na+-Water", "Cl-Water", "x/y", "row 0" -- so
    they cannot be used as paths directly. Accents are folded, everything outside
    ``[A-Za-z0-9._-]`` collapses to a single underscore.
    """
    folded = unicodedata.normalize("NFKD", str(name))
    folded = folded.encode("ascii", "ignore").decode("ascii")
    slug = _UNSAFE.sub("_", folded).strip("._-")
    return slug or "system"


def _unique_slugs(names):
    """``{name: slug}``, disambiguated so two names never claim one file.

    Slugging is lossy -- "Na+-Water" and "Na_-Water" fold together -- so a collision would
    silently overwrite one trajectory with another.
    """
    slugs = {}
    taken = set()
    for name in names:
        base = slugify(name)
        slug = base
        counter = 2
        while slug in taken:
            slug = f"{base}-{counter}"
            counter += 1
        taken.add(slug)
        slugs[name] = slug
    return slugs


# --- writing ----------------------------------------------------------------


def _write_atomic(path, write):
    """Write via a temp file in the destination directory, then ``os.replace``.

    Same directory so the replace is a rename within one filesystem, which is what makes it
    atomic. A crash leaves the previous file intact rather than a half-written one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        write(temp)
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _write_json(path, payload):
    def write(temp):
        with open(temp, "w", encoding="utf-8") as handle:
            # allow_nan=False turns a missed conversion into an exception rather than a file
            # that looks fine until something tries to parse it.
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")

    _write_atomic(path, write)


def _expected_shape(quantity, n_atoms):
    """What a valid array for one quantity looks like, once A and B are joined."""
    # A dipole is a vector per atom; a charge and a volume ratio are one number per atom.
    return (n_atoms, 3) if quantity == "dipoles" else (n_atoms,)


def _atom_array(value, shape):
    """One column's cell as a float array of ``shape``, or None if it is not one.

    ``np.atleast_1d`` is load-bearing: ``volume ratios A`` in ``131_Na-benzene.pkl`` is a 0-d
    scalar in every row, because monomer A there is a single Na atom, and a 0-d array cannot
    be concatenated at all.
    """
    if value is None:
        return None
    try:
        array = np.atleast_1d(np.asarray(value, dtype=float))
    except (TypeError, ValueError):
        return None
    return array if array.shape == shape else None


def _monomer_array(row, columns, shape):
    """The two isolated-monomer arrays joined into one atom-ordered array, or None.

    Concatenating A then B is what aligns element-wise with ``symbols`` and ``coords``: atom
    ``i`` is in A iff ``i < n_atoms_A``, so monomer values must arrive in that same order.
    """
    parts = []
    for column, n_atoms in columns:
        part = _atom_array(row.get(column), (n_atoms,) + shape[1:])
        if part is None:
            return None
        parts.append(part)
    return np.concatenate(parts, axis=0)


def _multipole_arrays(row, n_atoms, n_atoms_A):
    """Per-atom MBIS multipoles for one frame, isolated-monomer beside in-dimer.

    Every array is checked against the atom count before it is emitted, and a quantity that
    does not measure up is reported missing rather than raised on. A wrong-length array would
    otherwise render as a table of plausible numbers attached to the wrong atoms.

    Missing data is reported at two levels: a quantity is None when neither side could be
    assembled, and ``monomer`` or ``dimer`` alone is None when only one side exists.
    """
    n_atoms_B = n_atoms - n_atoms_A
    multipoles = {}
    for quantity, columns in dataframe_schema.MULTIPOLE_QUANTITIES.items():
        shape = _expected_shape(quantity, n_atoms)
        monomer = _monomer_array(
            row, ((columns["A"], n_atoms_A), (columns["B"], n_atoms_B)), shape
        )
        dimer = _atom_array(row.get(columns["dimer"]), shape)
        multipoles[quantity] = (
            None
            if monomer is None and dimer is None
            else {"monomer": _jsonable(monomer), "dimer": _jsonable(dimer)}
        )
    return multipoles


def _frame_payload(frame, row):
    """One frame of a trajectory, as the frontend receives it."""
    payload = {
        "row_index": int(frame["row_index"]),
        "system_id": frame["system_id"],
        "frame_index": int(frame["frame_index"]),
        "separation": _jsonable(frame["separation"]),
        "separation_units": frame["separation_units"],
        "separation_label": frame["separation_label"],
        "contact_distance_ang": _jsonable(row.get(system_processing.CONTACT_COLUMN)),
        "eq_ratio": _jsonable(row.get(system_processing.EQ_RATIO_COLUMN)),
    }

    geometry = system_processing.geometry_fields(row.get("qcel_molecule"))
    if geometry is None:
        # A molecule validation would have flagged. Written as nulls rather than dropped:
        # a frame vanishing from the middle of a trajectory is harder to notice than one
        # that says it has no geometry.
        payload.update(
            {"symbols": None, "n_atoms": None, "n_atoms_A": None, "coords": None, "xyz": None}
        )
        # No atom count to validate multipole lengths against, so there is nothing that can
        # be emitted honestly.
        payload["multipoles"] = None
    else:
        payload.update(
            {
                "symbols": geometry["symbols"],
                "n_atoms": int(geometry["n_atoms"]),
                "n_atoms_A": int(geometry["n_atoms_A"]),
                "coords": _jsonable(geometry["coords"]),
                "xyz": geometry["xyz"],
            }
        )
        payload["multipoles"] = _multipole_arrays(
            row, int(geometry["n_atoms"]), int(geometry["n_atoms_A"])
        )
    return payload


def save_collection(
    collection_id, df, trajectories, *, source_filename, display_name=None, validation=None
):
    """Write a processed upload to disk and return its manifest.

    ``df`` is the normalized frame, ``trajectories`` the output of
    ``system_processing.group_trajectories``. Every frame handed over is written -- deciding
    which rows deserve to be here belongs to the caller.

    ``validation`` is recorded in the manifest as-is. A caller that drops rows before
    getting here should pass one, since a dropped row is otherwise invisible.
    """
    directory = collection_dir(collection_id)
    features = dataframe_schema.feature_availability(df.columns)
    columns = {dataframe_schema.canonical_name(name) for name in df.columns}
    slugs = _unique_slugs(trajectory["name"] for trajectory in trajectories)

    systems = []
    for trajectory in trajectories:
        name = trajectory["name"]
        frames = [
            _frame_payload(frame, df.iloc[frame["row_index"]])
            for frame in trajectory["frames"]
        ]
        relative = f"{SYSTEMS_DIRNAME}/{slugs[name]}.json"
        _write_json(
            directory / relative,
            {
                "collection_id": str(collection_id),
                "system": name,
                "n_frames": len(frames),
                "features": features,
                "frames": frames,
            },
        )
        systems.append({"name": name, "file": relative, "n_frames": len(frames)})

    manifest = {
        "collection_id": str(collection_id),
        "display_name": display_name or Path(source_filename).stem,
        "schema_version": dataframe_schema.SCHEMA_VERSION,
        "source_filename": source_filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system_count": len(systems),
        "frame_count": sum(entry["n_frames"] for entry in systems),
        "available_fields": sorted(dataframe_schema.KNOWN_COLUMNS & columns),
        "missing_optional_fields": sorted(dataframe_schema.KNOWN_COLUMNS - columns),
        "features": features,
        "validation": validation,
        "systems": systems,
    }

    _write_atomic(directory / PROCESSED_NAME, lambda temp: df.to_pickle(temp))
    # Manifest last: it is what marks a directory as a complete collection, so a crash
    # part-way leaves one that is ignored rather than one that half-resolves.
    _write_json(directory / MANIFEST_NAME, manifest)
    return manifest


# --- reading ----------------------------------------------------------------


def load_manifest(collection_id):
    """A collection's manifest."""
    path = collection_dir(collection_id) / MANIFEST_NAME
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_system(collection_id, name):
    """One trajectory's frames, looked up by trajectory name.

    Reads the manifest's name-to-file mapping rather than re-deriving the slug, so the
    disambiguation applied at write time cannot be guessed at differently here.
    """
    manifest = load_manifest(collection_id)
    for entry in manifest["systems"]:
        if entry["name"] == name:
            path = collection_dir(collection_id) / entry["file"]
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
    raise KeyError(f"No system named {name!r} in collection {collection_id!r}")


def load_processed(collection_id):
    """The normalized dataframe, for backend work that needs the full fidelity."""
    return pd.read_pickle(collection_dir(collection_id) / PROCESSED_NAME)


def list_collections():
    """Every stored collection's manifest, newest first.

    A directory with no readable manifest is skipped rather than raising: manifest.json is
    written last, so one half-written or corrupt collection must not empty the whole list.
    """
    if not COLLECTIONS_DIR.is_dir():
        return []

    manifests = []
    for directory in COLLECTIONS_DIR.iterdir():
        if not directory.is_dir():
            continue
        try:
            manifests.append(load_manifest(directory.name))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    manifests.sort(key=lambda manifest: manifest.get("created_at", ""), reverse=True)
    return manifests
