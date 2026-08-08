"""Is this uploaded dataframe internally valid?

Reports, never raises. A frame with twelve good rows and two broken ones comes back with
``valid_rows: 12`` and the two named -- refusing the whole upload over a couple of malformed
geometries throws away everything that would have worked. Whether a report becomes an upload
error is ``upload_system``'s decision, not this module's.

The check that earns this module its place is fragment contiguity. ``n_atoms_A`` is the only
encoding of monomer membership downstream, so every consumer slices ``coords[:n_atoms_A]``
for monomer A. Fragments like ``[[0, 2, 4], [1, 3, 5]]`` have the right *count* and the wrong
*membership*: nothing would fail, and every coordinate, symbol and multipole would be
attributed to the wrong monomer. ``system_processing`` documents this as an assumption it
does not verify. Here is where it is verified.

Depends on ``dataframe_schema`` for the column vocabulary. Deliberately does not import
``system_processing`` -- the flow is schema -> validation -> processing, and re-deriving
``molecule.fragments`` here is two lines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.services import dataframe_schema

GEOMETRY_COLUMN = "qcel_molecule"


def normalize_columns(df):
    """A copy of the frame with recognised aliases renamed to their canonical names."""
    return df.rename(columns=dataframe_schema.rename_map(df.columns))


def validate_row(row):
    """``(code, message)`` describing what is wrong with a row, or None if it is fine."""
    molecule = row.get(GEOMETRY_COLUMN)
    if molecule is None or (isinstance(molecule, float) and np.isnan(molecule)):
        return ("missing_molecule", f"No {GEOMETRY_COLUMN!r} value.")

    found = getattr(molecule, "fragments", None)
    if found is None or len(found) != 2:
        count = "none" if found is None else len(found)
        return (
            "not_a_dimer",
            f"{GEOMETRY_COLUMN!r} must describe a dimer with exactly 2 fragments; "
            f"found {count}.",
        )

    try:
        symbols = list(molecule.symbols)
        geometry = np.asarray(molecule.geometry, dtype=float)
    except (AttributeError, TypeError, ValueError) as exc:
        return ("geometry_mismatch", f"Could not read the molecule's geometry: {exc}.")

    n_atoms = len(symbols)
    if geometry.ndim != 2 or geometry.shape != (n_atoms, 3):
        return (
            "geometry_mismatch",
            f"Geometry has shape {tuple(geometry.shape)}, but the molecule has {n_atoms} "
            f"atoms and needs {(n_atoms, 3)}.",
        )

    # The whole point of this module. Atom i is in monomer A iff i < n_atoms_A, so the
    # fragments must partition the atoms in that same contiguous order -- a fragment list
    # with the right sizes but interleaved indices would attribute every atom to the wrong
    # monomer without anything failing.
    fragment_a = np.asarray(found[0])
    fragment_b = np.asarray(found[1])
    n_atoms_a = len(fragment_a)

    if not np.array_equal(fragment_a, np.arange(n_atoms_a)):
        return (
            "fragments_not_contiguous",
            f"Monomer A's atoms must be the first {n_atoms_a} in {GEOMETRY_COLUMN!r}, "
            f"contiguous and in order; got {fragment_a.tolist()}.",
        )
    if not np.array_equal(fragment_b, np.arange(n_atoms_a, n_atoms)):
        return (
            "fragments_not_contiguous",
            f"Monomer B's atoms must immediately follow monomer A's; expected "
            f"{list(range(n_atoms_a, n_atoms))}, got {fragment_b.tolist()}.",
        )

    return None


def validate(df):
    """Everything wrong with an uploaded frame, as one structured report.

    ``ok`` means the upload is worth registering: the required columns are present *and* at
    least one row survives every geometry check. A frame whose ``qcel_molecule`` column
    exists but holds nothing usable has no geometry to show, so it is not ``ok`` even though
    no column is missing -- the two cases carry different codes.

    ``invalid_rows`` lists what failed; it never filters. Deciding what to do about a partly
    broken frame belongs to the caller.
    """
    missing = sorted(dataframe_schema.REQUIRED_COLUMNS - set(df.columns))
    total_rows = int(len(df))

    if missing:
        return {
            "ok": False,
            "code": "missing_geometry",
            "message": (
                f"This dataframe is missing {', '.join(repr(c) for c in missing)}, so there "
                f"is no geometry to display. Every upload needs one."
            ),
            "missing_columns": missing,
            "total_rows": total_rows,
            "valid_rows": 0,
            "invalid_rows": [],
        }

    invalid_rows = []
    for position, (_, row) in enumerate(df.iterrows()):
        problem = validate_row(row)
        if problem is not None:
            code, message = problem
            invalid_rows.append(
                {"row_index": position, "code": code, "message": message}
            )

    valid_rows = total_rows - len(invalid_rows)
    if not valid_rows:
        return {
            "ok": False,
            "code": "no_valid_rows",
            "message": (
                f"None of the {total_rows} rows carry a usable geometry, so there is "
                f"nothing to display."
            ),
            "missing_columns": [],
            "total_rows": total_rows,
            "valid_rows": 0,
            "invalid_rows": invalid_rows,
        }

    message = f"Valid upload: {valid_rows} of {total_rows} rows carry a usable geometry."
    if invalid_rows:
        message += f" {len(invalid_rows)} row(s) were flagged and will not be shown."

    return {
        "ok": True,
        "code": None,
        "message": message,
        "missing_columns": [],
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
    }
