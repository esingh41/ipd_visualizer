"""Row-level Delta-MTP science adapter.

This module knows the ``apnet_pt`` kernel's data shape, but knows nothing about uploads,
collection ids, Flask, files, or persistence. Importing ``apnet_pt`` is deliberately lazy so
ordinary trajectory browsing does not pay the PyTorch import cost and can still run when the
optional science dependency is unavailable.
"""

from __future__ import annotations

import numpy as np

from backend.services.dataframe_schema import DELTA_MTP_OUTPUT_COLUMNS


def _kernel():
    """Load the optional multipole kernel only when a calculation is requested."""
    from apnet_pt.multipole import multipoles_elst_ind_dimer

    return multipoles_elst_ind_dimer


def _multipoles(row, suffix):
    """The q/mu/theta tuple expected by ``multipoles_elst_ind_dimer``."""
    return (
        np.asarray(row[f"q hf/adz {suffix}"], dtype=float),
        np.asarray(row[f"mu hf/adz {suffix}"], dtype=float),
        np.asarray(row[f"theta hf/adz {suffix}"], dtype=float),
    )


def _one_finite_result(values, name):
    """Turn one kernel result sequence into a finite scalar, rejecting surprises."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Delta-MTP output {name!r} is not numeric: {exc}") from exc
    if array.size != 1:
        raise ValueError(
            f"Delta-MTP output {name!r} contains {array.size} values; expected exactly one."
        )
    value = float(array.reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError(f"Delta-MTP output {name!r} is non-finite.")
    return value


def compute_delta_mtp_row(row, *, kernel=None):
    """Compute all four finite Delta-MTP outputs for one dataframe row.

    The kernel is batch-shaped, so one row is wrapped in one-element lists. This function is
    pure with respect to the row and dataframe: it returns a complete result dictionary and
    writes nothing. Callers can therefore leave an old stored result untouched if import,
    input, kernel, or numeric-output validation fails.
    """
    kernel = kernel or _kernel()
    electrostatics, dimer_electrostatics, induction = kernel(
        [row["qcel_molecule"]],
        [_multipoles(row, "dimer")],
        [_multipoles(row, "A")],
        [_multipoles(row, "B")],
    )

    mtp_elst = _one_finite_result(electrostatics, "mtp elst")
    mtp_elst_dimer = _one_finite_result(dimer_electrostatics, "mtp elst dimer")
    mtp_ind = _one_finite_result(induction, "mtp ind")
    result = {
        "mtp elst": mtp_elst,
        "mtp elst dimer": mtp_elst_dimer,
        "mtp ind": mtp_ind,
        "mtp ind / 2": mtp_ind / 2.0,
    }
    # Keep this assertion close to the adapter: adding an output in schema without teaching
    # the science writer how to produce it must fail before persistence.
    if tuple(result) != tuple(DELTA_MTP_OUTPUT_COLUMNS):
        raise RuntimeError("Delta-MTP output definitions and science adapter are out of sync.")
    return result


def write_delta_mtp_row(df, label, result):
    """Write one already-successful complete result into a dataframe in place."""
    missing = [column for column in DELTA_MTP_OUTPUT_COLUMNS if column not in result]
    if missing:
        raise ValueError(f"Incomplete Delta-MTP result; missing {missing!r}.")

    # Validate every value before the first assignment. A bad result never partly replaces an
    # older four-column result.
    values = {
        column: _one_finite_result([result[column]], column)
        for column in DELTA_MTP_OUTPUT_COLUMNS
    }
    for column, value in values.items():
        df.at[label, column] = value
