"""Stored IPD results, translated into what the browser draws.

The translation layer between stored scientific results and the browser::

    dataframe_schema  -> uploaded input vocabulary
    thole_damping     -> IPD result vocabulary and computation
    this module       -> translate stored IPD results into frontend-friendly JSON

Pure translation. It reads a dataframe and returns dictionaries; it opens no files, imports no
Flask, and runs no science. That is what lets ``system_serialization`` import it to attach IPD
metadata to a frame without a cycle back through the module that saves collections.

Two levels of detail, deliberately separate, because induced-dipole history scales as
frames x modes x iterations x atoms x 3:

:func:`frame_ipd_metadata`
    Per frame, small enough to ride along in every trajectory payload: what exists, what could
    be computed, iteration counts, and the maximum magnitude needed for common arrow scaling.

:func:`frame_ipd_history`
    One frame, one mode, the whole SCF trajectory. Lazy-loaded, and for a run that hit the
    200-iteration cap on a 13-atom dimer it is ~150 KB on its own.

**Only ``thole_damping``'s column names are read.** Uploaded pickles may carry histories under
older names -- ``IPD (MBIS) 0.39 all`` and friends -- and those are ignored on purpose. Every history this app shows is one it computed and filed under a name that states
its own parameterization, so a displayed result can always be traced to the choices that made
it.
"""

from __future__ import annotations

import numpy as np

from backend.services import system_processing, thole_damping
from backend.services.errors import IpdError

# The damping modes the UI offers: universal Thole 0.39, or the TS-vdW radius combination
# rule on the intermolecular edges only.
#
# The single place a mode id, its label and its thole_damping arguments meet. `parameterization`
# splats into ipd_column_names, has_ipd_result, read_ipd_row, write_ipd_row and compute_ipd_row
# alike, so a mode cannot be read under one parameterization and written under another.
#
# thole_damping supports two further combinations (ts_mbis_radii on intra, and on both). Each
# would be one entry here; they are left out because the spec asks for these two.
MODES = {
    "thole_0.39": {
        "label": "Thole 0.39 (inter + intra)",
        "parameterization": {
            "inter_thole_parameterization": "fixed_0.39",
            "intra_thole_parameterization": "fixed_0.39",
        },
    },
    "ts_mbis_inter": {
        "label": "Thole TS-vdW radii (inter only)",
        "parameterization": {
            "inter_thole_parameterization": "ts_mbis_radii",
            "intra_thole_parameterization": "fixed_0.39",
        },
    },
}


def mode_parameterization(mode_id):
    """The ``thole_damping`` keyword arguments for a mode id, or an IpdError for a bad one."""
    try:
        return MODES[mode_id]["parameterization"]
    except KeyError:
        raise IpdError(
            "unknown_ipd_mode",
            f"Unknown IPD mode {mode_id!r}. Expected one of {sorted(MODES)}.",
            details={"allowed": sorted(MODES), "received": mode_id},
        )


def mode_label(mode_id):
    return MODES[mode_id]["label"]


# --- shape agreement --------------------------------------------------------


def _finite(value):
    """A float, or None when it is not finite. json.dump(allow_nan=False) rejects bare NaN."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _history_problem(hist_A, hist_B, n_atoms, n_atoms_A):
    """Why this stored history does not describe this geometry, or None if it does.

    Checked before the two halves are joined, and checked at *both* levels -- metadata reports
    the problem, history refuses to serve. A history whose atom counts disagree with the
    molecule would concatenate into an array of the right rank and the wrong meaning, drawing
    arrows on atoms they do not belong to. Nothing downstream could notice: the numbers are
    real, they are just attached to the wrong nuclei.
    """
    if hist_A.ndim != 3 or hist_B.ndim != 3:
        return (
            f"stored history has rank {hist_A.ndim}/{hist_B.ndim}, expected 3 "
            "(iteration, atom, xyz)"
        )
    if hist_A.shape[0] != hist_B.shape[0]:
        return (
            f"monomer histories have different lengths: A has {hist_A.shape[0]} "
            f"iterations, B has {hist_B.shape[0]}"
        )
    if hist_A.shape[2] != 3 or hist_B.shape[2] != 3:
        return "stored history vectors are not 3-component"

    n_atoms_B = n_atoms - n_atoms_A
    if hist_A.shape[1] != n_atoms_A or hist_B.shape[1] != n_atoms_B:
        return (
            f"stored history covers {hist_A.shape[1]}+{hist_B.shape[1]} atoms, but the "
            f"geometry has {n_atoms_A}+{n_atoms_B}"
        )
    return None


def _stored_history(df, row_index, mode_id):
    """``(hist_A, hist_B, result)`` for a stored mode, or None when nothing is stored."""
    parameterization = mode_parameterization(mode_id)
    if not thole_damping.has_ipd_result(df, row_index, **parameterization):
        return None
    result = thole_damping.read_ipd_row(df, row_index, **parameterization)
    return (
        np.asarray(result["mu_hist_A"], dtype=float),
        np.asarray(result["mu_hist_B"], dtype=float),
        result,
    )


# --- per-frame metadata -----------------------------------------------------


def _mode_entry(df, row_index, mode_id, n_atoms, n_atoms_A):
    """One mode's line in a frame's IPD block, stored or not.

    Every mode is listed whether or not it has a result. A mode that vanished from the list
    when it had not been run would leave the panel unable to say *which* calculation is
    available to run -- the same reasoning that keeps an incompatible dataset visible in the
    sidebar instead of refusing the upload.
    """
    entry = {
        "id": mode_id,
        "label": mode_label(mode_id),
        "stored": False,
        "iteration_count": None,
        "energy": None,
        "converged": None,
        "max_abs_mu": None,
        "problem": None,
    }

    stored = _stored_history(df, row_index, mode_id)
    if stored is None:
        return entry

    hist_A, hist_B, result = stored
    problem = _history_problem(hist_A, hist_B, n_atoms, n_atoms_A)
    if problem is not None:
        # Reported, not raised, and not silently dropped. A frame whose stored history does not
        # fit its geometry is a real thing that happened to the data, and saying so is more use
        # than either pretending it is absent or taking the whole trajectory down.
        entry["problem"] = problem
        return entry

    # One frame summary lets the browser choose a single selected-mode scale over the whole
    # trajectory without eager-loading every full SCF history. Ignore non-finite magnitudes;
    # JSON serialization rejects NaN and an invalid vector should not poison every frame's fit.
    magnitudes = np.linalg.norm(np.concatenate([hist_A, hist_B], axis=1), axis=-1)
    finite_magnitudes = magnitudes[np.isfinite(magnitudes)]
    max_abs_mu = float(finite_magnitudes.max()) if finite_magnitudes.size else 0.0

    entry.update(
        {
            "stored": True,
            # The number of history entries, which is the number of dots on the timeline:
            # entry 0 is the pre-SCF seed the UI labels "Initial". thole_damping counts the
            # same array as `iterations = len - 1`, so this is named to not be that.
            "iteration_count": int(hist_A.shape[0]),
            "energy": _finite(result["energy"]),
            "converged": bool(result["converged"]),
            "max_abs_mu": max_abs_mu,
        }
    )
    return entry


def frame_ipd_metadata(df, row_index, *, n_atoms, n_atoms_A, computable):
    """What IPD functionality exists for one frame -- small enough for every payload.

    ``computable`` is the ``ipd_computable`` feature: whether the row carries the MBIS inputs
    an IPD run reads. It says nothing about whether *this server* can run one, which is
    ``capability.capability()`` and a separate question with a separate answer.
    """
    modes = [
        _mode_entry(df, row_index, mode_id, n_atoms, n_atoms_A) for mode_id in MODES
    ]
    return {
        "computable": bool(computable),
        "available": any(mode["stored"] for mode in modes),
        "modes": modes,
    }


# --- one frame, one mode, the whole history ---------------------------------


def frame_ipd_history(df, row_index, mode_id):
    """One frame's full SCF trajectory for one damping mode, in geometry atom order.

    ``mu_history[i]`` is ``(n_atoms, 3)`` and lines up element-wise with the frame's ``coords``
    and ``symbols``: monomer A's atoms first, then monomer B's. The join happens here rather
    than in the browser because atom order is a backend invariant -- atom ``i`` is in A iff
    ``i < n_atoms_A`` -- and reconstructing it client-side would be a second place for that
    rule to live.

    Raises :class:`IpdError` when nothing is stored for this mode, or when what is stored does
    not describe this geometry.
    """
    parameterization = mode_parameterization(mode_id)

    geometry = system_processing.geometry_fields(df.at[row_index, "qcel_molecule"])
    if geometry is None:
        raise IpdError(
            "no_geometry",
            f"Row {row_index} has no readable geometry, so an induced-dipole history "
            "cannot be placed on its atoms.",
            status=404,
        )

    stored = _stored_history(df, row_index, mode_id)
    if stored is None:
        raise IpdError(
            "ipd_not_computed",
            f"No stored {mode_label(mode_id)} result for this frame.",
            status=404,
            details={"mode": mode_id, "parameterization": dict(parameterization)},
        )

    hist_A, hist_B, result = stored
    n_atoms = int(geometry["n_atoms"])
    n_atoms_A = int(geometry["n_atoms_A"])

    problem = _history_problem(hist_A, hist_B, n_atoms, n_atoms_A)
    if problem is not None:
        raise IpdError(
            "ipd_history_mismatch",
            f"The stored {mode_label(mode_id)} history does not match this frame's "
            f"geometry: {problem}.",
            status=409,
            details={"mode": mode_id},
            user_fixable=False,
        )

    # A then B, matching coords and symbols. Guaranteed to be meaningful by the fragment
    # contiguity dataframe_validation enforces at upload.
    mu_history = np.concatenate([hist_A, hist_B], axis=1)

    return {
        "mode": mode_id,
        "label": mode_label(mode_id),
        "energy": _finite(result["energy"]),
        "iteration_count": int(mu_history.shape[0]),
        "converged": bool(result["converged"]),
        "n_atoms": n_atoms,
        "n_atoms_A": n_atoms_A,
        "mu_history": _jsonable_array(mu_history),
    }


def _jsonable_array(array):
    """A nested list of floats with every non-finite entry replaced by None.

    ``np.where`` rather than a per-element walk: these arrays reach 201 x 13 x 3, and a
    comprehension over that is a visible cost repeated per request.
    """
    array = np.asarray(array, dtype=float)
    if np.isfinite(array).all():
        return array.tolist()
    return np.where(np.isfinite(array), array, None).tolist()
