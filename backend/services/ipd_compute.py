"""Orchestration around :mod:`backend.services.radius_thole`.

Owns request validation, row selection, persistence and error translation. It owns none
of the science and none of the result-column naming -- both of those live in
``radius_thole``, and duplicating either here is how the two would drift apart.
"""

from __future__ import annotations

import inspect
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.services import dataset_store, dipole_data
from backend.services.errors import IpdError

# The same constant scripts/build_ipd_dataset.py converts with. qcelemental geometries are
# in bohr; the viewer places arrows in Angstrom. Deliberately the literal rather than a
# qcelemental lookup, so an on-demand geometry and a pre-baked one agree exactly.
BOHR2ANG = 0.52917720859

# The UI offers four mutually exclusive damping modes. This mapping is the *only* place
# the wire value becomes a Python value, and the reason the wire vocabulary is lowercase
# "none" rather than "None": radius_thole treats a non-empty string as truthy, so a
# literal "None" reaching the science would leave radius-based damping switched on while
# the results were filed under the undamped column name. Here that is a 400 instead.
RADIUS_FROM_API: Dict[str, Optional[str]] = {
    "none": None,
    "intra": "intra",
    "inter": "inter",
    "both": "both",
}

# Inverse, for labelling a stored result in a response.
RADIUS_TO_API: Dict[Optional[str], str] = {v: k for k, v in RADIUS_FROM_API.items()}


def parse_radius(value: Any) -> Optional[str]:
    """Map a request's radius string to the value ``radius_thole`` expects."""
    if not isinstance(value, str) or value.lower() not in RADIUS_FROM_API:
        raise IpdError(
            "unsupported_radius",
            f"radius must be one of {sorted(RADIUS_FROM_API)}; got {value!r}",
            details={"allowed": sorted(RADIUS_FROM_API), "received": value},
        )
    return RADIUS_FROM_API[value.lower()]


# --- capability -------------------------------------------------------------

# Keyword arguments radius_thole.compute_ipd_row passes into
# induced_dipole_induction_optimized_no_correction. Checked by name rather than by a
# version string because apnet_pt self-reports "0.0.1" on every commit, while the
# function's signature has genuinely changed between checkouts -- an older one has no
# alpha_*_external, no per-channel damping parameters and no dipole history at all, which
# is exactly the incompatibility a version pin would fail to catch.
REQUIRED_IPD_KWARGS = (
    "alpha_A_external",
    "alpha_B_external",
    "max_iterations",
    "thole_damping_param_direct",
    "thole_damping_param_mutual_AB",
    "thole_damping_param_AA",
    "thole_damping_param_BB",
    "direct_damping_style",
    "mutual_AB_damping_style",
    "AA_damping_style",
    "BB_damping_style",
    "return_components",
    "return_induced_dipoles",
    "return_lambdas",
    "return_induced_dipole_history",
    "include_H",
)

_CAPABILITY: Optional[Dict[str, Any]] = None


def missing_ipd_kwargs(ipd_fn) -> List[str]:
    """Which of the arguments ``radius_thole`` passes this function does not accept.

    Separated from the probe so it can be checked against a stand-in signature without
    importing apnet_pt -- which pulls in torch, and is far too heavy a thing to drag into
    a unit test.
    """
    parameters = inspect.signature(ipd_fn).parameters
    return [name for name in REQUIRED_IPD_KWARGS if name not in parameters]


def _probe_capability() -> Dict[str, Any]:
    """Import apnet_pt and confirm it exposes the API ``radius_thole`` depends on."""
    try:
        import apnet_pt
        from apnet_pt.AtomPairwiseModels.mtp_mtp import (
            induced_dipole_induction_optimized_no_correction as ipd_fn,
        )
        from apnet_pt.pt_datasets.ap2_fused_ds import (  # noqa: F401
            qcel_dimer_to_fused_data,
            ap2_fused_collate_update_no_target,
        )
    except ImportError as exc:
        return {
            "available": False,
            "code": "apnet_pt_missing",
            "reason": (
                "apnet_pt is not installed in the server's environment. "
                f"Install it (pip install -e ~/gits/qcmlforge). Original error: {exc}"
            ),
            "version": None,
            "path": None,
            "missing_kwargs": [],
        }
    except Exception as exc:
        # Installed but not importable -- an incompatible transitive dependency rather
        # than a missing package. Observed for real: apnet_pt under a Python 3.14 env
        # whose pydantic/qcelemental pairing raises RuntimeError partway through import.
        # Catching only ImportError here would turn that into a 500 on a capability
        # *query*, which is the one endpoint that must always be able to answer.
        return {
            "available": False,
            "code": "apnet_pt_broken",
            "reason": (
                "apnet_pt is installed but failed to import, which usually means a "
                "conflicting dependency in this environment rather than a missing "
                f"package. {type(exc).__name__}: {exc}"
            ),
            "version": None,
            "path": None,
            "missing_kwargs": [],
        }

    version = getattr(apnet_pt, "__version__", None)
    path = getattr(apnet_pt, "__file__", None)

    missing = missing_ipd_kwargs(ipd_fn)
    if missing:
        return {
            "available": False,
            "code": "apnet_pt_incompatible",
            "reason": (
                "The installed apnet_pt exposes an incompatible "
                "induced_dipole_induction_optimized_no_correction: it is missing "
                f"{len(missing)} argument(s) radius_thole.py requires."
            ),
            "version": version,
            "path": path,
            "missing_kwargs": missing,
        }

    return {
        "available": True,
        "code": None,
        "reason": None,
        "version": version,
        "path": path,
        "missing_kwargs": [],
    }


def capability(refresh: bool = False) -> Dict[str, Any]:
    """Whether this server can run an IPD calculation, and why not when it cannot.

    Cached after the first call: the answer cannot change without restarting the process,
    and the probe costs a full ``import torch``. The frontend calls this on page load, so
    that multi-second import is paid before the user presses anything rather than on top
    of their first calculation.
    """
    global _CAPABILITY
    if _CAPABILITY is None or refresh:
        _CAPABILITY = _probe_capability()
    return dict(_CAPABILITY)


def require_capability() -> None:
    """Raise the matching :class:`IpdError` when IPD computation is unavailable."""
    cap = capability()
    if cap["available"]:
        return
    raise IpdError(
        cap["code"],
        cap["reason"],
        status=503,
        details={
            "missing_kwargs": cap["missing_kwargs"],
            "apnet_pt_path": cap["path"],
            "apnet_pt_version": cap["version"],
        },
        retryable=False,
    )


# --- dataset requirements ---------------------------------------------------

# Grouped so a compatibility report can say *what kind* of thing is missing rather than
# just listing eight column names. Read off radius_thole.compute_ipd_row, which is the
# only consumer that matters.
REQUIRED_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "grouping": ("system_id",),
    "geometry": ("qcel_molecule",),
    "mbis_a": ("q hf/adz A", "mu hf/adz A", "theta hf/adz A", "volume ratios A"),
    "mbis_b": ("q hf/adz B", "mu hf/adz B", "theta hf/adz B", "volume ratios B"),
    "mbis_dimer": ("q hf/adz dimer", "mu hf/adz dimer", "theta hf/adz dimer", "volume ratios dimer"),
}

ALL_REQUIRED_COLUMNS: Tuple[str, ...] = tuple(
    column for group in REQUIRED_COLUMNS.values() for column in group
)

# Slugs and labels matching scripts/build_ipd_dataset.py's ION_WATER_MODELS, so a computed
# result and a pre-baked one name the same model the same way.
MODEL_FOR_RADIUS: Dict[Optional[str], Dict[str, str]] = {
    None: {
        "slug": "mbis_0_39_all",
        "label": "mu_ind hist (MBIS) 0.39 all",
    },
    "intra": {
        "slug": "radius_thole_ts_mbis_intra_only",
        "label": "mu_ind hist (radius thole ts_mbis intra only)",
    },
    "inter": {
        "slug": "radius_thole_ts_mbis_inter_only",
        "label": "mu_ind hist (radius thole ts_mbis inter only)",
    },
    "both": {
        "slug": "radius_thole_ts_mbis_intra_inter",
        "label": "mu_ind hist (radius thole ts_mbis intra inter)",
    },
}


# --- grouping and labelling -------------------------------------------------

# A trajectory's rows are named for their separation: 01_Na-Water_1.00, 01_Na-Water_1.05,
# 28_Benzene-Uracil_pi-pi_1.50. Stripping that trailing token recovers the trajectory the
# System selector should list, and the token itself *is* the separation -- so one pattern
# serves both, and they cannot drift apart about where a group name ends.
#
# When a dataset does not follow the convention nothing is stripped, every group holds one
# geometry, and the cascade still works -- it just has nothing to collapse.
_SEPARATION_TOKEN = dipole_data.SEPARATION_TOKEN


def system_group(system_id: str) -> str:
    """The trajectory a ``system_id`` belongs to."""
    stripped = _SEPARATION_TOKEN.sub("", str(system_id))
    return stripped or str(system_id)


# Re-exported so callers that only care about the separation do not have to reach into
# dipole_data for it. The regex above is shared, so the group name and the separation can
# never disagree about where a group name ends.
separation_from_system_id = dipole_data.separation_from_system_id


# --- validation -------------------------------------------------------------


def _fragments(molecule) -> Optional[List[np.ndarray]]:
    fragments = getattr(molecule, "fragments", None)
    if fragments is None:
        return None
    return [np.asarray(fragment) for fragment in fragments]


def _contact_distance(molecule) -> Optional[float]:
    """Closest A-to-B contact for a row's molecule, in Angstrom, or None.

    Never raises: this feeds a catalog entry, and one geometry with an odd molecule must
    not take the whole listing down with it.
    """
    if molecule is None:
        return None
    fragments = _fragments(molecule)
    if fragments is None or len(fragments) != 2:
        return None
    try:
        # qcelemental keeps geometry in bohr; the catalog reports Angstrom throughout.
        coords = np.asarray(molecule.geometry, dtype=float) * BOHR2ANG
    except (AttributeError, TypeError, ValueError):
        return None
    return dipole_data.contact_distance(coords, len(fragments[0]))


def _row_is_usable(row) -> bool:
    """Cheap per-row screen for the dataset-level count. Not a substitute for validate_row."""
    molecule = row.get("qcel_molecule")
    fragments = _fragments(molecule)
    if fragments is None or len(fragments) != 2:
        return False
    for column in ALL_REQUIRED_COLUMNS:
        value = row.get(column)
        if value is None:
            return False
    return True


def inspect_compatibility(df) -> Dict[str, Any]:
    """Dataset-level report: can IPD be run against this dataframe at all?

    Never fatal. An incompatible dataframe still registers, and the sidebar explains why
    the button is disabled -- which is far more useful than refusing the upload.
    """
    missing = [column for column in ALL_REQUIRED_COLUMNS if column not in df.columns]
    missing_by_group = {
        group: [column for column in columns if column not in df.columns]
        for group, columns in REQUIRED_COLUMNS.items()
    }

    usable = 0
    if not missing:
        usable = int(sum(1 for _, row in df.iterrows() if _row_is_usable(row)))

    return {
        "ok": not missing and usable > 0,
        "missing_columns": missing,
        "missing_by_group": {k: v for k, v in missing_by_group.items() if v},
        "usable_rows": usable,
        "total_rows": int(len(df)),
    }


def validate_row(row, radius: Optional[str]) -> None:
    """Row-level checks run immediately before a calculation.

    Deliberately stops short of the scientific validation radius_thole and apnet_pt
    already do -- edge construction, polarizability lookups, damping. What it covers is
    the shape and presence errors that would otherwise surface as an opaque traceback
    from deep inside the SCF.
    """
    missing = [column for column in ALL_REQUIRED_COLUMNS if row.get(column) is None]
    if missing:
        raise IpdError(
            "invalid_row_data",
            f"This geometry is missing required values: {', '.join(missing)}.",
            status=422,
            details={"missing": missing},
        )

    molecule = row["qcel_molecule"]
    fragments = _fragments(molecule)
    if fragments is None or len(fragments) != 2:
        count = "none" if fragments is None else len(fragments)
        raise IpdError(
            "invalid_row_data",
            f"qcel_molecule must describe a dimer with exactly 2 fragments; found {count}.",
            status=422,
            details={"field": "qcel_molecule"},
        )

    n_a, n_b = len(fragments[0]), len(fragments[1])
    n_atoms = n_a + n_b
    # The induced dipoles are concatenated monomer-A-then-B to line up with the
    # coordinates, so the fragments must partition the atoms in that same contiguous
    # order. The identical check guards the offline path in build_ipd_dataset.convert_row.
    if not np.array_equal(np.concatenate(fragments), np.arange(n_atoms)):
        raise IpdError(
            "invalid_row_data",
            "qcel_molecule's fragments are not contiguous monomer-A-then-B, so induced "
            "dipoles could not be matched to atoms.",
            status=422,
            details={"field": "qcel_molecule"},
        )

    # Checked on element count per atom rather than on an exact shape, because that is
    # what the science actually requires: radius_thole reshapes the volume ratios flat,
    # and apnet_pt documents the monopoles as "(N,) or (N,1)". Demanding one exact shape
    # rejects valid data -- a single-atom monomer stores its dipole as (1, 3), which is
    # indistinguishable from (3,) once squeezed.
    for monomer, count in (("A", n_a), ("B", n_b)):
        for column, per_atom, description in (
            (f"q hf/adz {monomer}", 1, "one monopole per atom"),
            (f"mu hf/adz {monomer}", 3, "three dipole components per atom"),
            (f"volume ratios {monomer}", 1, "one volume ratio per atom"),
        ):
            array = np.asarray(row[column], dtype=float)
            if array.size != count * per_atom:
                raise IpdError(
                    "invalid_row_data",
                    f"{column!r} has shape {tuple(array.shape)} ({array.size} values), but "
                    f"monomer {monomer} has {count} atoms and needs {description} "
                    f"({count * per_atom} values).",
                    status=422,
                    details={
                        "field": column,
                        "expected_values": count * per_atom,
                        "actual_shape": list(array.shape),
                    },
                )

        theta = np.asarray(row[f"theta hf/adz {monomer}"], dtype=float)
        if theta.size % count or theta.shape[0] != count:
            raise IpdError(
                "invalid_row_data",
                f"'theta hf/adz {monomer}' has shape {tuple(theta.shape)}, whose leading "
                f"axis does not match monomer {monomer}'s {count} atoms.",
                status=422,
                details={"field": f"theta hf/adz {monomer}",
                         "actual_shape": list(theta.shape)},
            )

        ratios = np.asarray(row[f"volume ratios {monomer}"], dtype=float)
        if not np.isfinite(ratios).all():
            # radius_thole._ts_mbis_radius_ang raises on this, because R_vdw(TS) is NaN
            # for ionic rows. Caught here so the user gets a sentence instead of a
            # traceback about non-finite TS/MBIS radii.
            raise IpdError(
                "non_finite_volume_ratios",
                f"'volume ratios {monomer}' contains non-finite values, which the "
                "TS/MBIS damping radii cannot be derived from. This is common for bare "
                "monatomic ions whose volume ratios were stored as NaN.",
                status=422,
                details={"field": f"volume ratios {monomer}"},
            )


# --- registration -----------------------------------------------------------


# Columns the app derives for a dataframe that does not carry them. Both describe the
# geometry and nothing else, so deriving them can never contradict the uploaded science.
CONTACT_COLUMN = "contact_distance_ang"
EQ_RATIO_COLUMN = "eq_ratio"


def normalize_dataset(df) -> List[str]:
    """Fill in the derivable metadata columns in place, and name the ones that were added.

    A dataframe that states its separation and contact distance is easier to work with than
    one that does not, and both are recoverable from what every usable row already carries --
    so supplying them is this app's job rather than the uploader's.

    **Never overwrites a column the dataframe already has.** If the uploader supplied
    ``eq_ratio``, theirs wins: the point is to fill gaps, not to correct their science.

    ``eq_ratio`` is written only where a ratio is genuinely recoverable -- an explicit column
    or the ``system_id`` suffix. Where only a contact distance exists the cell is left NaN,
    because an Angstrom distance is not a ratio and writing it as one would be a lie that
    every later reader would believe.
    """
    added: List[str] = []

    if CONTACT_COLUMN not in df.columns:
        df[CONTACT_COLUMN] = [
            _contact_distance(row.get("qcel_molecule")) for _, row in df.iterrows()
        ]
        added.append(CONTACT_COLUMN)

    if EQ_RATIO_COLUMN not in df.columns:
        ratios = [
            dipole_data.separation_from_system_id(row.get("system_id", ""))
            for _, row in df.iterrows()
        ]
        if any(ratio is not None for ratio in ratios):
            df[EQ_RATIO_COLUMN] = [
                np.nan if ratio is None else ratio for ratio in ratios
            ]
            added.append(EQ_RATIO_COLUMN)

    return added


def restore(dataset_id: str) -> Dict[str, Any]:
    """Discard every computed result, then rebuild the working copy's invariants.

    Reinstating ``original.pkl`` verbatim would undo more than the calculations: the working
    copy is also canonically indexed and carries the derived metadata columns, neither of
    which the uploaded file has. So restore re-applies both, leaving a working copy identical
    to the one registration produced -- minus the results, which is the whole point.
    """
    directory = dataset_store.dataset_dir(dataset_id)
    with dataset_store.dataset_lock(dataset_id):
        df = dataset_store.read_dataframe(directory / dataset_store.ORIGINAL_NAME)
        df = df.reset_index(drop=True)
        added_columns = normalize_dataset(df)
        dataset_store.save_working(dataset_id, df)

        meta = dataset_store.read_meta(dataset_id)
        meta["added_columns"] = added_columns
        dataset_store.write_meta(dataset_id, meta)

    _CATALOG_CACHE.pop(dataset_id, None)
    return summary(dataset_id)


def register(raw: bytes, filename: str) -> Dict[str, Any]:
    """Register an uploaded dataframe and return its summary.

    Idempotent by content: re-uploading the same bytes re-opens the existing dataset with
    whatever results have already been computed into it, rather than starting over.
    """
    dataset_id, directory, is_new = dataset_store.store_upload(raw, filename)

    if not is_new:
        return summary(dataset_id)

    df = dataset_store.read_dataframe(directory / dataset_store.ORIGINAL_NAME)
    # One canonical ordering, fixed at registration. Every row_id is a position in *this*
    # frame, and the server never reorders or filters working.pkl afterwards.
    df = df.reset_index(drop=True)

    added_columns = normalize_dataset(df)

    compatibility = inspect_compatibility(df)
    rows = {
        dataset_store.row_id_for(position): {
            "system_id": str(row.get("system_id", "")),
            "fingerprint": dataset_store.row_fingerprint(row),
        }
        for position, (_, row) in enumerate(df.iterrows())
    }
    groups = {system_group(entry["system_id"]) for entry in rows.values()}

    meta = {
        "dataset_id": dataset_id,
        "name": filename,
        "registered_at": dataset_store.now_iso(),
        "n_rows": int(len(df)),
        "n_systems": len(groups),
        "compatibility": compatibility,
        "added_columns": added_columns,
        "rows": rows,
    }

    # working.pkl before meta.json: meta.json is what marks a directory as a registered
    # dataset, so writing it last means a crash mid-registration leaves a directory that
    # is ignored rather than one that half-resolves.
    df.to_pickle(directory / dataset_store.WORKING_NAME)
    dataset_store.write_meta(dataset_id, meta)
    return summary(dataset_id)


# --- listing ----------------------------------------------------------------


def list_systems(df, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Trajectories, their geometries, and which radius modes already have results.

    Stored-result detection is delegated to ``radius_thole.has_ipd_result`` rather than
    re-deriving column names here, so the badges cannot disagree with what a calculation
    would actually reuse.
    """
    from backend.services import radius_thole

    rows_meta = meta.get("rows", {})

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for position, (idx, row) in enumerate(df.iterrows()):
        row_id = dataset_store.row_id_for(position)
        system_id = str(row.get("system_id", ""))
        stored = {
            api_value: radius_thole.has_ipd_result(df, idx, radius)
            for api_value, radius in RADIUS_FROM_API.items()
        }
        groups.setdefault(system_group(system_id), []).append(
            {
                "row_id": row_id,
                "system_id": system_id,
                "position": position,
                "stored": stored,
                "usable": _row_is_usable(row),
                "fingerprint": rows_meta.get(row_id, {}).get("fingerprint"),
                # Kept only long enough to derive the separation below; dropped before this
                # is serialized, since a qcelemental Molecule is not JSON.
                "_row": row,
            }
        )

    # Per group, not per row: the source has to describe the whole trajectory or the
    # geometries would sort an Re ratio against an Angstrom distance. See derive_separations.
    for geometries in groups.values():
        derived = dipole_data.derive_separations(
            {
                "system_id": geometry["system_id"],
                "row": geometry["_row"],
                "contact_ang": _contact_distance(geometry["_row"].get("qcel_molecule")),
            }
            for geometry in geometries
        )
        for geometry, values in zip(geometries, derived):
            geometry.pop("_row")
            geometry["separation"] = values["separation"]
            geometry["separation_units"] = values["separation_units"]
            geometry["label"] = values["separation_label"]

    # A dataframe's rows are in whatever order they were built in -- s66x8 interleaves all
    # 66 trajectories -- so a trajectory only reads as one if its geometries are sorted by
    # separation here. Rows with no separation to sort by keep their dataframe position and
    # fall after the ones that have one.
    for geometries in groups.values():
        geometries.sort(
            key=lambda geometry: (
                geometry["separation"] is None,
                geometry["separation"] if geometry["separation"] is not None else 0.0,
                geometry["position"],
            )
        )

    return [
        {"group": group, "geometries": geometries}
        for group, geometries in sorted(groups.items())
    ]


def summary(dataset_id: str) -> Dict[str, Any]:
    """Everything the sidebar needs to render for one dataset, in a single response."""
    meta = dataset_store.read_meta(dataset_id)
    df = dataset_store.load_working(dataset_id)
    systems = list_systems(df, meta)

    n_with_results = sum(
        1
        for group in systems
        for geometry in group["geometries"]
        if any(geometry["stored"].values())
    )

    return {
        "dataset_id": dataset_id,
        "name": meta.get("name"),
        "registered_at": meta.get("registered_at"),
        "n_systems": len(systems),
        "n_geometries": int(len(df)),
        "n_with_results": n_with_results,
        "compatibility": meta.get("compatibility", {}),
        # Named rather than silent: the working copy carries columns the upload did not, and
        # the export hands them back, so the sidebar says which.
        "added_columns": meta.get("added_columns", []),
        "apnet_pt": capability(),
        "systems": systems,
    }


# --- payload ----------------------------------------------------------------


def _build_payload(row, result: Dict[str, Any], radius: Optional[str],
                   source_file: str) -> Dict[str, Any]:
    """Turn one calculation's output into the payload the viewer already understands."""
    molecule = row["qcel_molecule"]
    fragments = _fragments(molecule)
    n_atoms_A = len(fragments[0])

    # Monomer A then monomer B along the *atom* axis, matching the order of coords and
    # the offline path's stack_dipoles. Both halves share the SCF iteration axis.
    mu_a = np.asarray(result["mu_hist_A"], dtype=float)
    mu_b = np.asarray(result["mu_hist_B"], dtype=float)
    if mu_a.shape[0] != mu_b.shape[0]:
        raise IpdError(
            "calculation_failed",
            f"Monomer histories disagree on iteration count: {mu_a.shape[0]} vs "
            f"{mu_b.shape[0]}.",
            status=500,
            user_fixable=False,
        )
    mu = np.concatenate([mu_a, mu_b], axis=1)

    model = MODEL_FOR_RADIUS[radius]
    payload = dipole_data.build_history_payload(
        mu,
        np.asarray(molecule.geometry, dtype=float) * BOHR2ANG,
        [str(symbol) for symbol in molecule.symbols],
        system_id=str(row.get("system_id", "")),
        model=model["slug"],
        models=[dict(model)],
        n_atoms_A=n_atoms_A,
        source=source_file,
        mu_label=f"mu_ind hist (radius={radius!r})",
    )
    return payload


# --- the calculation --------------------------------------------------------


def run(dataset_id: str, row_id: str, radius_api: str,
        force_recompute: bool = False) -> Dict[str, Any]:
    """Load or compute one geometry's induced-dipole history, and persist it.

    Runs entirely under the dataset lock, so "read the frame, check the row, compute,
    write it back" is atomic with respect to any other request touching this dataset.
    """
    from backend.services import radius_thole

    radius = parse_radius(radius_api)
    require_capability()

    with dataset_store.dataset_lock(dataset_id):
        meta = dataset_store.read_meta(dataset_id)
        df = dataset_store.load_working(dataset_id)

        rows_meta = meta.get("rows", {})
        if row_id not in rows_meta:
            raise IpdError(
                "row_not_found",
                f"No geometry {row_id!r} in this dataset.",
                status=404,
                details={"row_id": row_id},
            )

        position = int(row_id)
        if position >= len(df):
            raise IpdError(
                "row_not_found",
                f"Geometry {row_id!r} is past the end of the dataframe.",
                status=404,
                details={"row_id": row_id},
            )
        idx = df.index[position]
        row = df.loc[idx]

        # The request was built against a listing; confirm that listing still describes
        # this row before writing anything into it.
        expected = rows_meta[row_id].get("fingerprint")
        actual = dataset_store.row_fingerprint(row)
        if expected and expected != actual:
            raise IpdError(
                "row_changed",
                "This geometry has changed since the page loaded it. Reload the dataset "
                "and try again.",
                status=409,
                details={"row_id": row_id},
                retryable=True,
                user_fixable=False,
            )

        warnings: List[str] = []
        stored = radius_thole.has_ipd_result(df, idx, radius)

        if stored and not force_recompute:
            result = radius_thole.read_ipd_row(df, idx, radius)
            result_source = "stored"
            dataset_updated = False
        else:
            validate_row(row, radius)
            try:
                result = radius_thole.compute_ipd_row(row, radius, verbose=False)
            except IpdError:
                raise
            except Exception as exc:
                raise IpdError(
                    "calculation_failed",
                    f"The IPD calculation failed. {type(exc).__name__}: {exc}",
                    status=500,
                    details={"row_id": row_id, "radius": radius_api},
                    retryable=True,
                    user_fixable=False,
                )
            radius_thole.write_ipd_row(df, idx, result, radius)
            dataset_store.save_working(dataset_id, df)
            result_source = "computed"
            dataset_updated = True

        # Not an error: a non-converged SCF is a real result, it is what the command-line
        # path would have stored too, and hiding it would be worse than labelling it.
        if not result.get("converged", True):
            warnings.append(
                f"The SCF did not converge within {radius_thole.MAX_SCF_ITERATIONS} "
                "iterations; the dipoles shown are the last iterate."
            )

        payload = _build_payload(row, result, radius, dataset_store.WORKING_NAME)

    payload.update(
        {
            "status": "success",
            "result_source": result_source,
            "dataset_updated": dataset_updated,
            "dataset_id": dataset_id,
            "system": system_group(str(row.get("system_id", ""))),
            "row_id": row_id,
            "radius": radius_api.lower(),
            "converged": bool(result.get("converged", True)),
            "iterations": int(result.get("iterations", 0)),
            "energy_kcalmol": (
                None if result.get("energy") is None else float(result["energy"])
            ),
            "columns_written": (
                sorted(radius_thole.ipd_column_names(radius).values())
                if dataset_updated
                else []
            ),
            "warnings": warnings,
        }
    )
    return payload


# --- catalog integration ----------------------------------------------------

# Reverse of MODEL_FOR_RADIUS, so a model slug arriving from the viewer's Model selector
# resolves back to the radius mode whose columns hold it.
RADIUS_FOR_MODEL_SLUG: Dict[str, Optional[str]] = {
    model["slug"]: radius for radius, model in MODEL_FOR_RADIUS.items()
}

# Catalog entries per dataset, keyed by the working file's (mtime, size). Building them
# reads every stored history to find its frame count and peak magnitude; without this the
# viewer would pay for that on every catalog fetch, and the frontend refetches the catalog
# after every calculation.
_CATALOG_CACHE: Dict[str, Tuple[Tuple[float, int], List[Dict[str, Any]]]] = {}


def parse_dataset_system_id(system_id: str) -> Optional[Tuple[str, str]]:
    """Split a catalog ``system_id`` into ``(dataset_id, row_id)``, or None if bundled.

    On-demand geometries are namespaced ``<dataset_id>/<row_id>`` so they cannot collide
    with the bundled catalog's ids, and so one string still identifies a geometry
    everywhere the viewer passes it around.
    """
    if "/" not in system_id:
        return None
    dataset_id, _, row_id = system_id.partition("/")
    if not dataset_id.startswith("d-") or not row_id:
        return None
    return dataset_id, row_id


# Detected from the columns, not declared per dataset, and shared with the offline converter
# so an uploaded dataframe and a bundled one report the same levels for the same columns.
_reference_energies = dipole_data.reference_energies


def _dataset_catalog_entries(dataset_id: str, df, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One catalog entry per geometry that has at least one stored radius mode.

    The entry shape matches ``data/ipd_catalog.json`` exactly, so the viewer's existing
    selector cascade consumes computed geometries with no special handling.
    """
    from backend.services import radius_thole

    label = Path(meta.get("name") or dataset_id).stem
    dataset_label = f"{label} (uploaded)"
    entries: List[Dict[str, Any]] = []

    for group in list_systems(df, meta):
        for geometry in group["geometries"]:
            stored_modes = [
                api for api, present in geometry["stored"].items() if present
            ]
            if not stored_modes:
                # Nothing to look at yet. The geometry joins the catalog the moment a
                # result exists for it -- whether that came from a calculation here or was
                # already in the uploaded file.
                continue

            idx = df.index[geometry["position"]]
            row = df.loc[idx]
            molecule = row.get("qcel_molecule")
            n_atoms = len(molecule.symbols) if molecule is not None else 0

            models = []
            max_abs_mu = 0.0
            for api in ("none", "intra", "inter", "both"):
                if api not in stored_modes:
                    continue
                radius = RADIUS_FROM_API[api]
                names = radius_thole.ipd_column_names(radius)
                mu_a = np.asarray(df.at[idx, names["mu_hist_A"]], dtype=float)
                mu_b = np.asarray(df.at[idx, names["mu_hist_B"]], dtype=float)
                try:
                    energy = float(df.at[idx, names["energy"]])
                except (TypeError, ValueError):
                    energy = None

                # One scale for every model at this geometry, so switching models compares
                # arrows rather than renormalising each to its own longest vector -- the
                # same invariant scripts/build_ipd_dataset.py maintains for bundled files.
                for mu in (mu_a, mu_b):
                    if mu.size:
                        max_abs_mu = max(
                            max_abs_mu, float(np.linalg.norm(mu, axis=-1).max())
                        )

                models.append(
                    {
                        **MODEL_FOR_RADIUS[radius],
                        "n_frames": int(mu_a.shape[0]),
                        "energy_kcalmol": (
                            None if energy is None or not np.isfinite(energy)
                            else round(energy, 6)
                        ),
                        "radius": api,
                    }
                )

            separation = geometry["separation"]
            entries.append(
                {
                    "system_id": f"{dataset_id}/{geometry['row_id']}",
                    "dataset": dataset_label,
                    "system_name": group["group"],
                    # list_systems has already chosen one separation source for the whole
                    # trajectory and stamped its real units, so nothing is guessed here. The
                    # row position is the last resort for a dataset that declares nothing and
                    # whose geometry will not parse -- it still has to sort somehow.
                    "separation": separation if separation is not None else float(geometry["position"]),
                    "separation_units": geometry["separation_units"],
                    "separation_label": geometry["label"],
                    "separation_alt_label": geometry["system_id"],
                    # Geometry-derived, so it means the same thing here as it does for a
                    # bundled system -- unlike separation_alt_label just above, which
                    # carries a distance for bundled entries and an id for these.
                    "contact_distance_ang": _contact_distance(molecule),
                    "n_atoms": n_atoms,
                    "n_frames": max(model["n_frames"] for model in models),
                    # Unrounded, unlike the bundled catalog's copy of this number: that one
                    # is rounded for file compactness, but the bundled *payload* reads the
                    # full-precision value out of the .npz. Keeping it exact here makes the
                    # two paths agree to the last bit for the same geometry.
                    "max_abs_mu_system": max_abs_mu,
                    "models": models,
                    "reference_energies": _reference_energies(row),
                    "file": None,
                    "source": "dataset",
                    "dataset_id": dataset_id,
                    "row_id": geometry["row_id"],
                }
            )
    return entries


def catalog_entries() -> List[Dict[str, Any]]:
    """Catalog entries for every registered dataset, for merging into /api/ipd/systems."""
    entries: List[Dict[str, Any]] = []
    for row in dataset_store.list_datasets():
        dataset_id = row["dataset_id"]
        try:
            stamp = dataset_store.working_stamp(dataset_id)
            cached = _CATALOG_CACHE.get(dataset_id)
            if cached is not None and cached[0] == stamp:
                entries.extend(cached[1])
                continue
            df = dataset_store.load_working(dataset_id)
            meta = dataset_store.read_meta(dataset_id)
        except (IpdError, dataset_store.DatasetNotFound, OSError):
            # One unreadable dataset must not empty the whole catalog, including the
            # bundled half of it.
            continue
        built = _dataset_catalog_entries(dataset_id, df, meta)
        _CATALOG_CACHE[dataset_id] = (stamp, built)
        entries.extend(built)
    return entries


def load_stored(dataset_id: str, row_id: str, model: Optional[str] = None) -> Dict[str, Any]:
    """Serve a stored result to the viewer, in the same shape as a bundled system.

    This is what ``GET /api/ipd/system`` answers with for a dataset-scoped system_id, so
    the merged catalog is browsable exactly like the pre-baked one.
    """
    from backend.services import radius_thole

    entry = next(
        (
            candidate
            for candidate in catalog_entries()
            if candidate["dataset_id"] == dataset_id and candidate["row_id"] == row_id
        ),
        None,
    )
    if entry is None:
        raise IpdError(
            "row_not_found",
            f"No stored IPD result for geometry {row_id!r} in dataset {dataset_id!r}.",
            status=404,
            details={"dataset_id": dataset_id, "row_id": row_id},
        )

    available = [candidate["slug"] for candidate in entry["models"]]
    if model is None:
        model = available[0]
    elif model not in available:
        raise IpdError(
            "model_not_stored",
            f"No stored result for model {model!r} at this geometry; available: "
            f"{', '.join(available)}.",
            details={"available": available, "requested": model},
        )

    radius = RADIUS_FOR_MODEL_SLUG[model]
    df = dataset_store.load_working(dataset_id)
    idx = df.index[int(row_id)]
    row = df.loc[idx]

    result = radius_thole.read_ipd_row(df, idx, radius)
    payload = _build_payload(row, result, radius, dataset_store.WORKING_NAME)

    # Merge the catalog metadata in, mirroring dipole_data.get_system. max_abs_mu comes
    # from the entry rather than this model's own history so arrow lengths stay comparable
    # when the Model selector changes at a fixed geometry.
    payload["models"] = entry["models"]
    payload["max_abs_mu"] = entry["max_abs_mu_system"]
    payload["dataset"] = entry["dataset"]
    payload["system_name"] = entry["system_name"]
    payload["separation"] = entry["separation"]
    payload["separation_units"] = entry["separation_units"]
    payload["separation_label"] = entry["separation_label"]
    payload["separation_alt_label"] = entry["separation_alt_label"]
    payload["reference_energies"] = entry["reference_energies"]
    payload["energy_kcalmol"] = next(
        (m["energy_kcalmol"] for m in entry["models"] if m["slug"] == model), None
    )
    payload["radius"] = RADIUS_TO_API[radius]
    payload["row_id"] = row_id
    payload["dataset_id"] = dataset_id
    return payload


# --- batch calculation ------------------------------------------------------


def run_group(dataset_id: str, system: str, radius_api: str,
              force_recompute: bool = False) -> Dict[str, Any]:
    """Compute every geometry in one trajectory, in a single load/save cycle.

    A batch endpoint rather than a frontend loop over :func:`run`, because the science is
    milliseconds per geometry while reading and rewriting the working pickle is not: doing
    it once for eight geometries instead of eight times is the whole point.
    """
    from backend.services import radius_thole

    radius = parse_radius(radius_api)
    require_capability()
    started = time.monotonic()

    with dataset_store.dataset_lock(dataset_id):
        meta = dataset_store.read_meta(dataset_id)
        df = dataset_store.load_working(dataset_id)

        group = next(
            (g for g in list_systems(df, meta) if g["group"] == system), None
        )
        if group is None:
            raise IpdError(
                "system_not_found",
                f"No system {system!r} in this dataset.",
                status=404,
                details={"system": system},
            )

        computed: List[str] = []
        reused: List[str] = []
        failed: List[Dict[str, Any]] = []
        not_converged: List[str] = []

        for geometry in group["geometries"]:
            row_id = geometry["row_id"]
            idx = df.index[geometry["position"]]
            row = df.loc[idx]

            if radius_thole.has_ipd_result(df, idx, radius) and not force_recompute:
                reused.append(row_id)
                continue

            try:
                validate_row(row, radius)
                result = radius_thole.compute_ipd_row(row, radius, verbose=False)
            except IpdError as err:
                # One unusable geometry must not cost the other seven. Record it and keep
                # going; whatever succeeded is still saved below.
                failed.append({"row_id": row_id, "code": err.code, "error": err.message})
                continue
            except Exception as exc:
                failed.append(
                    {
                        "row_id": row_id,
                        "code": "calculation_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            radius_thole.write_ipd_row(df, idx, result, radius)
            computed.append(row_id)
            if not result.get("converged", True):
                not_converged.append(row_id)

        if computed:
            dataset_store.save_working(dataset_id, df)

        # Whatever is now viewable, preferring something this call produced.
        first = (computed or reused or [None])[0]

    return {
        "status": "success",
        "dataset_id": dataset_id,
        "system": system,
        "radius": radius_api.lower(),
        "computed": len(computed),
        "reused": len(reused),
        "failed": failed,
        "not_converged": not_converged,
        "dataset_updated": bool(computed),
        "first_row_id": first,
        "elapsed_s": round(time.monotonic() - started, 3),
    }
