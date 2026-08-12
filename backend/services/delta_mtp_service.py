"""Collection-aware orchestration for selected-frame and selected-system Delta-MTP."""

from __future__ import annotations

import numpy as np

from backend.services import (
    capability,
    collection_update,
    dataframe_schema,
    delta_mtp_compute,
    system_serialization,
    trajectory_service,
)
from backend.services.errors import DeltaMtpError


def _expected_shapes(row):
    """Expected Delta-MTP array shapes from the row's two-fragment molecule."""
    molecule = row.get("qcel_molecule")
    try:
        n_atoms = len(molecule.symbols)
        n_atoms_a = len(molecule.fragments[0])
        if len(molecule.fragments) != 2:
            raise ValueError("not a two-fragment dimer")
    except (AttributeError, TypeError, IndexError, ValueError) as exc:
        raise DeltaMtpError(
            "delta_mtp_molecule_invalid",
            f"This row does not contain a usable two-fragment qcel_molecule: {exc}.",
            status=422,
            details={"invalid_columns": ["qcel_molecule"]},
        ) from exc

    counts = {"A": n_atoms_a, "B": n_atoms - n_atoms_a, "dimer": n_atoms}
    tails = {"q": (), "mu": (3,), "theta": (3, 3)}
    return {
        f"{prefix} hf/adz {suffix}": (count,) + tail
        for prefix, tail in tails.items()
        for suffix, count in counts.items()
    }


def _require_inputs(df, label):
    """Raise a structured error unless all nine row inputs are finite and correctly shaped."""
    required = dataframe_schema.FEATURE_REQUIREMENTS["delta_mtp_computable"]
    missing = sorted(column for column in required if column not in df.columns)
    if missing:
        raise DeltaMtpError(
            "delta_mtp_inputs_missing",
            "This dataset cannot be used for a Delta-MTP calculation: it is missing "
            f"{', '.join(repr(column) for column in missing)}.",
            details={"missing_columns": missing},
        )

    row = df.loc[label]
    expected = _expected_shapes(row)
    invalid = []
    found_shapes = {}
    for column in sorted(required):
        try:
            value = np.asarray(row[column], dtype=float)
        except (TypeError, ValueError):
            invalid.append(column)
            found_shapes[column] = None
            continue
        found_shapes[column] = list(value.shape)
        if value.shape != expected[column] or not np.isfinite(value).all():
            invalid.append(column)

    if invalid:
        raise DeltaMtpError(
            "delta_mtp_inputs_invalid",
            f"Row {label} cannot be used for a Delta-MTP calculation: required input(s) "
            f"{', '.join(repr(column) for column in invalid)} are non-finite or have the "
            "wrong atom-aligned shape.",
            status=422,
            details={
                "invalid_columns": invalid,
                "expected_shapes": {column: list(expected[column]) for column in invalid},
                "found_shapes": {column: found_shapes[column] for column in invalid},
                "row_index": int(label),
            },
        )


def _compute_into(df, label):
    """Compute one complete row result, then and only then replace its four columns."""
    _require_inputs(df, label)
    original_columns = set(df.columns)
    original_values = {
        column: df.at[label, column]
        for column in dataframe_schema.DELTA_MTP_OUTPUT_COLUMNS
        if column in original_columns
    }
    try:
        result = delta_mtp_compute.compute_delta_mtp_row(df.loc[label])
        delta_mtp_compute.write_delta_mtp_row(df, label, result)
    except DeltaMtpError:
        raise
    except Exception as exc:
        # A write failure is unlikely after scalar validation, but a batch may persist later
        # successful rows. Roll this row back so that cannot turn one failed replacement into
        # a partial stored result.
        for column, value in original_values.items():
            df.at[label, column] = value
        newly_created = [
            column
            for column in dataframe_schema.DELTA_MTP_OUTPUT_COLUMNS
            if column not in original_columns and column in df.columns
        ]
        if newly_created:
            df.drop(columns=newly_created, inplace=True)
        raise DeltaMtpError(
            "delta_mtp_computation_failed",
            f"The Delta-MTP calculation failed for row {label}. "
            f"{type(exc).__name__}: {exc}",
            status=500,
            details={"row_index": int(label)},
            user_fixable=False,
        ) from exc
    return result


def compute(upload_id, slug, frame_index):
    """Compute one selected frame, persist, and return refreshed browser state."""
    capability.require_delta_mtp_capability()
    _, frame = collection_update.frame_entry(upload_id, slug, frame_index)

    with collection_update.lock_for(upload_id):
        df = system_serialization.load_processed(upload_id)
        label = collection_update.row_label(df, frame)
        result = _compute_into(df, label)
        collection_update.persist(upload_id, df)

        refreshed_system, refreshed = collection_update.frame_entry(
            upload_id, slug, frame_index
        )
        return {
            "frame_index": int(frame_index),
            "delta_mtp": refreshed["delta_mtp"],
            "energies": refreshed.get("energies", {}),
            "energy_catalog": refreshed_system.get("energy_catalog", []),
            "outputs": result,
        }


def compute_system(upload_id, slug):
    """Recompute every frame in one selected system, continuing after row failures."""
    capability.require_delta_mtp_capability()
    system = trajectory_service.get_trajectory(upload_id, slug)

    with collection_update.lock_for(upload_id):
        df = system_serialization.load_processed(upload_id)
        computed, failed = [], []
        for frame in system["frames"]:
            index = int(frame["frame_index"])
            label = collection_update.row_label(df, frame)
            try:
                _compute_into(df, label)
            except DeltaMtpError as exc:
                failed.append(
                    {"frame_index": index, "code": exc.code, "error": exc.message}
                )
                continue
            computed.append(index)

        if computed:
            collection_update.persist(upload_id, df)

        refreshed = trajectory_service.get_trajectory(upload_id, slug)
        return {
            "label": "Delta-MTP",
            "computed": computed,
            "failed": failed,
            "energy_catalog": refreshed.get("energy_catalog", []),
            "frames": [
                {
                    "frame_index": int(frame["frame_index"]),
                    "delta_mtp": frame["delta_mtp"],
                    "energies": frame.get("energies", {}),
                }
                for frame in refreshed["frames"]
            ],
        }
