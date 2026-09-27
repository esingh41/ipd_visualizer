"""Build the curated NENCI-2021 binding-motif dataframe from exactly two sources.

The inputs are trusted research pickles and are read-only. The checked-in CSV decides both
membership and display order: omitting a parent excludes its complete trajectory.

Run from the repository root in the qcml_10 environment::

    conda run -n qcml_10 python scripts/build_nenci2021_curated.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

# qcelemental/Pydantic models pickle internal sets. Fix Python's hash seed before importing
# science dependencies so identical inputs and mapping produce byte-identical output, which in
# turn preserves the app's content-derived collection id across rebuilds.
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "0":
    environment = dict(os.environ, PYTHONHASHSEED="0")
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services import dataframe_validation, system_processing  # noqa: E402

DEFAULT_NENCI = (
    ROOT / "data/source/nenci2021_subset_nons66x8_equilibrium_extended_saptdft.pkl"
)
DEFAULT_S66 = ROOT / "data/source/s66x8_dimer_multipoles_hippo.pkl"
DEFAULT_MAP = ROOT / "curation/nenci2021_binding_motifs.csv"
DEFAULT_OUTPUT = ROOT / "data/source/nenci2021_binding_motifs.pkl"

# Only scientifically equivalent spellings share a group. Conventional S66 SAPT0 is
# deliberately absent: its SAPT0 dispersion is not run_saptdft.py's empirical D4(i) term.
SAPT_ALIAS_GROUPS = {
    "SAPT(PBE0)-D4(i) TOTAL kcalmol": ("F-Total",),
    "SAPT(PBE0)-D4(i) ELST kcalmol": ("F-Electrostatics",),
    "SAPT(PBE0)-D4(i) EXCH kcalmol": ("F-Exchange",),
    "SAPT(PBE0)-D4(i) INDU kcalmol": ("F-Induction",),
    "SAPT(PBE0)-D4(i) DISP kcalmol": ("F-Dispersion",),
    "SAPT(PBE0)-D4(i) ind20,r kcalmol": ("ind20,r_dft",),
    "SAPT(PBE0)-D4(i) exch-ind20,r kcalmol": ("exch-ind20,r_dft",),
    "delta HF correction": ("delta hf correction",),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    raw = df[column]
    numeric = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & numeric.isna()
    if invalid.any():
        position = int(np.flatnonzero(invalid.to_numpy())[0])
        system_id = df.iloc[position].get("system_id", f"row {position}")
        raise ValueError(
            f"Energy column {column!r} is nonnumeric at {system_id!r}: "
            f"{raw.iloc[position]!r}."
        )
    return numeric.astype(float)


def coalesce_sapt_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with genuine SAPT aliases coalesced into explicit method names.

    A canonical value wins only when aliases are absent. Populated overlaps must agree;
    disagreement is data damage, not a priority decision.
    """
    df = df.copy()
    for canonical, aliases in SAPT_ALIAS_GROUPS.items():
        present_aliases = [alias for alias in aliases if alias in df.columns]
        if canonical not in df.columns and not present_aliases:
            continue

        output = (
            _numeric_column(df, canonical)
            if canonical in df.columns
            else pd.Series(np.nan, index=df.index, dtype=float)
        )
        for alias in present_aliases:
            candidate = _numeric_column(df, alias)
            overlap = output.notna() & candidate.notna()
            conflict = overlap & ~np.isclose(
                output.fillna(0.0),
                candidate.fillna(0.0),
                rtol=1e-10,
                atol=1e-8,
            )
            if conflict.any():
                position = int(np.flatnonzero(conflict.to_numpy())[0])
                system_id = df.iloc[position].get("system_id", f"row {position}")
                raise ValueError(
                    f"Conflicting SAPT aliases at {system_id!r}: {canonical!r}="
                    f"{output.iloc[position]!r}, {alias!r}={candidate.iloc[position]!r}."
                )
            output = output.fillna(candidate)

        df[canonical] = output
        if present_aliases:
            df = df.drop(columns=present_aliases)
    return df


def _fill_non_lithium_monoatomic_volume_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Fill isolated Na/F/Cl ratios from each parent's farthest dimer frame.

    Isolated monoatomic MBIS calculations carry no finite stockholder-volume ratio. For these
    ion–pi trajectories, the atom's dimer value at the largest separation is used as its
    asymptotic isolated reference across the scan. Lithium is deliberately excluded, even in
    the exceptional trajectory where a finite dimer value exists.
    """
    df = df.copy()
    repaired_parents = set()
    repaired_rows = set()

    for parent, group in df.groupby("_parent_system_id", sort=False):
        separations = group["system_id"].map(
            system_processing.separation_from_system_id
        )
        if separations.isna().any():
            raise ValueError(
                f"Cannot choose a farthest frame for trajectory {parent!r}: "
                "one or more system IDs have no separation token."
            )
        maximum = float(separations.max())
        farthest_labels = separations[separations == maximum].index.tolist()
        if len(farthest_labels) != 1:
            raise ValueError(
                f"Trajectory {parent!r} has {len(farthest_labels)} frames at its largest "
                f"separation {maximum}."
            )
        farthest_label = farthest_labels[0]
        farthest_molecule = df.at[farthest_label, "qcel_molecule"]
        fragments = getattr(farthest_molecule, "fragments", None)
        symbols = list(map(str, getattr(farthest_molecule, "symbols", ())))
        if fragments is None or len(fragments) != 2 or not symbols:
            raise ValueError(f"Trajectory {parent!r} has no readable two-fragment dimer.")

        candidates = []
        for fragment_number, monomer_column in enumerate(
            ("volume ratios A", "volume ratios B")
        ):
            atom_indices = np.asarray(fragments[fragment_number], dtype=int).reshape(-1)
            if len(atom_indices) != 1:
                continue
            atom_index = int(atom_indices[0])
            if not 0 <= atom_index < len(symbols):
                raise ValueError(
                    f"Monoatomic fragment index {atom_index} is invalid for {parent!r}."
                )
            symbol = symbols[atom_index]
            if symbol != "Li":
                candidates.append(
                    (fragment_number, monomer_column, atom_indices, atom_index, symbol)
                )

        # Ignore Li before reading the dimer ratio: most Li rows store a scalar NaN for the
        # entire failed dimer MBIS result, and the explicit exception means that is acceptable.
        if not candidates:
            continue
        try:
            dimer_ratios = np.asarray(
                df.at[farthest_label, "volume ratios dimer"], dtype=float
            ).reshape(-1)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot read farthest-frame dimer volume ratios for {parent!r}."
            ) from exc
        if dimer_ratios.shape != (len(symbols),):
            raise ValueError(
                f"Farthest-frame dimer ratios for {parent!r} have shape "
                f"{dimer_ratios.shape}; expected {(len(symbols),)}."
            )

        for (
            fragment_number,
            monomer_column,
            atom_indices,
            atom_index,
            symbol,
        ) in candidates:
            reference = float(dimer_ratios[atom_index])
            if not np.isfinite(reference):
                raise ValueError(
                    f"The farthest frame of non-Li monoatomic trajectory {parent!r} has "
                    f"no finite dimer volume ratio for {symbol}."
                )
            if monomer_column not in df.columns:
                raise ValueError(
                    f"Trajectory {parent!r} is missing {monomer_column!r}."
                )

            # The source condition this repair addresses is one non-finite scalar per isolated
            # atom. Refuse to replace already finite or malformed data silently.
            for row_label in group.index:
                molecule = df.at[row_label, "qcel_molecule"]
                row_fragments = getattr(molecule, "fragments", None)
                row_symbols = list(map(str, getattr(molecule, "symbols", ())))
                if (
                    row_fragments is None
                    or len(row_fragments) != 2
                    or row_symbols != symbols
                    or not np.array_equal(
                        np.asarray(row_fragments[fragment_number], dtype=int).reshape(-1),
                        atom_indices,
                    )
                ):
                    raise ValueError(
                        f"Fragment/atom order changes within trajectory {parent!r}."
                    )
                try:
                    isolated = np.asarray(
                        df.at[row_label, monomer_column], dtype=float
                    ).reshape(-1)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Malformed {monomer_column!r} at {df.at[row_label, 'system_id']!r}."
                    ) from exc
                if isolated.shape != (1,) or np.isfinite(isolated).any():
                    raise ValueError(
                        f"Expected one non-finite isolated ratio at "
                        f"{df.at[row_label, 'system_id']!r}; got {isolated!r}."
                    )
                df.at[row_label, monomer_column] = np.asarray([reference], dtype=float)
                repaired_rows.add(int(row_label))
            repaired_parents.add(str(parent))

    if len(repaired_parents) != 30 or len(repaired_rows) != 300:
        raise ValueError(
            "Expected to repair exactly 30 Na/F/Cl monoatomic parents and 300 rows; "
            f"repaired {len(repaired_parents)} parents and {len(repaired_rows)} rows."
        )
    return df


def _normalize_source(path: Path, source_dataset: str) -> pd.DataFrame:
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"{path} must contain a non-empty pandas DataFrame.")
    df = dataframe_validation.normalize_columns(df).reset_index(drop=True)
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"{path} has duplicate columns after normalization: {duplicates}.")
    if "system_id" not in df.columns:
        raise ValueError(f"{path} has no system_id after normalization.")

    df = coalesce_sapt_aliases(df)
    df["_source_row"] = np.arange(len(df), dtype=int)
    df["_parent_system_id"] = df["system_id"].map(system_processing.system_group)
    if (df["_parent_system_id"].str.len() == 0).any():
        raise ValueError(f"{path} contains a blank parent system id.")
    if source_dataset == "NENCI-2021 non-S66x8":
        df = _fill_non_lithium_monoatomic_volume_ratios(df)
    df["source_dataset"] = source_dataset
    return df


def _inventory(df: pd.DataFrame, expected_numbers: range, frames_per_parent: int, label: str):
    parents = df["_parent_system_id"]
    counts = parents.value_counts(sort=False)
    by_number = {}
    for parent in counts.index:
        token = str(parent).split("_", 1)[0]
        if not token.isdigit():
            raise ValueError(f"{label} parent {parent!r} has no numeric NENCI prefix.")
        number = int(token)
        if number in by_number:
            raise ValueError(
                f"{label} parents {by_number[number]!r} and {parent!r} share prefix {number}."
            )
        by_number[number] = parent

    expected = set(expected_numbers)
    actual = set(by_number)
    if actual != expected:
        raise ValueError(
            f"{label} parent inventory differs: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}."
        )
    wrong = {parent: int(count) for parent, count in counts.items() if count != frames_per_parent}
    if wrong:
        raise ValueError(
            f"{label} trajectories must each have {frames_per_parent} frames; got {wrong}."
        )
    return set(counts.index)


def _read_assignments(path: Path) -> pd.DataFrame:
    assignments = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected_columns = ["parent_system_id", "binding_motif"]
    if list(assignments.columns) != expected_columns:
        raise ValueError(f"{path} columns must be exactly {expected_columns} in that order.")
    for column in expected_columns:
        assignments[column] = assignments[column].str.strip()
        if (assignments[column] == "").any():
            row = int(np.flatnonzero((assignments[column] == "").to_numpy())[0]) + 2
            raise ValueError(f"{path}:{row} has a blank {column}.")
    if assignments["parent_system_id"].duplicated().any():
        duplicates = assignments.loc[
            assignments["parent_system_id"].duplicated(keep=False), "parent_system_id"
        ].tolist()
        raise ValueError(f"Duplicate parent ids in {path}: {duplicates}.")

    # A motif's one contiguous block defines display order unambiguously.
    seen, previous = set(), None
    for motif in assignments["binding_motif"]:
        if motif != previous:
            if motif in seen:
                raise ValueError(f"Binding motif {motif!r} appears in disjoint CSV blocks.")
            seen.add(motif)
            previous = motif
    return assignments


def build(
    nenci_path: Path = DEFAULT_NENCI,
    s66_path: Path = DEFAULT_S66,
    map_path: Path = DEFAULT_MAP,
    output_path: Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    input_hashes = {path: _sha256(path) for path in (nenci_path, s66_path)}

    nenci = _normalize_source(nenci_path, "NENCI-2021 non-S66x8")
    s66 = _normalize_source(s66_path, "S66x8")
    available = _inventory(s66, range(1, 67), 8, "S66x8") | _inventory(
        nenci, range(67, 142), 10, "NENCI non-S66x8"
    )
    assignments = _read_assignments(map_path)
    requested = set(assignments["parent_system_id"])
    unknown = requested - available
    if unknown:
        raise ValueError(f"Motif map names unknown parents: {sorted(unknown)}.")

    combined = pd.concat([s66, nenci], ignore_index=True, sort=False)
    motif_by_parent = assignments.set_index("parent_system_id")["binding_motif"]
    rank_by_parent = {
        parent: rank for rank, parent in enumerate(assignments["parent_system_id"])
    }
    selected = combined[combined["_parent_system_id"].isin(requested)].copy()
    selected["binding_motif"] = selected["_parent_system_id"].map(motif_by_parent)
    selected["_mapping_rank"] = selected["_parent_system_id"].map(rank_by_parent)

    if selected["binding_motif"].isna().any():
        raise AssertionError("A retained row did not receive a binding motif.")
    if selected.groupby("_parent_system_id")["binding_motif"].nunique().ne(1).any():
        raise AssertionError("A retained trajectory received more than one binding motif.")

    selected = selected.sort_values(
        ["_mapping_rank", "_source_row"], kind="stable"
    ).reset_index(drop=True)
    selected = selected.drop(
        columns=["_mapping_rank", "_source_row", "_parent_system_id"]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_pickle(output_path)
    for path, before in input_hashes.items():
        after = _sha256(path)
        if after != before:
            raise RuntimeError(f"Input changed while building: {path}.")

    print(
        f"Wrote {output_path}: {len(selected)} frames, "
        f"{len(assignments)} systems, {assignments.binding_motif.nunique()} motifs."
    )
    return selected


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nenci", type=Path, default=DEFAULT_NENCI)
    parser.add_argument("--s66", type=Path, default=DEFAULT_S66)
    parser.add_argument("--map", dest="map_path", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = _args()
    build(args.nenci, args.s66, args.map_path, args.output)
