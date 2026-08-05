"""Convert the trusted source pickles in ``data/source/`` into the bundled ``.npz`` +
catalog that the Flask app serves.

Run by hand, never by the web app::

    pip install -r requirements-data.txt
    python scripts/build_ipd_dataset.py

The source DataFrames carry ``qcelemental.Molecule`` objects and an Arrow-backed column
index, so reading them needs pandas, pyarrow and qcelemental. Keeping that here means the
app itself never imports pandas and never unpickles anything: it only reads plain numeric
``.npz`` archives with ``allow_pickle=False``.

Output layout::

    data/ipd_catalog.json                    selector metadata for /api/ipd/systems
    data/na-water/01_Na-Water_1.00.npz       one file per geometry

Induced dipoles are stored under ``mu_history__<slug>`` with shape ``(n_frames, n_atoms, 3)``.
The ``mu_ind hist …`` source columns carry a real per-iteration SCF trajectory, and different
models converge in different numbers of iterations, so ``n_frames`` varies *between models in
the same file* as well as between geometries. Models backed by a converged-only column
(``mu_ind model``, ``mu_ind ref``) store a single frame.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CATALOG_PATH = DATA_DIR / "ipd_catalog.json"

# Running this as `python scripts/build_ipd_dataset.py` puts scripts/ on sys.path, not the
# repository root, so the backend package has to be reached explicitly.
sys.path.insert(0, str(REPO_ROOT))

from backend.services import dipole_data  # noqa: E402  (needs the path set above)

# qcelemental stores geometry in bohr; the viewer and the PyMOL script both work in Angstrom.
BOHR2ANG = 0.52917720859

@dataclass(frozen=True)
class Model:
    """One dipole model to publish, and the source columns it is built from.

    Declared rather than discovered: the source pickles carry dozens of ``mu_ind`` columns
    in several incompatible naming shapes (``mu_ind model A``, ``mu_ind A (radius thole
    a_est)``, ``mu_ind hist A (MBIS) 0.39 all``), and only a chosen few belong in the model
    selector. Naming the columns here also removes the guesswork from the energy mapping:
    a model reports a polarization energy only when this registry says which column it is.
    """

    slug: str
    label: str
    column_a: str
    column_b: str
    energy_column: str | None = None

    @property
    def columns(self) -> tuple[str, ...]:
        """Every source column this model needs, for the up-front existence check."""
        required = (self.column_a, self.column_b)
        return required if self.energy_column is None else required + (self.energy_column,)


@dataclass(frozen=True)
class ReferenceEnergy:
    """A per-system energy shown next to the models rather than as one of them.

    These describe the *geometry*, not a dipole model, so they do not move when the model
    selector changes -- which is why they are declared separately and displayed in their own
    group rather than mixed in with ``Model.energy_column``.
    """

    label: str
    column: str


@dataclass(frozen=True)
class Source:
    """One trusted pickle, the models to take from it, and the grouping metadata the
    selectors need.

    The source DataFrames have no ``dataset``/``system_name`` columns, so the grouping is
    declared here rather than parsed out of ``system_id`` strings at runtime.
    """

    path: Path
    dataset: str
    system_name: str
    out_dir: str
    separation_column: str
    separation_units: str
    separation_format: str
    models: tuple[Model, ...]
    reference_energies: tuple[ReferenceEnergy, ...] = ()
    alt_separation_column: str | None = None
    alt_separation_format: str = "{:.3f} Å"


@dataclass(frozen=True)
class ExtraNpz:
    """An already-converted ``.npz`` that only needs a catalog entry."""

    path: Path
    dataset: str
    system_name: str
    separation: float
    separation_units: str
    separation_label: str
    note: str = ""


# The four Thole/MBIS models below are backed by ``mu_ind hist`` columns, whose last frame is
# the converged dipole that the matching ``mu_ind A/B (...)`` column holds. ``model`` and
# ``ref`` have no history column upstream and so contribute a single frame each.
#
# ``mu_ind model`` and ``mu_ind ref`` get no energy: they are *probably* ``IPD (MBIS)`` and
# ``IPD ref (MBIS)``, but a wrong guess would put a mislabelled number in the UI, so nothing
# is emitted rather than something possibly wrong.
#
# Shared by every ion-water source: the pickles use identical column names, so one registry
# covers them and the model selector offers the same six options whichever ion is chosen.
ION_WATER_MODELS: tuple[Model, ...] = (
    Model(
        slug="model",
        label="mu_ind model",
        column_a="mu_ind model A",
        column_b="mu_ind model B",
    ),
    Model(
        slug="ref",
        label="mu_ind ref",
        column_a="mu_ind ref A",
        column_b="mu_ind ref B",
    ),
    Model(
        slug="radius_thole_ts_mbis_intra_inter",
        label="mu_ind hist (radius thole ts_mbis intra inter)",
        column_a="mu_ind hist A (radius thole ts_mbis intra inter)",
        column_b="mu_ind hist B (radius thole ts_mbis intra inter)",
        energy_column="IPD (radius thole ts_mbis intra inter)",
    ),
    Model(
        slug="radius_thole_ts_mbis_intra_only",
        label="mu_ind hist (radius thole ts_mbis intra only)",
        column_a="mu_ind hist A (radius thole ts_mbis intra only)",
        column_b="mu_ind hist B (radius thole ts_mbis intra only)",
        energy_column="IPD (radius thole ts_mbis intra only)",
    ),
    Model(
        slug="radius_thole_ts_mbis_inter_only",
        label="mu_ind hist (radius thole ts_mbis inter only)",
        column_a="mu_ind hist A (radius thole ts_mbis inter only)",
        column_b="mu_ind hist B (radius thole ts_mbis inter only)",
        energy_column="IPD (radius thole ts_mbis inter only)",
    ),
    Model(
        slug="mbis_0_39_all",
        label="mu_ind hist (MBIS) 0.39 all",
        column_a="mu_ind hist A (MBIS) 0.39 all",
        column_b="mu_ind hist B (MBIS) 0.39 all",
        energy_column="IPD (MBIS) 0.39 all",
    ),
)

# The SAPT induction benchmark for a geometry, shown as the reference the models are read
# against. The last three sum to the first -- verified exact on every row of both pickles --
# so they are ordered total-first and kept together.
#
# The unqualified ``SAPT`` columns, not the ``SAPT0/cc-pVDZ`` variants: only one of the two
# pickles carries those. The column names come from the backend so the offline converter
# and the on-demand catalog cannot drift apart about which columns these are.
SAPT_INDUCTION_ENERGIES: tuple[ReferenceEnergy, ...] = tuple(
    ReferenceEnergy(label=label, column=column)
    for label, column in dipole_data.SAPT_INDUCTION_ENERGIES
)

SOURCES: list[Source] = [
    Source(
        path=DATA_DIR / "source" / "na_water_dimers.pkl",
        dataset="Na-water",
        system_name="Na+-Water",
        out_dir="na-water",
        separation_column="eq_ratio",
        separation_units="Re",
        separation_format="{:.2f} Re",
        models=ION_WATER_MODELS,
        reference_energies=SAPT_INDUCTION_ENERGIES,
        alt_separation_column="O-Na dist ang",
    ),
    Source(
        path=DATA_DIR / "source" / "cl_water_dimers.pkl",
        dataset="Cl-water",
        system_name="Cl⁻-Water",
        out_dir="cl-water",
        separation_column="eq_ratio",
        separation_units="Re",
        separation_format="{:.2f} Re",
        models=ION_WATER_MODELS,
        reference_energies=SAPT_INDUCTION_ENERGIES,
        alt_separation_column="O-Cl dist ang",
    ),
]

EXTRA_NPZ: list[ExtraNpz] = [
    ExtraNpz(
        path=DATA_DIR / "01_Water-Water_1.00_history_SYNTHETIC.npz",
        dataset="Demo (synthetic)",
        system_name="Water-Water",
        separation=1.00,
        separation_units="Re",
        separation_label="1.00 Re",
        note="Synthetic history with a fabricated damped oscillation; the only system here "
        "with more than one frame.",
    ),
]


def slugify(name: str) -> str:
    """Turn a model label into an identifier safe for an ``.npz`` key and a URL."""
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    if not slug:
        raise ValueError(f"Model name {name!r} slugifies to an empty string")
    return slug


def validate_models(models: tuple[Model, ...], where: str) -> None:
    """Check a hand-written registry before it reaches an ``.npz`` key or a URL.

    Slugs are typed out by hand here rather than derived, so this is what stops a stray
    space or bracket from producing a key the app cannot round-trip, and what catches two
    models quietly sharing one ``mu_history__<slug>`` array.
    """
    if not models:
        raise SystemExit(f"{where}: no dipole models declared")

    seen: set[str] = set()
    for model in models:
        if slugify(model.slug) != model.slug:
            raise SystemExit(
                f"{where}: slug {model.slug!r} is not URL/npz safe "
                f"(expected {slugify(model.slug)!r})"
            )
        if model.slug in seen:
            raise SystemExit(f"{where}: duplicate model slug {model.slug!r}")
        seen.add(model.slug)


def stack_dipoles(row, model: Model, n_a: int, n_b: int) -> np.ndarray:
    """Concatenate a model's monomer-A and monomer-B dipoles into ``(n_frames, n_atoms, 3)``.

    Two column shapes feed this: a converged ``(n_x, 3)`` column, which becomes a
    single-frame history, and a ``mu_ind hist`` column already shaped
    ``(n_frames, n_x, 3)``. The halves join along the *atom* axis, and the A-then-B order
    is what makes the result line up with ``coords``; the shape checks below are what prove
    it, since a transposed or mis-sized half would otherwise stack into a superficially
    valid array.
    """
    halves = []
    for column, expected in ((model.column_a, n_a), (model.column_b, n_b)):
        mu = np.asarray(row[column], dtype=float)
        if mu.ndim == 2:
            mu = mu[np.newaxis, :, :]
        if mu.ndim != 3 or mu.shape[1:] != (expected, 3):
            raise ValueError(
                f"{column!r} must have shape ({expected}, 3) or (n_frames, {expected}, 3) "
                f"to match its fragment; got {mu.shape}"
            )
        halves.append(mu)

    mu_a, mu_b = halves
    if mu_a.shape[0] != mu_b.shape[0]:
        raise ValueError(
            f"Model {model.slug!r}: {model.column_a!r} has {mu_a.shape[0]} frames but "
            f"{model.column_b!r} has {mu_b.shape[0]}; the two monomers must share an "
            "SCF iteration axis"
        )

    mu = np.concatenate([mu_a, mu_b], axis=1)
    if not np.isfinite(mu).all():
        raise ValueError(f"Model {model.slug!r} contains non-finite dipole components")
    return mu


def convert_row(row, source: Source) -> dict[str, Any]:
    """Build one system's arrays and catalog entry from a DataFrame row."""
    system_id = str(row["system_id"])
    molecule = row["qcel_molecule"]

    symbols = np.asarray([str(s) for s in molecule.symbols])
    coords = np.asarray(molecule.geometry, dtype=float) * BOHR2ANG
    n_atoms = len(symbols)

    if coords.shape != (n_atoms, 3):
        raise ValueError(f"{system_id}: geometry has shape {coords.shape}, expected ({n_atoms}, 3)")
    if not np.isfinite(coords).all():
        raise ValueError(f"{system_id}: geometry contains non-finite values")

    fragments = [np.asarray(f) for f in molecule.fragments]
    if len(fragments) != 2:
        raise ValueError(f"{system_id}: expected 2 fragments, got {len(fragments)}")
    n_a, n_b = len(fragments[0]), len(fragments[1])
    if n_a + n_b != n_atoms:
        raise ValueError(
            f"{system_id}: fragments cover {n_a + n_b} atoms but the molecule has {n_atoms}"
        )
    # The dipoles are concatenated A-then-B, so the fragments must partition the atoms in
    # that same contiguous order for the stacked array to line up with coords.
    expected_order = np.arange(n_atoms)
    if not np.array_equal(np.concatenate(fragments), expected_order):
        raise ValueError(
            f"{system_id}: fragments are not contiguous A-then-B "
            f"({fragments[0].tolist()} + {fragments[1].tolist()})"
        )

    models = source.models
    arrays: dict[str, Any] = {
        "coords": coords,
        "symbols": symbols,
        "n_atoms_A": np.int64(n_a),
        "system_id": np.asarray(system_id),
        "models": np.asarray([model.slug for model in models]),
        "model_labels": np.asarray([model.label for model in models]),
    }

    catalog_models = []
    max_abs_mu = 0.0
    for model in models:
        # Models converge in different numbers of iterations, so the frame counts within one
        # system legitimately differ; the app selects one model at a time and reads its own
        # leading axis, so there is nothing to reconcile here.
        mu = stack_dipoles(row, model, n_a, n_b)
        arrays[f"mu_history__{model.slug}"] = mu
        max_abs_mu = max(max_abs_mu, float(np.linalg.norm(mu, axis=-1).max()))

        catalog_models.append(
            {
                "slug": model.slug,
                "label": model.label,
                "n_frames": int(mu.shape[0]),
                "energy_kcalmol": (
                    round(float(row[model.energy_column]), 6)
                    if model.energy_column is not None
                    else None
                ),
            }
        )

    # One scale for every model at this geometry, so switching models compares arrows
    # rather than renormalizing each one to its own longest vector.
    arrays["max_abs_mu_system"] = np.float64(max_abs_mu)

    separation = float(row[source.separation_column])
    entry = {
        "system_id": system_id,
        "dataset": source.dataset,
        "system_name": source.system_name,
        "separation": separation,
        "separation_units": source.separation_units,
        "separation_label": source.separation_format.format(separation),
        "separation_alt_label": (
            source.alt_separation_format.format(float(row[source.alt_separation_column]))
            if source.alt_separation_column
            else None
        ),
        "n_atoms": n_atoms,
        # Summary only -- the longest history at this geometry. The per-model counts in
        # ``models`` are the real thing, and the app reads the frame count off the model it
        # actually loaded.
        "n_frames": max(model["n_frames"] for model in catalog_models),
        "max_abs_mu_system": round(max_abs_mu, 6),
        "models": catalog_models,
        # A list, not a dict: the order is part of the meaning, since the total comes first
        # and the terms that sum to it follow. Per-system, so it stays out of the .npz for
        # the same reason the per-model energies do.
        "reference_energies": [
            {"label": ref.label, "kcalmol": round(float(row[ref.column]), 6)}
            for ref in source.reference_energies
        ],
        "file": f"{source.out_dir}/{system_id}.npz",
    }
    return {"arrays": arrays, "entry": entry}


def convert_source(source: Source) -> Iterator[dict[str, Any]]:
    """Yield one converted system per row of a source pickle."""
    import pandas as pd  # imported lazily so --help works without the ETL extras

    if not source.path.is_file():
        raise SystemExit(f"Source pickle not found: {source.path}")

    validate_models(source.models, source.path.name)

    df = pd.read_pickle(source.path)
    for column in ("system_id", "qcel_molecule", source.separation_column):
        if column not in df.columns:
            raise SystemExit(f"{source.path.name} is missing the required column {column!r}")

    # A registry typo must fail the build loudly. Silently dropping the model or the energy
    # instead would leave a panel that is short by one row and nothing to say why.
    available = set(df.columns)
    declared = [column for model in source.models for column in model.columns]
    declared += [ref.column for ref in source.reference_energies]
    missing = [column for column in declared if column not in available]
    if missing:
        raise SystemExit(
            f"{source.path.name} is missing {len(missing)} declared model column(s):\n    "
            + "\n    ".join(repr(column) for column in missing)
        )

    print(f"{source.path.name}: {len(df)} rows, {len(source.models)} dipole models")
    for model in source.models:
        print(f"    {model.slug:42} {model.label}")

    seen_separations: set[float] = set()
    for _, row in df.iterrows():
        converted = convert_row(row, source)
        separation = converted["entry"]["separation"]
        if separation in seen_separations:
            raise SystemExit(
                f"{source.path.name}: duplicate separation {separation} in "
                f"{source.dataset}/{source.system_name}"
            )
        seen_separations.add(separation)
        yield converted


def catalog_entry_for_npz(extra: ExtraNpz) -> dict[str, Any]:
    """Describe an existing ``.npz`` without rewriting it.

    Files predating the model axis carry a bare ``mu_history``; they are presented as a
    single model named ``history`` so the same catalog shape covers both.
    """
    if not extra.path.is_file():
        raise SystemExit(f"Bundled history not found: {extra.path}")

    with np.load(extra.path, allow_pickle=False) as data:
        mu = np.asarray(data["mu_history"], dtype=float)
        symbols = data["symbols"]

    return {
        "system_id": extra.path.stem,
        "dataset": extra.dataset,
        "system_name": extra.system_name,
        "separation": extra.separation,
        "separation_units": extra.separation_units,
        "separation_label": extra.separation_label,
        "separation_alt_label": None,
        "n_atoms": len(symbols),
        "n_frames": int(mu.shape[0]),
        "max_abs_mu_system": round(float(np.linalg.norm(mu, axis=-1).max()), 6),
        "models": [
            {
                "slug": "history",
                "label": "SCF history",
                "n_frames": int(mu.shape[0]),
                "energy_kcalmol": None,
            }
        ],
        "file": extra.path.name,
        "note": extra.note or None,
    }


def main() -> int:
    '''

    Read through all the pickle files in the data directory.
    '''
    catalog: list[dict[str, Any]] = []

    
    for source in SOURCES:
        out_dir = DATA_DIR / source.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for converted in convert_source(source):
            entry = converted["entry"]
            np.savez_compressed(DATA_DIR / entry["file"], **converted["arrays"])
            catalog.append(entry)
            frames = ",".join(str(model["n_frames"]) for model in entry["models"])
            print(
                f"  wrote {entry['file']:44} "
                f"{entry['n_atoms']} atoms, frames per model [{frames}], "
                f"max|mu| {entry['max_abs_mu_system']:.4f} a.u."
            )

    for extra in EXTRA_NPZ:
        entry = catalog_entry_for_npz(extra)
        catalog.append(entry)
        print(f"  registered {entry['file']} ({entry['n_frames']} frames)")

    ids = [entry["system_id"] for entry in catalog]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise SystemExit(f"Duplicate system_id across sources: {duplicates}")

    CATALOG_PATH.write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"\n{CATALOG_PATH.relative_to(REPO_ROOT)}: {len(catalog)} systems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
