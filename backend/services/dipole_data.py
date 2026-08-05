"""Load per-atom induced-dipole data from ``.npz`` into JSON-ready dicts, and index the
bundled systems for the selector catalog.

Every file read here is a plain numeric ``.npz`` produced by ``scripts/build_ipd_dataset.py``
and read with ``allow_pickle=False``. The source ``.pkl`` files are converted offline, so
nothing in the running app ever deserializes a pickle.

    mu_history__<slug>  (n_frames, n_atoms, 3) f8  induced dipoles for one polarization
                                                   model, atomic units (e*bohr),
                                                   monA-then-monB. n_frames is per model:
                                                   models in one file converge in different
                                                   numbers of SCF iterations.
    coords              (n_atoms, 3)           f8  fixed geometry, Angstrom, same atom order
    symbols             (n_atoms,)             <U2 element symbols
    models              (n_models,)            <U  model slugs, ordered
    model_labels        (n_models,)            <U  display names for those slugs
    max_abs_mu_system   scalar                 f8  max |mu| over every model and frame
    n_atoms_A           scalar                 i8  optional; atom count of monomer A
    system_id           scalar                 <U  optional; label for the dimer

Older files (the bundled synthetic history) carry a bare ``mu_history`` with no ``models``
key; they are presented as a single model named ``history`` so one code path covers both.

Frame 0 of a real SCF history is the direct-field seed, not zero, so it already carries a
full set of arrows. Models whose source column records only the converged dipole store a
single frame, so a one-frame history is normal and not a sign of a truncated file.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CATALOG_PATH = DATA_DIR / "ipd_catalog.json"

# Dipole components below this magnitude are dropped by the renderer; keeping more than
# six decimals in the JSON payload is wasted bytes. Matches MIN_MU in pymol_dipole.py.
MU_DECIMALS = 6

# Name used for a file that predates the model axis and stores a bare ``mu_history``.
LEGACY_MODEL = "history"

# The SAPT induction benchmark for a geometry, shown as the reference the models are read
# against. The last three sum to the first, so they are ordered total-first and kept
# together. These describe the *geometry*, not a dipole model, which is why they are
# displayed in their own group rather than mixed in with the per-model energies.
#
# Declared here rather than in scripts/build_ipd_dataset.py because both the offline
# converter and the on-demand catalog need the same four column names, and a second copy
# would eventually disagree with this one.
SAPT_INDUCTION_ENERGIES = (
    ("Induction total", "SAPT0 IND kcalmol"),
    ("ind20,r", "SAPT ind20,r kcalmol"),
    ("exch-ind20,r", "SAPT exch-ind20,r kcalmol"),
    ("δHF", "SAPT dHF ind kcalmol"),
)


class SystemNotFound(LookupError):
    """Raised when a requested ``system_id`` is not in the catalog.

    A distinct type so the Flask layer can answer 404 without this module importing Flask.
    """


def _build_xyz(symbols, coords, comment: str) -> str:
    """Format an XYZ string for ``viewer.addModel(xyz, "xyz")``."""
    lines = [str(len(symbols)), comment or "ipd-visualizer"]
    for symbol, (x, y, z) in zip(symbols, coords):
        lines.append(f"{symbol} {x:.8f} {y:.8f} {z:.8f}")
    return "\n".join(lines) + "\n"


def _read_arrays(path: Path) -> Dict[str, Any]:
    """Read an ``.npz`` of plain numeric arrays, or raise ValueError."""
    if not path.is_file():
        raise ValueError(f"Dipole history not found: {path}")

    try:
        # allow_pickle stays off: these files are plain numeric arrays.
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except Exception:  # zipfile/npy errors are not a single exception type
        # Deliberately not echoing numpy's message: for a non-npz file it suggests
        # re-reading with allow_pickle=True, which is exactly what must not happen.
        raise ValueError(
            f"Could not read '{path.name}' as a .npz archive of plain numeric arrays"
        )


def _model_table(arrays: Dict[str, Any]) -> List[Dict[str, str]]:
    """List the models a file offers, as ``{"slug", "label"}`` in stored order."""
    if "models" not in arrays:
        return [{"slug": LEGACY_MODEL, "label": "SCF history"}]

    slugs = [str(slug) for slug in arrays["models"]]
    labels = (
        [str(label) for label in arrays["model_labels"]]
        if "model_labels" in arrays
        else slugs
    )
    if len(labels) != len(slugs):
        raise ValueError("models and model_labels disagree on length")
    return [{"slug": slug, "label": label} for slug, label in zip(slugs, labels)]


def build_history_payload(
    mu,
    coords,
    symbols,
    *,
    system_id: str = "",
    model: str = "",
    models: List[Dict[str, str]] | None = None,
    n_atoms_A: int | None = None,
    max_abs_mu: float | None = None,
    source: str = "",
    mu_label: str = "mu_history",
) -> Dict[str, Any]:
    """Validate one SCF history and assemble the dict the viewer consumes.

    The single definition of the visualizer's data contract. Both sources of a history go
    through here -- the bundled ``.npz`` files via :func:`load_history_npz`, and the
    on-demand results ``backend.services.ipd_compute`` computes -- so there is no second,
    drifting representation of the same thing.

    ``mu_label`` only names the array in error messages; it is what tells a reader which
    ``.npz`` key or dataframe column is at fault.
    """
    mu = np.asarray(mu, dtype=float)
    coords = np.asarray(coords, dtype=float)
    symbols = [str(symbol) for symbol in symbols]

    if mu.ndim != 3 or mu.shape[2] != 3:
        raise ValueError(
            f"{mu_label} must have shape (n_frames, n_atoms, 3); got {mu.shape}"
        )
    if mu.shape[0] < 1:
        raise ValueError(f"{mu_label} must contain at least one frame")

    n_frames, n_atoms = mu.shape[0], mu.shape[1]

    # A silent atom-ordering mismatch produces a plausible but scientifically wrong
    # animation, so this is fatal rather than a warning.
    if not (n_atoms == len(coords) == len(symbols)):
        raise ValueError(
            "Atom counts disagree: "
            f"{mu_label} has {n_atoms}, coords has {len(coords)}, symbols has {len(symbols)}"
        )
    if coords.shape != (n_atoms, 3):
        raise ValueError(f"coords must have shape ({n_atoms}, 3); got {coords.shape}")

    if not np.isfinite(mu).all():
        raise ValueError(f"{mu_label} contains non-finite values")
    if not np.isfinite(coords).all():
        raise ValueError("coords contains non-finite values")

    if n_atoms_A is not None:
        n_atoms_A = int(n_atoms_A)
        if not 0 < n_atoms_A < n_atoms:
            raise ValueError(
                f"n_atoms_A must be between 1 and {n_atoms - 1}; got {n_atoms_A}"
            )

    # One global scale across every frame, atom *and model* is derived from this on the
    # client, so a shrinking arrow always means a shrinking dipole -- including when the
    # model selector changes at a fixed geometry.
    if max_abs_mu is None:
        max_abs_mu = float(np.linalg.norm(mu, axis=-1).max())

    return {
        "system_id": system_id,
        "model": model,
        "models": models if models is not None else [],
        "n_frames": n_frames,
        "n_atoms": n_atoms,
        "n_atoms_A": n_atoms_A,
        "symbols": symbols,
        "coords": coords.tolist(),
        "xyz": _build_xyz(symbols, coords, system_id),
        "mu_history": np.round(mu, MU_DECIMALS).tolist(),
        "max_abs_mu": float(max_abs_mu),
        "metadata": {
            "coordinate_units": "angstrom",
            "dipole_units": "atomic_units",
            "frame_zero": "direct_field_seed",
            "source_file": source,
        },
    }


def load_history_npz(path: Path | str, model: str | None = None) -> Dict[str, Any]:
    """Read and validate one model's SCF history, returning a JSON-serializable dict.

    ``model`` selects which polarization model's dipoles to return; ``None`` takes the first
    one in the file. Raises ValueError for anything malformed so the Flask error handler
    turns it into a 400 with an actionable message.
    """
    path = Path(path)
    arrays = _read_arrays(path)

    for key in ("coords", "symbols"):
        if key not in arrays:
            raise ValueError(f"'{path.name}' is missing the required key '{key}'")

    models = _model_table(arrays)
    available = [entry["slug"] for entry in models]
    if model is None:
        model = available[0]
    elif model not in available:
        raise ValueError(
            f"Unknown dipole model '{model}' for {path.name}; available: {', '.join(available)}"
        )

    mu_key = "mu_history" if model == LEGACY_MODEL else f"mu_history__{model}"
    if mu_key not in arrays:
        raise ValueError(f"'{path.name}' is missing the required key '{mu_key}'")

    n_atoms_A = (
        int(np.asarray(arrays["n_atoms_A"]).item()) if "n_atoms_A" in arrays else None
    )
    system_id = (
        str(np.asarray(arrays["system_id"]).item()) if "system_id" in arrays else ""
    )
    max_abs_mu = (
        float(np.asarray(arrays["max_abs_mu_system"]).item())
        if "max_abs_mu_system" in arrays
        else None
    )

    return build_history_payload(
        arrays[mu_key],
        arrays["coords"],
        arrays["symbols"],
        system_id=system_id,
        model=model,
        models=models,
        n_atoms_A=n_atoms_A,
        max_abs_mu=max_abs_mu,
        source=path.name,
        mu_label=mu_key,
    )


# --- catalog ----------------------------------------------------------------


@lru_cache(maxsize=1)
def load_catalog() -> List[Dict[str, Any]]:
    """Read ``data/ipd_catalog.json``, validating that it can drive the selectors.

    Cached: the catalog is a build artifact, so it does not change while the app runs.
    """
    if not CATALOG_PATH.is_file():
        raise RuntimeError(
            f"System catalog not found at {CATALOG_PATH}. "
            "Run 'python scripts/build_ipd_dataset.py' to generate it."
        )

    try:
        catalog = json.loads(CATALOG_PATH.read_text())
    except json.JSONDecodeError as err:
        raise RuntimeError(f"System catalog is not valid JSON: {err}")

    if not isinstance(catalog, list) or not catalog:
        raise RuntimeError("System catalog must be a non-empty list of systems")

    required = ("system_id", "dataset", "system_name", "separation", "file")
    seen = set()
    for entry in catalog:
        missing = [key for key in required if key not in entry]
        if missing:
            raise RuntimeError(
                f"Catalog entry {entry.get('system_id', '<unnamed>')} is missing {missing}"
            )
        # The three selectors resolve to exactly one row, so a repeated id would make one
        # of the geometries unreachable.
        if entry["system_id"] in seen:
            raise RuntimeError(f"Duplicate system_id in catalog: {entry['system_id']}")
        seen.add(entry["system_id"])

    return catalog


def list_systems() -> List[Dict[str, Any]]:
    """Selector and summary metadata for every bundled system.

    Deliberately carries no coordinates and no dipole histories: this is what the page
    fetches once at boot to build the four connected selectors.
    """
    return load_catalog()


def _resolve_file(entry: Dict[str, Any]) -> Path:
    """Turn a catalog entry's ``file`` into an absolute path inside ``data/``.

    The path comes from the catalog, never from a query parameter, but it is still
    confined to ``data/`` so a hand-edited catalog cannot reach outside the repo.
    """
    path = (DATA_DIR / entry["file"]).resolve()
    if not path.is_relative_to(DATA_DIR.resolve()):
        raise RuntimeError(f"Catalog entry {entry['system_id']} points outside data/")
    return path


def get_system(system_id: str, model: str | None = None) -> Dict[str, Any]:
    """Load one system by id, merging its catalog metadata into the payload."""
    for entry in load_catalog():
        if entry["system_id"] == system_id:
            break
    else:
        raise SystemNotFound(f"Unknown system_id: {system_id}")

    payload = load_history_npz(_resolve_file(entry), model)
    payload["system_id"] = entry["system_id"]
    payload["dataset"] = entry["dataset"]
    payload["system_name"] = entry["system_name"]
    payload["separation"] = entry["separation"]
    payload["separation_units"] = entry.get("separation_units")
    payload["separation_label"] = entry.get("separation_label")
    payload["separation_alt_label"] = entry.get("separation_alt_label")
    payload["note"] = entry.get("note")

    # The catalog carries the per-model energies; the .npz does not.
    energies = {m["slug"]: m.get("energy_kcalmol") for m in entry.get("models", [])}
    for entry_model in payload["models"]:
        entry_model["energy_kcalmol"] = energies.get(entry_model["slug"])
    payload["energy_kcalmol"] = energies.get(payload["model"])

    # Per-system rather than per-model, so this does not change when ``model`` does. Absent
    # for a source with no such data, which is a bare list rather than a missing key.
    payload["reference_energies"] = entry.get("reference_energies", [])

    return payload
