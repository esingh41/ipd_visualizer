"""Assemble one dimer scan into a playable trajectory: N geometries, one per separation.

The Visualize tab animates the SCF at a *fixed* geometry -- the nuclei never move, only the
arrows. This module answers the other question: what does the dimer look like as it is pulled
apart? A trajectory is every catalog entry sharing a (dataset, system_name), ordered by
separation, each contributing its geometry and its *converged* induced dipoles.

Nothing new is read off disk. Both loaders already return
:func:`backend.services.dipole_data.build_history_payload`'s shape, so a frame is that payload
minus the SCF axis: ``coords``/``xyz``/``symbols`` as-is, and ``mu_history[-1]`` for the
arrows. This module only orders, merges and checks them.

It lives apart from ``dipole_data`` because it needs both sources of a history -- the bundled
``.npz`` files and the on-demand dataframe results -- and ``dipole_data`` must not import
``ipd_compute``; the dependency runs the other way. ``trajectory -> {dipole_data,
ipd_compute}`` keeps that direction intact.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from backend.services import dipole_data, ipd_compute


def _catalog() -> List[Dict[str, Any]]:
    """The merged catalog, exactly as ``GET /api/ipd/systems`` builds it.

    Bundled entries plus one per computed geometry, so an uploaded scan is a trajectory on
    the same terms as a pre-baked one.
    """
    return dipole_data.list_systems() + ipd_compute.catalog_entries()


def _load_payload(entry: Dict[str, Any], model: str) -> Dict[str, Any]:
    """One geometry's history payload, from whichever source owns it.

    Mirrors the dispatch in ``backend.app.ipd_system``: a dataset-scoped ``system_id`` comes
    from the uploaded dataframe, anything else from the bundled catalog.
    """
    scoped = ipd_compute.parse_dataset_system_id(entry["system_id"])
    if scoped is not None:
        dataset_id, row_id = scoped
        return ipd_compute.load_stored(dataset_id, row_id, model)
    return dipole_data.get_system(entry["system_id"], model)


def _shared_models(entries: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Models offered at *every* separation, in the first entry's order.

    An intersection rather than a union: a model missing from one geometry would leave a hole
    in the middle of the animation, and a trajectory with a frame missing is worse than a
    trajectory that is not offered. Bundled scans carry all six slugs at every separation; a
    partially computed upload need not.
    """
    shared = set.intersection(
        *({model["slug"] for model in entry.get("models", [])} for entry in entries)
    )
    return [
        {"slug": model["slug"], "label": model["label"]}
        for model in entries[0].get("models", [])
        if model["slug"] in shared
    ]


def build_trajectory(
    dataset: str, system_name: str, model: Optional[str] = None
) -> Dict[str, Any]:
    """Every geometry of one scan, in separation order, with its converged dipoles.

    ``model`` selects which polarization model's converged dipoles the arrows show; ``None``
    takes the first one available at every separation.

    Raises :class:`dipole_data.SystemNotFound` for an unknown scan (404) and ``ValueError``
    for an unknown model or an internally inconsistent scan (400), both of which the Flask
    layer already has handlers for.
    """
    entries = sorted(
        (
            entry
            for entry in _catalog()
            if entry.get("dataset") == dataset
            and entry.get("system_name") == system_name
        ),
        key=lambda entry: entry["separation"],
    )
    if not entries:
        raise dipole_data.SystemNotFound(
            f"No system named {system_name!r} in dataset {dataset!r}"
        )

    models = _shared_models(entries)
    available = [candidate["slug"] for candidate in models]
    if not available:
        raise ValueError(
            f"The geometries of {system_name!r} share no dipole model, so they cannot be "
            "animated as one trajectory"
        )
    if model is None:
        model = available[0]
    elif model not in available:
        raise ValueError(
            f"Unknown dipole model {model!r} for {system_name!r}; "
            f"available at every separation: {', '.join(available)}"
        )

    frames: List[Dict[str, Any]] = []
    max_abs_mu = 0.0
    symbols: List[str] = []
    n_atoms_A: Optional[int] = None

    for entry in entries:
        payload = _load_payload(entry, model)

        # A single atom out of order between two frames renders as a plausible but wrong
        # animation -- the same failure mode, and so the same fatal treatment, as the
        # atom-count check in build_history_payload.
        if not frames:
            symbols = payload["symbols"]
            n_atoms_A = payload["n_atoms_A"]
        elif payload["symbols"] != symbols:
            raise ValueError(
                f"Geometry {entry['system_id']!r} does not match the rest of "
                f"{system_name!r}: symbols {payload['symbols']} vs {symbols}"
            )
        elif payload["n_atoms_A"] != n_atoms_A:
            raise ValueError(
                f"Geometry {entry['system_id']!r} splits its monomers at "
                f"{payload['n_atoms_A']} atoms, the rest of {system_name!r} at {n_atoms_A}"
            )

        # The last SCF iteration: these arrows are all converged, so one frame's arrow is
        # comparable with the next one's. Already rounded by build_history_payload.
        mu = payload["mu_history"][-1]
        magnitudes = np.linalg.norm(np.asarray(mu, dtype=float), axis=-1)
        if magnitudes.size:
            max_abs_mu = max(max_abs_mu, float(magnitudes.max()))

        frames.append(
            {
                "system_id": entry["system_id"],
                "separation": entry["separation"],
                "separation_units": entry.get("separation_units"),
                "separation_label": entry.get("separation_label"),
                "separation_alt_label": entry.get("separation_alt_label"),
                "contact_distance_ang": entry.get("contact_distance_ang"),
                "coords": payload["coords"],
                "xyz": payload["xyz"],
                "mu": mu,
                "energy_kcalmol": payload.get("energy_kcalmol"),
                "reference_energies": payload.get("reference_energies", []),
            }
        )

    return {
        "dataset": dataset,
        "system_name": system_name,
        "model": model,
        "models": models,
        "n_frames": len(frames),
        "n_atoms": len(symbols),
        "n_atoms_A": n_atoms_A,
        "symbols": symbols,
        "frames": frames,
        # One scale for the whole scan, so a shrinking arrow means a shrinking dipole across
        # separations -- the invariant the Visualize tab keeps across SCF iterations. Not the
        # catalog's max_abs_mu_system, which maxes over every *iteration* and every model at
        # one geometry and would leave these converged arrows stunted.
        "max_abs_mu": max_abs_mu,
        "metadata": {
            "coordinate_units": "angstrom",
            "dipole_units": "atomic_units",
            "frame": "converged_induced_dipoles",
        },
    }
