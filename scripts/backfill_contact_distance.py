"""Add ``contact_distance_ang`` to the bundled catalog without regenerating anything.

One-shot. Run once::

    python scripts/backfill_contact_distance.py

``scripts/build_ipd_dataset.py`` now emits this field itself, so the next full rebuild
produces it natively and this script can be deleted.

It exists because a full rebuild is the expensive way to add one number. The bundled
``.npz`` files already store ``coords`` (Angstrom) and ``n_atoms_A``, which is everything
the distance needs -- so this reads them directly and never touches ``data/source/*.pkl``.
That avoids pandas, pyarrow, and in particular qcelemental, whose Molecule class has to sit
at the same import path as the one that wrote a pickle for it to be readable at all
(see requirements-data.txt). numpy alone is enough here.

Idempotent: rerunning recomputes the same values and rewrites the same file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CATALOG_PATH = DATA_DIR / "ipd_catalog.json"

# Running this as `python scripts/backfill_contact_distance.py` puts scripts/ on sys.path,
# not the repository root, so the backend package has to be reached explicitly.
sys.path.insert(0, str(REPO_ROOT))

from backend.services import dipole_data  # noqa: E402  (needs the path set above)

FIELD = "contact_distance_ang"
PLACES = 4


def contact_for(entry: dict[str, Any]) -> float | None:
    """Read one catalog entry's .npz and measure its closest A-to-B contact."""
    relative = entry.get("file")
    if not relative:
        return None

    path = (DATA_DIR / relative).resolve()
    # The path comes from the catalog rather than user input, but it is still confined to
    # data/ so a hand-edited catalog cannot reach outside the repo -- the same guard
    # dipole_data._resolve_file applies.
    if not path.is_relative_to(DATA_DIR.resolve()):
        raise SystemExit(f"Catalog entry {entry['system_id']} points outside data/")
    if not path.is_file():
        raise SystemExit(f"Missing .npz for {entry['system_id']}: {path}")

    with np.load(path, allow_pickle=False) as data:
        if "coords" not in data.files or "n_atoms_A" not in data.files:
            return None
        coords = data["coords"]
        n_atoms_a = int(data["n_atoms_A"])

    return dipole_data.contact_distance(coords, n_atoms_a)


def main() -> int:
    catalog = json.loads(CATALOG_PATH.read_text())

    measured = 0
    skipped = []
    for entry in catalog:
        value = contact_for(entry)
        entry[FIELD] = None if value is None else round(value, PLACES)
        if value is None:
            skipped.append(entry["system_id"])
            continue
        measured += 1

        # The ion-water sources carry a hand-made "O-Na dist ang" / "O-Cl dist ang" column,
        # formatted into separation_alt_label. That is an oxygen-to-ion distance, which is
        # only the closest contact when the ion coordinates to the oxygen -- true for Na+,
        # false for Cl-, where the water donates a hydrogen bond and the contact is H...Cl,
        # shorter by about an O-H bond length. So a difference here is expected for anions
        # and is reported rather than treated as an error. What would be alarming is a
        # difference for Na-water, which shares the geometry convention.
        label = entry.get("separation_alt_label") or ""
        if label.endswith("Å"):
            try:
                existing = float(label.split()[0])
            except (ValueError, IndexError):
                continue
            if abs(existing - value) > 5e-3:
                print(
                    f"  - {entry['system_id']}: contact {value:.4f} A, "
                    f"O-to-ion {existing:.4f} A"
                )

    CATALOG_PATH.write_text(json.dumps(catalog, indent=2) + "\n")

    print(f"{CATALOG_PATH.relative_to(REPO_ROOT)}: {measured}/{len(catalog)} measured")
    if skipped:
        print(f"  no geometry in the .npz, left null: {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
