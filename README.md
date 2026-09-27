# IPD Visualizer

IPD Visualizer is a local Flask application for exploring intermolecular dimer scans. It combines an interactive 3D trajectory viewer with permanent MBIS multipoles, induced-point-dipole (IPD) histories, Delta-MTP results, and available SAPT or force-field energy curves.

A view-only eight-frame **S66x8 water–water** scan is bundled with the repository, so a fresh clone has something to explore immediately.

## Features

- Play and inspect dimer scan trajectories in 3D.
- Compare isolated-monomer and dimer MBIS charges and atomic dipoles.
- Display induced-dipole SCF histories when stored results are available.
- Compute IPD and Delta-MTP results when the optional `apnet_pt` APIs are installed.
- Plot available SAPT, multipole, AMOEBA+, IPD, and Delta-MTP energies.
- Organize compatible collections by binding motif.
- Measure atom-to-atom distances directly in either 3D viewer.

## Quick start

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install flask pandas numpy qcelemental
python app.py
```

Then open <http://127.0.0.1:5000>.

The frontend loads 3Dmol.js and Plotly from CDNs, so those visualizations require internet access on first load.

To use the optional IPD and Delta-MTP computation features, install a compatible build of `apnet_pt` and its PyTorch dependencies in the same environment. The application reports computation capability independently from ordinary trajectory viewing.

## Using your own data

Use the upload control to register a trusted `.pkl` or `.pickle` file containing a non-empty pandas DataFrame. The only required column is `qcel_molecule`; every usable molecule must be a two-fragment `qcelemental.Molecule`. A `system_id` column is recommended for grouping scan frames into trajectories. Additional MBIS, energy, and stored-result columns enable the corresponding analysis features.

Uploaded collections are written under:

```text
data/systems/<collection-id>/
```

This runtime data is intentionally excluded from Git. It remains on the local machine between application runs.

> **Security warning:** Python pickles can execute arbitrary code when opened. Upload only files you trust, and keep the application bound to loopback unless you deliberately accept the risk of exposing an upload endpoint.

## Bundled sample

The repository includes serialized, browser-ready JSON for the eight-frame `01_Water-Water` trajectory from S66x8. It demonstrates geometry, permanent multipoles, stored energies, plotting, and trajectory controls without opening a pickle at startup.

The sample is intentionally read-only and does not include a processed pickle. On-demand IPD and Delta-MTP computation is therefore disabled for it. Uploading a compatible full-fidelity DataFrame enables computation when the optional scientific dependencies are available.

## Configuration

The development server binds to `127.0.0.1:5000` by default. Override these values deliberately with:

```bash
IPD_HOST=127.0.0.1 IPD_PORT=5001 python app.py
```

Changing `IPD_HOST` to a network-facing address changes the security posture because the application accepts and unpickles uploaded files.

## API overview

- `GET /api/health` — health check
- `GET /api/uploads` — list collections, including the bundled sample
- `POST /api/uploads` — register a trusted DataFrame pickle under multipart field `file`
- `GET /api/uploads/<id>/systems` — list trajectories in a collection
- `GET /api/uploads/<id>/systems/<slug>/trajectory` — retrieve one trajectory
- `GET /api/ipd/capability` — report IPD computation support
- `GET /api/delta-mtp/capability` — report Delta-MTP computation support

## Development checks

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import ast
from pathlib import Path
for path in [Path("app.py"), *Path("backend").rglob("*.py")]:
    ast.parse(path.read_text())
print("Python syntax OK")
PY

node --input-type=module --check < frontend/viewer.js
node --input-type=module --check < frontend/plots.js
```

This project currently has no maintained automated test suite or checked-in dependency lockfile.
