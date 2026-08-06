from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import HTTPException

from backend.services import dataset_store, dipole_data, ipd_compute, trajectory
from backend.services.errors import IpdError

# Dimer dataframes with per-row molecules and multipoles run to a few hundred MB at most.
# A cap keeps a mistyped upload from being read entirely into memory before it is rejected.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024


def _uploaded_dataframe_bytes() -> tuple[bytes, str]:
    """Pull the uploaded pickle off the request, or raise a 400 explaining what was wrong."""
    if "file" not in request.files:
        raise IpdError(
            "unreadable_dataset",
            "No file uploaded. Provide the dataframe under form field 'file'.",
        )
    uploaded = request.files["file"]
    filename = uploaded.filename or ""
    if not filename.lower().endswith((".pkl", ".pickle")):
        raise IpdError(
            "unreadable_dataset",
            "The dataset must be a pandas pickle (.pkl or .pickle).",
            details={"filename": filename},
        )
    raw = uploaded.read()
    if not raw:
        raise IpdError("unreadable_dataset", "The uploaded file is empty.")
    return raw, filename


def create_app() -> Flask:
    root_dir = Path(__file__).resolve().parents[1]
    frontend_dir = root_dir / "frontend"

    app = Flask(__name__, static_folder=str(frontend_dir), static_url_path="")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    @app.get("/api/health")
    def health() -> Any:
        return jsonify({"status": "ok"})

    @app.get("/api/ipd/systems")
    def ipd_systems() -> Any:
        # Metadata only -- no coordinates, no dipole histories. This is what builds the
        # selectors, so it must stay cheap however many systems are bundled.
        #
        # The bundled .npz catalog plus one entry per computed geometry, so an uploaded
        # dataset's results appear in the same Dataset -> System -> Separation -> Model
        # cascade. The bundled half stays lru_cached; the computed half is rebuilt per
        # request (behind its own file-stamp cache) so a new result shows up immediately.
        return jsonify(dipole_data.list_systems() + ipd_compute.catalog_entries())

    @app.get("/api/ipd/system")
    def ipd_system() -> Any:
        # A query parameter rather than a path segment: system ids contain dots, and the
        # dataset-scoped ones contain a slash.
        system_id = request.args.get("system_id", "").strip()
        if not system_id:
            return jsonify({"error": "Missing required query parameter 'system_id'"}), 400

        model = request.args.get("model") or None
        scoped = ipd_compute.parse_dataset_system_id(system_id)
        if scoped is not None:
            dataset_id, row_id = scoped
            return jsonify(ipd_compute.load_stored(dataset_id, row_id, model))
        return jsonify(dipole_data.get_system(system_id, model))

    @app.get("/api/ipd/trajectory")
    def ipd_trajectory() -> Any:
        # Query parameters for the same reason /api/ipd/system uses them: a system_name
        # carries characters a path segment would mangle -- "Na+-Water", "Cl⁻-Water".
        dataset = request.args.get("dataset", "").strip()
        system = request.args.get("system", "").strip()
        if not dataset or not system:
            return (
                jsonify(
                    {"error": "Missing required query parameters 'dataset' and 'system'"}
                ),
                400,
            )
        return jsonify(
            trajectory.build_trajectory(dataset, system, request.args.get("model") or None)
        )

    @app.get("/api/ipd/capability")
    def ipd_capability() -> Any:
        # Always 200, even when IPD is unavailable: "can this server compute?" is a
        # question with a legitimate negative answer, not a failed request. The page
        # calls this on load, which also pays the multi-second torch import up front.
        return jsonify(ipd_compute.capability())

    # --- on-demand IPD datasets ---------------------------------------------

    @app.post("/api/datasets")
    def register_dataset() -> Any:
        raw, filename = _uploaded_dataframe_bytes()
        return jsonify(ipd_compute.register(raw, filename))

    @app.get("/api/datasets")
    def list_datasets() -> Any:
        return jsonify(dataset_store.list_datasets())

    @app.get("/api/datasets/<dataset_id>")
    def get_dataset(dataset_id: str) -> Any:
        # Systems, geometries and stored-result flags all ride along here rather than in
        # endpoints of their own: the whole thing is a few hundred rows even for S66x8,
        # and one response means the sidebar can never show a half-updated view.
        return jsonify(ipd_compute.summary(dataset_id))

    @app.post("/api/datasets/<dataset_id>/ipd")
    def run_ipd(dataset_id: str) -> Any:
        payload = request.get_json(silent=True) or {}
        row_id = str(payload.get("row_id", "")).strip()
        if not row_id:
            raise IpdError("row_not_found", "Request must include 'row_id'.", status=400)
        return jsonify(
            ipd_compute.run(
                dataset_id,
                row_id,
                payload.get("radius", ""),
                force_recompute=bool(payload.get("force_recompute", False)),
            )
        )

    @app.post("/api/datasets/<dataset_id>/ipd/group")
    def run_ipd_group(dataset_id: str) -> Any:
        payload = request.get_json(silent=True) or {}
        system = str(payload.get("system", "")).strip()
        if not system:
            raise IpdError("system_not_found", "Request must include 'system'.", status=400)
        return jsonify(
            ipd_compute.run_group(
                dataset_id,
                system,
                payload.get("radius", ""),
                force_recompute=bool(payload.get("force_recompute", False)),
            )
        )

    @app.post("/api/datasets/<dataset_id>/restore")
    def restore_dataset(dataset_id: str) -> Any:
        # ipd_compute owns the lock here: restoring is not a plain file copy, it has to
        # re-apply the canonical index and the derived columns the working copy carries.
        return jsonify(ipd_compute.restore(dataset_id))

    @app.get("/api/datasets/<dataset_id>/export")
    def export_dataset(dataset_id: str) -> Any:
        meta = dataset_store.read_meta(dataset_id)
        # Under the lock so an export can never read a file mid-replacement.
        with dataset_store.dataset_lock(dataset_id):
            path = dataset_store.working_path(dataset_id)
            name = Path(meta.get("name") or "dataset.pkl").stem
            return send_file(
                path,
                as_attachment=True,
                download_name=f"{name}_with_ipd.pkl",
                mimetype="application/octet-stream",
            )

    @app.errorhandler(IpdError)
    def handle_ipd_error(err: IpdError) -> Any:
        return jsonify(err.to_dict()), err.status

    @app.errorhandler(dataset_store.DatasetNotFound)
    def handle_dataset_not_found(err: dataset_store.DatasetNotFound) -> Any:
        return jsonify({"error": str(err), "code": "dataset_not_found"}), 404

    @app.errorhandler(dipole_data.SystemNotFound)
    def handle_system_not_found(err: dipole_data.SystemNotFound) -> Any:
        return jsonify({"error": str(err)}), 404

    @app.errorhandler(ValueError)
    def handle_value_error(err: ValueError) -> Any:
        return jsonify({"error": str(err)}), 400

    @app.errorhandler(RuntimeError)
    def handle_runtime_error(err: RuntimeError) -> Any:
        return jsonify({"error": str(err)}), 500

    @app.errorhandler(Exception)
    def handle_unexpected_error(err: Exception) -> Any:
        if isinstance(err, HTTPException):
            return err
        return jsonify({"error": f"Unexpected server error: {err}"}), 500

    @app.get("/")
    def serve_index() -> Any:
        return send_from_directory(frontend_dir, "index.html")

    return app
