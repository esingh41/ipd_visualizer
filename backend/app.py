"""HTTP for the upload pipeline and the trajectory tab.

Routes do HTTP and nothing else: pull the request apart, call one service, answer. None of
them knows how a collection is stored, and none reopens an uploaded dataframe -- the
trajectory tab is served entirely from the JSON written at upload time.

``app_old.py`` is the previous application, kept for reference. Its routes are gone.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from backend.services import capability, ipd_service, trajectory_service, upload_system
from backend.services.errors import IpdError

# Dimer dataframes with per-row molecules and multipoles run to a few hundred MB at most.
# A cap keeps a mistyped upload from being read entirely into memory before it is rejected.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024


def _uploaded_bytes():
    """The uploaded pickle, or an UploadError explaining what was wrong with the request."""
    if "file" not in request.files:
        raise upload_system.UploadError(
            upload_system.failure_report(
                "unreadable_dataset",
                "No file uploaded. Provide the dataframe under form field 'file'.",
            )
        )
    uploaded = request.files["file"]
    filename = uploaded.filename or ""
    if not filename.lower().endswith((".pkl", ".pickle")):
        raise upload_system.UploadError(
            upload_system.failure_report(
                "unreadable_dataset",
                "The dataset must be a pandas pickle (.pkl or .pickle).",
            )
        )
    raw = uploaded.read()
    if not raw:
        raise upload_system.UploadError(
            upload_system.failure_report("unreadable_dataset", "The uploaded file is empty.")
        )
    return raw, filename


def create_app() -> Flask:
    frontend_dir = Path(__file__).resolve().parents[1] / "frontend"

    app = Flask(__name__, static_folder=str(frontend_dir), static_url_path="")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    @app.errorhandler(upload_system.UploadError)
    def _upload_error(exc):
        # One error shape whichever step failed, so the page renders unreadable files and
        # missing columns the same way.
        return jsonify(exc.report), 400

    @app.errorhandler(trajectory_service.NotFound)
    def _not_found(exc):
        return jsonify({"error": str(exc)}), 404

    @app.errorhandler(IpdError)
    def _ipd_error(exc):
        # Carries its own status: 400 for a bad mode, 404 for nothing computed, 503 when
        # apnet_pt cannot run here. The body keeps the "error" key every other failure uses,
        # so one client-side handler reads them all.
        return jsonify(exc.to_dict()), exc.status

    @app.errorhandler(ValueError)
    def _bad_request(exc):
        # collection_dir raises this for an id that would escape data/systems.
        return jsonify({"error": str(exc)}), 400

    @app.get("/")
    def index():
        return send_from_directory(frontend_dir, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/uploads")
    def register_upload():
        raw, filename = _uploaded_bytes()
        return jsonify(upload_system.process_upload(raw, filename))

    @app.get("/api/uploads")
    def list_uploads():
        return jsonify({"uploads": trajectory_service.list_collections()})

    @app.get("/api/uploads/<upload_id>/systems")
    def list_systems(upload_id):
        return jsonify(trajectory_service.list_systems(upload_id))

    @app.get("/api/uploads/<upload_id>/systems/<slug>/trajectory")
    def get_trajectory(upload_id, slug):
        return jsonify(trajectory_service.get_trajectory(upload_id, slug))

    # --- IPD ----------------------------------------------------------------

    @app.get("/api/ipd/capability")
    def ipd_capability():
        # Answers whether the *server* can compute. Whether a given frame has the inputs is a
        # separate flag, and rides in that frame's "ipd" block.
        return jsonify(capability.capability())

    @app.get("/api/uploads/<upload_id>/systems/<slug>/frames/<int:frame_index>/ipd")
    def get_frame_ipd(upload_id, slug, frame_index):
        # Its own request rather than part of the trajectory payload: one history for a run
        # that hit the SCF iteration cap is ~150 KB, and there is one per frame per mode.
        return jsonify(
            ipd_service.get_history(upload_id, slug, frame_index, _requested_mode())
        )

    @app.post("/api/uploads/<upload_id>/systems/<slug>/frames/<int:frame_index>/ipd")
    def compute_frame_ipd(upload_id, slug, frame_index):
        return jsonify(
            ipd_service.compute(upload_id, slug, frame_index, _requested_mode())
        )

    @app.post("/api/uploads/<upload_id>/systems/<slug>/ipd")
    def compute_system_ipd(upload_id, slug):
        return jsonify(ipd_service.compute_system(upload_id, slug, _requested_mode()))

    return app


def _requested_mode():
    """The damping mode from ``?mode=`` or a JSON body, whichever the request used.

    GET carries it in the query string and POST in the body; the value means the same thing
    either way, so it is read in one place. Validating it is ``ipd_results``' job -- this only
    has to find it.
    """
    mode = request.args.get("mode")
    if mode is None and request.is_json:
        payload = request.get_json(silent=True) or {}
        mode = payload.get("mode")
    if not mode:
        raise IpdError(
            "missing_ipd_mode",
            "No IPD damping mode given. Pass ?mode=<id> or a JSON body {\"mode\": \"<id>\"}.",
        )
    return mode
