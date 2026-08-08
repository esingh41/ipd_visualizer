"""Retrieval for the trajectory tab: read the processed JSON, and nothing more.

Every expensive or dataframe-specific step -- grouping, ordering, contact distances,
equilibrium ratios -- already happened during upload. This module opens the JSON that came
out of it and hands pieces to the routes. It imports no pandas, no qcelemental and none of
``system_processing``; if it ever needs to, something has been serialized too thinly.

Systems are addressed by **slug**, not by name. Trajectory names come from user data and can
contain "/", "+" and non-ASCII characters, none of which survive a URL path segment. The
manifest already stores a slug per system, so that is the identifier the API uses, and the
display name rides along in the payload.
"""

from __future__ import annotations

import json

from backend.services import system_serialization


class NotFound(LookupError):
    """A collection or system that does not exist, for the routes to answer 404 with."""


def _slug_of(entry):
    """The slug in a manifest system entry, recovered from its stored file path."""
    return entry["file"].rsplit("/", 1)[-1].removesuffix(".json")


def list_collections():
    """Everything stored, as the collection selector needs it."""
    return [
        {
            "collection_id": manifest["collection_id"],
            "display_name": manifest["display_name"],
            "system_count": manifest["system_count"],
            "frame_count": manifest["frame_count"],
            "created_at": manifest["created_at"],
            "features": manifest["features"],
        }
        for manifest in system_serialization.list_collections()
    ]


def _manifest(upload_id):
    try:
        return system_serialization.load_manifest(upload_id)
    except (OSError, ValueError, json.JSONDecodeError):
        raise NotFound(f"No such upload: {upload_id!r}")


def list_systems(upload_id):
    """Lightweight metadata for the system dropdown -- no geometry, no frames."""
    manifest = _manifest(upload_id)
    return {
        "upload_id": manifest["collection_id"],
        "display_name": manifest["display_name"],
        "features": manifest["features"],
        "systems": [
            {
                "system_id": entry["name"],
                "slug": _slug_of(entry),
                "frame_count": entry["n_frames"],
            }
            for entry in manifest["systems"]
        ],
    }


def get_trajectory(upload_id, slug):
    """One complete trajectory: every frame, already ordered, in a single response.

    Whole-trajectory rather than frame-at-a-time on purpose. A dimer scan is a handful of
    frames, and holding them in the browser makes slider movement local instead of a
    round trip per step.
    """
    manifest = _manifest(upload_id)
    for entry in manifest["systems"]:
        if _slug_of(entry) == slug:
            path = system_serialization.collection_dir(upload_id) / entry["file"]
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
    raise NotFound(f"No system {slug!r} in upload {upload_id!r}")
