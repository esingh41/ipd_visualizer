"""Orchestration around :mod:`backend.services.radius_thole`.

Owns request validation, row selection, persistence and error translation. It owns none
of the science and none of the result-column naming -- both of those live in
``radius_thole``, and duplicating either here is how the two would drift apart.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

from backend.services.errors import IpdError

# The UI offers four mutually exclusive damping modes. This mapping is the *only* place
# the wire value becomes a Python value, and the reason the wire vocabulary is lowercase
# "none" rather than "None": radius_thole treats a non-empty string as truthy, so a
# literal "None" reaching the science would leave radius-based damping switched on while
# the results were filed under the undamped column name. Here that is a 400 instead.
RADIUS_FROM_API: Dict[str, Optional[str]] = {
    "none": None,
    "intra": "intra",
    "inter": "inter",
    "both": "both",
}

# Inverse, for labelling a stored result in a response.
RADIUS_TO_API: Dict[Optional[str], str] = {v: k for k, v in RADIUS_FROM_API.items()}


def parse_radius(value: Any) -> Optional[str]:
    """Map a request's radius string to the value ``radius_thole`` expects."""
    if not isinstance(value, str) or value.lower() not in RADIUS_FROM_API:
        raise IpdError(
            "unsupported_radius",
            f"radius must be one of {sorted(RADIUS_FROM_API)}; got {value!r}",
            details={"allowed": sorted(RADIUS_FROM_API), "received": value},
        )
    return RADIUS_FROM_API[value.lower()]


# --- capability -------------------------------------------------------------

# Keyword arguments radius_thole.compute_ipd_row passes into
# induced_dipole_induction_optimized_no_correction. Checked by name rather than by a
# version string because apnet_pt self-reports "0.0.1" on every commit, while the
# function's signature has genuinely changed between checkouts -- an older one has no
# alpha_*_external, no per-channel damping parameters and no dipole history at all, which
# is exactly the incompatibility a version pin would fail to catch.
REQUIRED_IPD_KWARGS = (
    "alpha_A_external",
    "alpha_B_external",
    "max_iterations",
    "thole_damping_param_direct",
    "thole_damping_param_mutual_AB",
    "thole_damping_param_AA",
    "thole_damping_param_BB",
    "direct_damping_style",
    "mutual_AB_damping_style",
    "AA_damping_style",
    "BB_damping_style",
    "return_components",
    "return_induced_dipoles",
    "return_lambdas",
    "return_induced_dipole_history",
    "include_H",
)

_CAPABILITY: Optional[Dict[str, Any]] = None


def _probe_capability() -> Dict[str, Any]:
    """Import apnet_pt and confirm it exposes the API ``radius_thole`` depends on."""
    try:
        import apnet_pt
        from apnet_pt.AtomPairwiseModels.mtp_mtp import (
            induced_dipole_induction_optimized_no_correction as ipd_fn,
        )
        from apnet_pt.pt_datasets.ap2_fused_ds import (  # noqa: F401
            qcel_dimer_to_fused_data,
            ap2_fused_collate_update_no_target,
        )
    except ImportError as exc:
        return {
            "available": False,
            "code": "apnet_pt_missing",
            "reason": (
                "apnet_pt is not installed in the server's environment. "
                f"Install it (pip install -e ~/gits/qcmlforge). Original error: {exc}"
            ),
            "version": None,
            "path": None,
            "missing_kwargs": [],
        }
    except Exception as exc:
        # Installed but not importable -- an incompatible transitive dependency rather
        # than a missing package. Observed for real: apnet_pt under a Python 3.14 env
        # whose pydantic/qcelemental pairing raises RuntimeError partway through import.
        # Catching only ImportError here would turn that into a 500 on a capability
        # *query*, which is the one endpoint that must always be able to answer.
        return {
            "available": False,
            "code": "apnet_pt_broken",
            "reason": (
                "apnet_pt is installed but failed to import, which usually means a "
                "conflicting dependency in this environment rather than a missing "
                f"package. {type(exc).__name__}: {exc}"
            ),
            "version": None,
            "path": None,
            "missing_kwargs": [],
        }

    version = getattr(apnet_pt, "__version__", None)
    path = getattr(apnet_pt, "__file__", None)

    parameters = inspect.signature(ipd_fn).parameters
    missing = [name for name in REQUIRED_IPD_KWARGS if name not in parameters]
    if missing:
        return {
            "available": False,
            "code": "apnet_pt_incompatible",
            "reason": (
                "The installed apnet_pt exposes an incompatible "
                "induced_dipole_induction_optimized_no_correction: it is missing "
                f"{len(missing)} argument(s) radius_thole.py requires."
            ),
            "version": version,
            "path": path,
            "missing_kwargs": missing,
        }

    return {
        "available": True,
        "code": None,
        "reason": None,
        "version": version,
        "path": path,
        "missing_kwargs": [],
    }


def capability(refresh: bool = False) -> Dict[str, Any]:
    """Whether this server can run an IPD calculation, and why not when it cannot.

    Cached after the first call: the answer cannot change without restarting the process,
    and the probe costs a full ``import torch``. The frontend calls this on page load, so
    that multi-second import is paid before the user presses anything rather than on top
    of their first calculation.
    """
    global _CAPABILITY
    if _CAPABILITY is None or refresh:
        _CAPABILITY = _probe_capability()
    return dict(_CAPABILITY)


def require_capability() -> None:
    """Raise the matching :class:`IpdError` when IPD computation is unavailable."""
    cap = capability()
    if cap["available"]:
        return
    raise IpdError(
        cap["code"],
        cap["reason"],
        status=503,
        details={
            "missing_kwargs": cap["missing_kwargs"],
            "apnet_pt_path": cap["path"],
            "apnet_pt_version": cap["version"],
        },
        retryable=False,
    )
