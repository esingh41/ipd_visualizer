"""Can this server run its optional IPD and Delta-MTP kernels, and if not, why not.

Each server capability is answered and cached independently. Nothing here opens a dataframe --
whether a *frame* carries a feature's inputs is a different question, answered by serialized
frame metadata. Dependency compatibility and row computability fail independently and deserve
different messages.

The probe checks the SCF kernel's **argument names**, not a version string. apnet_pt
self-reports ``0.0.1`` on every commit, while
``induced_dipole_induction_optimized_no_correction``'s signature has genuinely changed between
checkouts -- an older one has no ``alpha_*_external``, no per-channel damping parameters and no
dipole history at all. That is exactly the incompatibility a version pin would fail to catch.
"""

from __future__ import annotations

import inspect

from backend.services.errors import DeltaMtpError, IpdError

# The keyword arguments thole_damping.compute_ipd_row passes into
# induced_dipole_induction_optimized_no_correction. Only the discriminating ones: the data
# arguments (ZA, RA, qA, ...) have been there since the beginning, so their presence proves
# nothing about which build is installed.
#
# thole_damping_param and debug are here because thole_damping passes them. A probe that
# omitted them could green-light a build that then raises TypeError at call time -- which is
# the one thing this module exists to prevent.
REQUIRED_IPD_KWARGS = (
    "alpha_A_external",
    "alpha_B_external",
    "max_iterations",
    "thole_damping_param",
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
    "debug",
    "include_H",
)

_CAPABILITY = None
_DELTA_MTP_CAPABILITY = None

REQUIRED_DELTA_MTP_ARGS = ("mols", "dimer", "monA", "monB")


def missing_ipd_kwargs(ipd_fn):
    """Which of the arguments ``thole_damping`` passes this function does not accept.

    Separated from the probe so it can be checked against a stand-in signature without
    importing apnet_pt -- which pulls in torch, and is far too heavy a thing to drag into a
    unit test.
    """
    parameters = inspect.signature(ipd_fn).parameters
    return [name for name in REQUIRED_IPD_KWARGS if name not in parameters]


def _probe_capability():
    """Import apnet_pt and confirm it exposes the API ``thole_damping`` depends on."""
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
        # Installed but not importable -- an incompatible transitive dependency rather than a
        # missing package. Observed for real: apnet_pt under a Python 3.14 env whose
        # pydantic/qcelemental pairing raises RuntimeError partway through import. Catching
        # only ImportError here would turn that into a 500 on a capability *query*, which is
        # the one endpoint that must always be able to answer.
        return {
            "available": False,
            "code": "apnet_pt_broken",
            "reason": (
                "apnet_pt is installed but failed to import, which usually means a "
                "conflicting dependency in this environment rather than a missing package. "
                f"{type(exc).__name__}: {exc}"
            ),
            "version": None,
            "path": None,
            "missing_kwargs": [],
        }

    version = getattr(apnet_pt, "__version__", None)
    path = getattr(apnet_pt, "__file__", None)

    missing = missing_ipd_kwargs(ipd_fn)
    if missing:
        return {
            "available": False,
            "code": "apnet_pt_incompatible",
            "reason": (
                "The installed apnet_pt exposes an incompatible "
                "induced_dipole_induction_optimized_no_correction: it is missing "
                f"{len(missing)} argument(s) thole_damping.py requires."
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


def capability(refresh=False):
    """Whether this server can run an IPD calculation, and why not when it cannot.

    Cached after the first call: the answer cannot change without restarting the process, and
    the probe costs a full ``import torch``. The frontend calls this on page load, so that
    multi-second import is paid before the user presses anything rather than on top of their
    first calculation.
    """
    global _CAPABILITY
    if _CAPABILITY is None or refresh:
        _CAPABILITY = _probe_capability()
    return dict(_CAPABILITY)


def require_capability():
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


def _probe_delta_mtp_capability():
    """Check only the multipole kernel Delta-MTP uses, independent of IPD's SCF API."""
    try:
        import apnet_pt
        from apnet_pt.multipole import multipoles_elst_ind_dimer
    except ImportError as exc:
        return {
            "available": False,
            "code": "apnet_pt_missing",
            "reason": (
                "apnet_pt is not installed in the server's environment, so Delta-MTP "
                f"cannot run. Original error: {exc}"
            ),
            "version": None,
            "path": None,
            "missing_args": [],
        }
    except Exception as exc:
        return {
            "available": False,
            "code": "apnet_pt_broken",
            "reason": (
                "apnet_pt is installed but its Delta-MTP multipole kernel failed to import. "
                f"{type(exc).__name__}: {exc}"
            ),
            "version": None,
            "path": None,
            "missing_args": [],
        }

    parameters = inspect.signature(multipoles_elst_ind_dimer).parameters
    missing = [name for name in REQUIRED_DELTA_MTP_ARGS if name not in parameters]
    version = getattr(apnet_pt, "__version__", None)
    path = getattr(apnet_pt, "__file__", None)
    if missing:
        return {
            "available": False,
            "code": "apnet_pt_delta_mtp_incompatible",
            "reason": (
                "The installed apnet_pt exposes an incompatible "
                "multipoles_elst_ind_dimer kernel; required argument(s) are missing."
            ),
            "version": version,
            "path": path,
            "missing_args": missing,
        }
    return {
        "available": True,
        "code": None,
        "reason": None,
        "version": version,
        "path": path,
        "missing_args": [],
    }


def delta_mtp_capability(refresh=False):
    """Whether this server can run the Delta-MTP multipole kernel."""
    global _DELTA_MTP_CAPABILITY
    if _DELTA_MTP_CAPABILITY is None or refresh:
        _DELTA_MTP_CAPABILITY = _probe_delta_mtp_capability()
    return dict(_DELTA_MTP_CAPABILITY)


def require_delta_mtp_capability():
    """Raise a structured 503 when the Delta-MTP kernel is unavailable."""
    cap = delta_mtp_capability()
    if cap["available"]:
        return
    raise DeltaMtpError(
        cap["code"],
        cap["reason"],
        status=503,
        details={
            "missing_args": cap["missing_args"],
            "apnet_pt_path": cap["path"],
            "apnet_pt_version": cap["version"],
        },
        retryable=False,
    )
