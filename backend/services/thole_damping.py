"""Thole-damped induced-point-dipole induction, and the naming its results are stored under.

**Every scheme here is Thole damping.** The configurable choice is not *whether* to damp but
**how the Thole parameter ``a`` is obtained**, and that choice is made independently for the
two kinds of edge:

``inter``
    intermolecular AB edges -- every atom of monomer A against every atom of monomer B.

``intra``
    intramolecular AA and BB edges -- atoms within one monomer, against each other.

Each side takes one of two parameterizations:

``fixed_0.39``
    The universal Thole value ``a = 0.39`` on every selected edge. Depends on no radius data
    at all.

``ts_mbis_radii``
    An edge-specific ``a`` derived from HFVR-scaled Tkatchenko-Scheffler vdW radii::

        R_i   = R_i,free^TS * HFVR_i^(1/3)      per atom, Angstrom
        a_M,ij = sqrt(R_i^2 + R_j^2)            per edge, Angstrom -- a *length*
        a_ij   = sqrt(alpha_i alpha_j) / a_M,ij^3   per edge, dimensionless

    Note the two different meanings of "a": ``a_M`` is a damping *length* in Angstrom, and
    ``a_ij`` is the dimensionless parameter apnet_pt actually consumes. The conversion is
    :func:`_thole_parameter_from_damping_length` and happens in exactly one place.

That gives four combinations, all of them Thole damping::

    inter=fixed_0.39     intra=fixed_0.39      universal a everywhere
    inter=ts_mbis_radii  intra=fixed_0.39      radius-derived on AB only
    inter=fixed_0.39     intra=ts_mbis_radii   radius-derived within each monomer only
    inter=ts_mbis_radii  intra=ts_mbis_radii   radius-derived everywhere

Kept deliberately separate from *damping style* (``inter_damping_style`` /
``intra_damping_style``), which selects the mathematical form -- ``"mutual"``, the
``1 - exp(-a u^3)`` default, or ``"linear"``, Thole's original piecewise form. Where ``a``
comes from and what is done with it are different questions.

Usage::

    # Universal Thole a = 0.39 everywhere
    compute_ipd_row(row,
                    inter_thole_parameterization="fixed_0.39",
                    intra_thole_parameterization="fixed_0.39")

    # Radius-derived Thole parameter on AB only
    compute_ipd_row(row,
                    inter_thole_parameterization="ts_mbis_radii",
                    intra_thole_parameterization="fixed_0.39")

    # Radius-derived Thole parameter on AA/BB only
    compute_ipd_row(row,
                    inter_thole_parameterization="fixed_0.39",
                    intra_thole_parameterization="ts_mbis_radii")

    # Radius-derived Thole parameter everywhere
    compute_ipd_row(row,
                    inter_thole_parameterization="ts_mbis_radii",
                    intra_thole_parameterization="ts_mbis_radii")

``torch`` and ``apnet_pt`` are imported lazily inside the computing functions, never at module
scope, so the naming and dataframe-inspection half of this API stays usable on a machine that
cannot run the science at all. That is what lets an app report "apnet_pt unavailable" instead
of failing to start.
"""

import functools
from importlib import resources

import numpy as np
import pandas as pd
import qcelemental as qcel

_ANG_PER_BOHR = qcel.constants.conversion_factor("bohr", "angstrom")

# The two ways an edge can obtain its Thole parameter. Both are Thole damping; neither is a
# different damping model.
THOLE_PARAMETERIZATIONS = (
    "fixed_0.39",
    "ts_mbis_radii",
)

# The universal Thole value.
DEFAULT_THOLE_PARAMETER = 0.39

# apnet_pt's own default. Passed explicitly so the convergence test in compute_ipd_row --
# "did the SCF run all the way to the cap?" -- cannot drift away from the real limit.
MAX_SCF_ITERATIONS = 200

# Every column an IPD run writes, as a role -> the prefix substituted for "IPD (" in the base
# column name. The base name itself is the "energy" role.
_DERIVED_PREFIXES = {
    "energy_qu": "IPD qu (",
    "energy_uu": "IPD uu (",
    "a_ang": "a ang (",
    "mu_ind_A": "mu_ind A (",
    "mu_ind_B": "mu_ind B (",
    "mu_hist_A": "mu_ind hist A (",
    "mu_hist_B": "mu_ind hist B (",
    "lam3_AB": "lam3 AB (",
    "lam5_AB": "lam5 AB (",
    "lam3_AA": "lam3 AA (",
    "lam5_AA": "lam5 AA (",
    "lam3_BB": "lam3 BB (",
    "lam5_BB": "lam5 BB (",
}

# Roles holding one number per row; everything else holds a NumPy array and therefore needs an
# object-dtype column.
_SCALAR_ROLES = frozenset({"energy", "energy_qu", "energy_uu", "a_ang"})

# What must be present for a stored result to be usable by a visualizer. The lambdas and
# component energies are diagnostics -- a row missing those is still renderable.
_ESSENTIAL_ROLES = ("energy", "mu_hist_A", "mu_hist_B")


def _validate_thole_parameterization(value, *, name):
    if value not in THOLE_PARAMETERIZATIONS:
        raise ValueError(
            f"{name} must be one of {THOLE_PARAMETERIZATIONS!r}; got {value!r}"
        )


# --- column naming ----------------------------------------------------------


def ipd_column_names(
    *,
    inter_thole_parameterization="fixed_0.39",
    intra_thole_parameterization="fixed_0.39",
    include_H=True,
    suffix="",
):
    """Every column an IPD run with these choices writes, keyed by role.

    The only place these names are constructed. The base name states both choices outright::

        IPD (inter=ts_mbis_radii, intra=fixed_0.39)

    so a stored column says what produced it without a lookup table. The other thirteen are
    derived from it by substitution, which is what keeps one combination from overwriting
    another's results.

    ``include_H=False`` tags its own columns. Hydrogen keeps its permanent multipoles but gets
    no induced dipole, so those results are not comparable with an ``include_H=True`` run and
    must not land in the same column.
    """
    _validate_thole_parameterization(
        inter_thole_parameterization, name="inter_thole_parameterization"
    )
    _validate_thole_parameterization(
        intra_thole_parameterization, name="intra_thole_parameterization"
    )

    col = (
        f"IPD (inter={inter_thole_parameterization}, "
        f"intra={intra_thole_parameterization})"
    )

    tags = []
    if not include_H:
        tags.append("noH")
    if suffix:
        tags.append(suffix)
    if tags:
        # Every base name ends in ")", so the tags splice into the existing group.
        col = f"{col[:-1]}, {', '.join(tags)})"

    names = {"energy": col}
    for role, prefix in _DERIVED_PREFIXES.items():
        names[role] = col.replace("IPD (", prefix)
    return names


# --- dataframe read/write ---------------------------------------------------


def _is_missing(value):
    """True when a dataframe cell holds no usable result.

    ``pd.isna`` returns an *array* for an array cell, which is not a truth value, so arrays are
    tested on their size instead.
    """
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def has_ipd_result(
    df,
    idx,
    *,
    inter_thole_parameterization="fixed_0.39",
    intra_thole_parameterization="fixed_0.39",
    include_H=True,
    suffix="",
):
    """True when row ``idx`` already carries a usable result for these choices."""
    names = ipd_column_names(
        inter_thole_parameterization=inter_thole_parameterization,
        intra_thole_parameterization=intra_thole_parameterization,
        include_H=include_H,
        suffix=suffix,
    )
    if any(names[role] not in df.columns for role in _ESSENTIAL_ROLES):
        return False
    return not any(_is_missing(df.at[idx, names[role]]) for role in _ESSENTIAL_ROLES)


def read_ipd_row(
    df,
    idx,
    *,
    inter_thole_parameterization="fixed_0.39",
    intra_thole_parameterization="fixed_0.39",
    include_H=True,
    suffix="",
):
    """Read a stored result back out, in the same shape :func:`compute_ipd_row` returns.

    Raises KeyError when no result is stored, so callers must gate on :func:`has_ipd_result`.
    """
    choices = {
        "inter_thole_parameterization": inter_thole_parameterization,
        "intra_thole_parameterization": intra_thole_parameterization,
        "include_H": include_H,
        "suffix": suffix,
    }
    if not has_ipd_result(df, idx, **choices):
        raise KeyError(
            f"No stored IPD result at row {idx!r} for "
            f"inter={inter_thole_parameterization!r}, "
            f"intra={intra_thole_parameterization!r}"
        )

    names = ipd_column_names(**choices)
    result = {}
    for role, col in names.items():
        value = df.at[idx, col] if col in df.columns else None
        result[role] = None if _is_missing(value) else value

    mu_hist_A = np.asarray(result["mu_hist_A"])
    result["iterations"] = int(mu_hist_A.shape[0]) - 1
    result["converged"] = result["iterations"] < MAX_SCF_ITERATIONS
    return result


def write_ipd_row(
    df,
    idx,
    result,
    *,
    inter_thole_parameterization="fixed_0.39",
    intra_thole_parameterization="fixed_0.39",
    include_H=True,
    suffix="",
):
    """Write one row's result into ``df`` at label ``idx``, in place.

    Missing columns are created first -- ``df.at[]`` cannot create a column, and an array value
    needs an object-dtype one to land in rather than being broadcast.
    """
    names = ipd_column_names(
        inter_thole_parameterization=inter_thole_parameterization,
        intra_thole_parameterization=intra_thole_parameterization,
        include_H=include_H,
        suffix=suffix,
    )
    for role, col in names.items():
        if col not in df.columns:
            if role in _SCALAR_ROLES:
                df[col] = np.nan
            else:
                df[col] = pd.Series([None] * len(df), index=df.index, dtype=object)
        df.at[idx, col] = result[role]


# --- TS/MBIS radii ----------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _read_full_polarizability_table():
    """Untruncated ``vdw-params.csv`` (includes ionic rows like "Na+", "Cl-"), indexed by the
    ``symbol`` column.

    apnet_pt's own ``polarizability_table`` (constants.py) truncates this same file to
    ``nrows=102`` -- neutral elements only, Z-indexable. We read it without that limit to reach
    the ion rows, which only have unique ``symbol`` strings, not unique Z.
    """
    path = resources.files("apnet_pt").joinpath("data", "vdw-params.csv")
    return pd.read_csv(path, header=0, index_col=0, sep=",")


def _ts_mbis_radius_ang(Z, volume_ratios):
    """Per-atom damping radius (Angstrom): ``R_vdw(TS) * HFVR^(1/3)``.

    More generalizable than a fixed ion/vdW-radius table, and its appeal is that *everything*
    comes from the one CSV: the ionic rows there have ``R_vdw(TS) = NaN``, but Psi4's MBIS
    volume ratio for a monatomic-ion fragment is referenced to the neutral free atom, so the
    ratio already carries the neutral->ion contraction.
    """
    tab = _read_full_polarizability_table()
    Z = np.asarray(Z).astype(int).ravel()
    ratios = np.asarray(volume_ratios, dtype=float).ravel()

    # Neutral rows only: "Na" and "Na+" are distinct index strings, so looking up the bare
    # element symbol never hits an ionic row (which are all NaN here).
    r_free = np.array(
        [float(tab.loc[qcel.periodictable.to_E(int(z)), "R_vdw(TS)"]) for z in Z]
    )
    out = r_free * ratios ** (1.0 / 3.0) * _ANG_PER_BOHR
    if not np.all(np.isfinite(out)):
        raise ValueError(
            f"non-finite TS/MBIS radius for Z={Z.tolist()} ratios={ratios.tolist()}; "
            "R_vdw(TS) is NaN for ionic rows and some pickles store NaN monomer volume "
            "ratios for bare monatomic ions"
        )
    return out


# --- Thole parameter construction -------------------------------------------


def _edge_damping_length_ang(radius_i, radius_j):
    """The TS radius combination rule: ``a_M,ij = sqrt(R_i^2 + R_j^2)``, in Angstrom.

    One definition, because the AB length is also reported as the ``a_ang`` diagnostic and two
    spellings of this would eventually disagree.
    """
    return np.hypot(radius_i, radius_j)


def _thole_parameter_from_damping_length(a_length_ang, alpha_i, alpha_j):
    """Convert a Thole damping *length* (Angstrom) into the dimensionless per-edge Thole
    parameter apnet_pt expects::

        a_ij = sqrt(alpha_i alpha_j) / a_M,ij^3

    Polarizabilities are in bohr^3, so the length converts to bohr first.
    """
    import torch

    a_M_bohr = (
        torch.as_tensor(a_length_ang, dtype=alpha_i.dtype, device=alpha_i.device)
        / _ANG_PER_BOHR
    )
    return torch.sqrt(alpha_i * alpha_j) / (a_M_bohr**3)


def _build_thole_parameters(parameterization, radius_i, radius_j, alpha_i, alpha_j):
    """The Thole parameter for a set of edges -- the one place that answers *how these edges
    obtain their* ``a``.

    ``radius_i`` / ``radius_j`` are per-edge atom radii and are unused (and may be None) for
    ``fixed_0.39``, which by construction depends on no radius data.
    """
    if parameterization == "fixed_0.39":
        return DEFAULT_THOLE_PARAMETER

    if parameterization == "ts_mbis_radii":
        return _thole_parameter_from_damping_length(
            _edge_damping_length_ang(radius_i, radius_j), alpha_i, alpha_j
        )

    raise ValueError(
        f"parameterization must be one of {THOLE_PARAMETERIZATIONS!r}; "
        f"got {parameterization!r}"
    )


# --- row-level science ------------------------------------------------------


def compute_ipd_row(
    row,
    *,
    inter_thole_parameterization="fixed_0.39",
    intra_thole_parameterization="fixed_0.39",
    inter_damping_style="mutual",
    intra_damping_style="mutual",
    bare_param=100.0,
    include_H=True,
    verbose=False,
):
    """Thole-damped IPD induction for a *single* dataframe row.

    Every AB and intramolecular edge is damped; the two parameterizations decide where each
    edge's Thole parameter comes from. Hydrogen is optionally made unpolarizable. Returns the
    results keyed by the same roles :func:`ipd_column_names` uses, plus ``iterations`` and
    ``converged``.

    The row must carry ``qcel_molecule`` (a two-fragment :class:`qcelemental.Molecule`,
    geometry in bohr) and the eight MBIS columns read below.

    ``bare_param`` is accepted for compatibility and is **not used**: it was the value
    substituted on undamped edges, and no edge is undamped -- every AB, AA and BB edge is
    damped by construction, so the selection mask it fed was always all-True.
    """
    from apnet_pt.AtomPairwiseModels.mtp_mtp import (
        induced_dipole_induction_optimized_no_correction,
    )
    from apnet_pt import constants as apnet_constants
    from apnet_pt.pt_datasets.ap2_fused_ds import (
        qcel_dimer_to_fused_data,
        ap2_fused_collate_update_no_target,
    )
    import torch

    _validate_thole_parameterization(
        inter_thole_parameterization, name="inter_thole_parameterization"
    )
    _validate_thole_parameterization(
        intra_thole_parameterization, name="intra_thole_parameterization"
    )

    def log(message):
        if verbose:
            print(message)

    mol = row["qcel_molecule"]

    batch = ap2_fused_collate_update_no_target(
        [qcel_dimer_to_fused_data(mol, r_cut_im=99999.0, dimer_ind=0)]
    )

    def t(col_name):
        return torch.tensor(np.asarray(row[col_name]), dtype=torch.float32)

    qA, qB = t("q hf/adz A"), t("q hf/adz B")
    muA, muB = t("mu hf/adz A"), t("mu hf/adz B")
    quadA, quadB = t("theta hf/adz A"), t("theta hf/adz B")
    hfvr_A, hfvr_B = t("volume ratios A"), t("volume ratios B")

    # Zeroing the H polarizability is NOT done here: hfvr feeds both the radii below and the
    # damping length, which divides by alpha (-> NaN). It happens inside the SCF instead, via
    # include_H, after the tensors are built.
    alpha_A = (
        apnet_constants.polarizability_table[batch.ZA.long()]
        * hfvr_A.reshape(-1) ** (4 / 3.0)
    )
    alpha_B = (
        apnet_constants.polarizability_table[batch.ZB.long()]
        * hfvr_B.reshape(-1) ** (4 / 3.0)
    )

    src, tgt = batch.e_ABsr_source, batch.e_ABsr_target
    e_AA_source, e_AA_target = batch.e_AA_source, batch.e_AA_target
    e_BB_source, e_BB_target = batch.e_BB_source, batch.e_BB_target

    # A universal-0.39 calculation must not depend on TS radius data at all -- and it need not,
    # since nothing downstream reads a radius. Skipping this also skips _ts_mbis_radius_ang's
    # hard failure on the NaN volume ratios some pickles store for bare monatomic ions.
    uses_ts_mbis_radii = "ts_mbis_radii" in (
        inter_thole_parameterization,
        intra_thole_parameterization,
    )
    if uses_ts_mbis_radii:
        R_A = _ts_mbis_radius_ang(batch.ZA.numpy(), hfvr_A.numpy())
        R_B = _ts_mbis_radius_ang(batch.ZB.numpy(), hfvr_B.numpy())
    else:
        R_A = R_B = None

    def edge_radii(radii, index):
        """Per-edge atom radii, or None when radii were never computed."""
        return None if radii is None else radii[index.numpy()]

    # Intermolecular AB. Per-edge alphas are gathered in the same (source, target) order the
    # kernel walks the edges in, so they line up edge-for-edge with what it sees.
    R_A_AB, R_B_AB = edge_radii(R_A, src), edge_radii(R_B, tgt)
    a_edge_AB = _build_thole_parameters(
        inter_thole_parameterization,
        R_A_AB,
        R_B_AB,
        alpha_A.index_select(0, src),
        alpha_B.index_select(0, tgt),
    )
    log(f"{a_edge_AB = }")

    # Intramolecular A-A.
    a_edge_AA = _build_thole_parameters(
        intra_thole_parameterization,
        edge_radii(R_A, e_AA_source),
        edge_radii(R_A, e_AA_target),
        alpha_A.index_select(0, e_AA_source),
        alpha_A.index_select(0, e_AA_target),
    )
    log(f"{a_edge_AA = }")

    # Intramolecular B-B.
    a_edge_BB = _build_thole_parameters(
        intra_thole_parameterization,
        edge_radii(R_B, e_BB_source),
        edge_radii(R_B, e_BB_target),
        alpha_B.index_select(0, e_BB_source),
        alpha_B.index_select(0, e_BB_target),
    )
    log(f"{a_edge_BB = }")

    a_ang = float("nan")
    if inter_thole_parameterization == "ts_mbis_radii":
        lengths = _edge_damping_length_ang(R_A_AB, R_B_AB)
        if lengths.size:
            a_ang = float(lengths.min())

    with torch.no_grad():
        (
            ind, ind_qu, ind_uu, muiA, muiB,
            lam3_AB, lam5_AB, lam3_AA, lam5_AA, lam3_BB, lam5_BB,
            muiA_hist, muiB_hist,
        ) = induced_dipole_induction_optimized_no_correction(
            ZA=batch.ZA, RA=batch.RA, qA=qA, muA=muA, quadA=quadA,
            alpha_A_external=alpha_A,
            ZB=batch.ZB, RB=batch.RB, qB=qB, muB=muB, quadB=quadB,
            alpha_B_external=alpha_B,
            e_AB_source=src, e_AB_target=tgt,
            e_AA_source=e_AA_source, e_BB_source=e_BB_source,
            e_AA_target=e_AA_target, e_BB_target=e_BB_target,
            max_iterations=MAX_SCF_ITERATIONS,
            thole_damping_param=DEFAULT_THOLE_PARAMETER,  # unused fallback
            # inter parameterization -> AB a values, intra -> AA/BB a values.
            thole_damping_param_direct=a_edge_AB,     # AB permanent->induced + energy
            thole_damping_param_mutual_AB=a_edge_AB,  # AB induced<->induced
            thole_damping_param_AA=a_edge_AA,         # intra-A induced<->induced
            thole_damping_param_BB=a_edge_BB,         # intra-B induced<->induced
            # inter style -> AB damping form, intra style -> AA/BB damping form.
            direct_damping_style=inter_damping_style,
            mutual_AB_damping_style=inter_damping_style,
            AA_damping_style=intra_damping_style,
            BB_damping_style=intra_damping_style,
            return_components=True,
            return_induced_dipoles=True,
            return_lambdas=True,
            return_induced_dipole_history=True,
            debug=verbose,
            include_H=include_H,
        )

    if not include_H:
        # Direct check on the include_H contract rather than trusting it: alpha_H is zeroed
        # inside the SCF, so every H row of the converged induced dipoles must be exactly zero
        # (not merely small).
        assert not muiA[batch.ZA == 1].any(), "nonzero induced dipole on an A hydrogen"
        assert not muiB[batch.ZB == 1].any(), "nonzero induced dipole on a B hydrogen"

    # The history's leading axis is the pre-SCF seed plus one entry per iteration run, so it is
    # also the iteration count -- and the SCF breaks on convergence, which makes "ran all the
    # way to the cap" exactly the non-converged case.
    iterations = int(np.asarray(muiA_hist).shape[0]) - 1

    return {
        "energy": ind.sum().item(),
        "energy_qu": ind_qu.sum().item(),
        "energy_uu": ind_uu.sum().item(),
        "a_ang": a_ang,
        # Converged induced dipoles per atom (a.u.).
        "mu_ind_A": muiA.numpy(),
        "mu_ind_B": muiB.numpy(),
        # Per-iteration SCF trajectory (a.u.), shape (n_iter+1, n_atoms, 3); entry 0 is the
        # pre-SCF direct-field seed. The browser visualizer depends on this shape.
        "mu_hist_A": muiA_hist,
        "mu_hist_B": muiB_hist,
        # Per-edge Thole lambda_3/lambda_5 for each channel.
        "lam3_AB": lam3_AB.numpy(),
        "lam5_AB": lam5_AB.numpy(),
        "lam3_AA": lam3_AA.numpy(),
        "lam5_AA": lam5_AA.numpy(),
        "lam3_BB": lam3_BB.numpy(),
        "lam5_BB": lam5_BB.numpy(),
        "iterations": iterations,
        "converged": iterations < MAX_SCF_ITERATIONS,
    }


# --- whole-dataframe entry point --------------------------------------------


def compute_ipd_all(
    df,
    *,
    inter_thole_parameterization="fixed_0.39",
    intra_thole_parameterization="fixed_0.39",
    inter_damping_style="mutual",
    intra_damping_style="mutual",
    bare_param=100.0,
    include_H=True,
    suffix="",
    verbose=True,
):
    """Run :func:`compute_ipd_row` over every row of ``df``, writing results in place.

    Takes and returns a DataFrame rather than a filename: the caller owns reading and writing,
    which keeps persistence -- and the atomicity of it -- out of the science.

    No duplicate science: this is a loop over :func:`compute_ipd_row` and
    :func:`write_ipd_row`.
    """
    choices = {
        "inter_thole_parameterization": inter_thole_parameterization,
        "intra_thole_parameterization": intra_thole_parameterization,
        "include_H": include_H,
    }
    for idx, row in df.iterrows():
        result = compute_ipd_row(
            row,
            inter_damping_style=inter_damping_style,
            intra_damping_style=intra_damping_style,
            bare_param=bare_param,
            verbose=verbose,
            **choices,
        )
        write_ipd_row(df, idx, result, suffix=suffix, **choices)
    return df
