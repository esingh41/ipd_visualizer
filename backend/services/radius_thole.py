"""Radius-based Thole IPD induction, and the naming scheme its results are stored under.

This module is the source of truth for three things, and nothing outside it should
reimplement any of them:

* which dataframe columns an IPD run writes (:func:`ipd_column_names`);
* how one geometry's induced dipoles are computed (:func:`compute_ipd_row`);
* how a result is written to, detected in, and read back out of a dataframe
  (:func:`write_ipd_row`, :func:`has_ipd_result`, :func:`read_ipd_row`).

:func:`compute_ipd_radius_thole_all` -- the original whole-file entry point -- is a thin
loop over those, so the command-line path and ``backend.services.ipd_compute``'s
on-demand path run identical code.

``torch`` and ``apnet_pt`` are imported lazily inside the computing functions rather than
at module scope, so the naming and dataframe-inspection half of this API stays usable on
a machine that cannot run the science at all. That is what lets the web app report
"apnet_pt unavailable" instead of failing to start.
"""

import pandas as pd
import functools
import qcelemental as qcel
import numpy as np
from pprint import pprint as pp
from importlib import resources

_ANG_PER_BOHR = qcel.constants.conversion_factor("bohr", "angstrom")

# The four mutually exclusive damping modes. ``None`` is the Python value, not the string
# "None": a string here is truthy, which would leave the radius-based edge parameters
# active while the column name below fell through to the undamped "IPD (MBIS) 0.39 all",
# storing radius-damped numbers under the undamped name with no error.
RADIUS_MODES = (None, "intra", "inter", "both")

# apnet_pt's own default. Passed explicitly so the convergence test in compute_ipd_row --
# "did the SCF run all the way to the cap?" -- cannot drift away from the real limit.
MAX_SCF_ITERATIONS = 200

# Every column an IPD run writes, as a role -> the prefix substituted for "IPD (" in the
# base column name. The base name itself is the "energy" role.
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

# Roles holding one number per row; everything else holds a NumPy array and therefore
# needs an object-dtype column.
_SCALAR_ROLES = frozenset({"energy", "energy_qu", "energy_uu", "a_ang"})

# What must be present for a stored result to be usable by the visualizer. The lambdas and
# component energies are diagnostics -- a row missing those is still renderable.
_ESSENTIAL_ROLES = ("energy", "mu_hist_A", "mu_hist_B")


# --- column naming ----------------------------------------------------------


def ipd_column_names(radius=None, suffix=""):
    """Every column an IPD run at this ``radius`` writes, keyed by role.

    The only place these names are constructed. ``radius`` selects the base name; the
    other twelve are derived from it by substitution, which is what keeps a run at one
    damping mode from overwriting another's results.
    """
    if radius not in RADIUS_MODES:
        raise ValueError(
            f"radius must be one of {RADIUS_MODES!r}; got {radius!r}. "
            "Note the string 'None' is not accepted -- pass the Python value None."
        )

    # H keeps its permanent multipoles but gets no induced dipole. Tag the column
    # automatically so an include_H=False run can never overwrite the include_H=True
    # one, and so the radius-damped-intra run never overwrites the constant-0.39-intra
    # one; an explicitly-passed suffix still wins.
    if radius == "inter":
        col = "IPD (radius thole ts_mbis inter only)"
    elif radius == "intra":
        col = "IPD (radius thole ts_mbis intra only)"
    elif radius == "both":
        col = "IPD (radius thole ts_mbis intra inter)"
    else:
        col = "IPD (MBIS) 0.39 all"

    if suffix:
        # "IPD (MBIS) 0.39 all" has no trailing paren to splice into, so it gets its own
        # parenthesised group rather than losing its final character.
        col = f"{col[:-1]}, {suffix})" if col.endswith(")") else f"{col} ({suffix})"

    names = {"energy": col}
    for role, prefix in _DERIVED_PREFIXES.items():
        names[role] = col.replace("IPD (", prefix)
    return names


# --- dataframe read/write ---------------------------------------------------


def _is_missing(value):
    """True when a dataframe cell holds no usable result.

    ``pd.isna`` returns an *array* for an array cell, which is not a truth value, so
    arrays are tested on their size instead.
    """
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def has_ipd_result(df, idx, radius=None, suffix=""):
    """True when row ``idx`` already carries a usable IPD result for this radius mode."""
    names = ipd_column_names(radius, suffix)
    if any(names[role] not in df.columns for role in _ESSENTIAL_ROLES):
        return False
    return not any(_is_missing(df.at[idx, names[role]]) for role in _ESSENTIAL_ROLES)


def read_ipd_row(df, idx, radius=None, suffix=""):
    """Read a stored result back out, in the same shape :func:`compute_ipd_row` returns.

    Raises KeyError when no result is stored, so callers must gate on
    :func:`has_ipd_result` first.
    """
    if not has_ipd_result(df, idx, radius, suffix):
        raise KeyError(f"No stored IPD result at row {idx!r} for radius={radius!r}")

    names = ipd_column_names(radius, suffix)
    result = {}
    for role, col in names.items():
        value = df.at[idx, col] if col in df.columns else None
        result[role] = None if _is_missing(value) else value

    mu_hist_A = np.asarray(result["mu_hist_A"])
    result["iterations"] = int(mu_hist_A.shape[0]) - 1
    result["converged"] = result["iterations"] < MAX_SCF_ITERATIONS
    return result


def write_ipd_row(df, idx, result, radius=None, suffix=""):
    """Write one row's result into ``df`` at label ``idx``, in place.

    Missing columns are created first -- ``df.at[]`` cannot create a column, and an array
    value needs an object-dtype one to land in rather than being broadcast.
    """
    names = ipd_column_names(radius, suffix)
    for role, col in names.items():
        if col not in df.columns:
            if role in _SCALAR_ROLES:
                df[col] = np.nan
            else:
                df[col] = pd.Series([None] * len(df), index=df.index, dtype=object)
        df.at[idx, col] = result[role]


# --- science ----------------------------------------------------------------


def _ts_mbis_radius_ang(Z, volume_ratios):
    """Per-atom damping radius (Angstrom) from the neutral ``R_vdw(TS)`` column of
    ``vdw-params.csv``, scaled by the MBIS Hirshfeld volume ratio$^\\frac{1}{3}$.

    More generalizable than a fixed ion/vdW-radius table, and its appeal is that
    *everything* comes from the one CSV: the ionic rows there have
    ``R_vdw(TS) = NaN``, but Psi4's MBIS volume ratio for a monatomic-ion fragment
    is referenced to the neutral free atom, so the ratio already carries the neutral->ion contraction.
    """
    tab = _read_full_polarizability_table()
    Z = np.asarray(Z).astype(int).ravel()
    ratios = np.asarray(volume_ratios, dtype=float).ravel()

    # Neutral rows only: "Na" and "Na+" are distinct index strings, so looking up
    # the bare element symbol never hits an ionic row (which are all NaN here).
    r_free = np.array(
        [float(tab.loc[qcel.periodictable.to_E(int(z)), "R_vdw(TS)"]) for z in Z]
    )
    out = r_free * ratios ** (1.0 / 3.0) * _ANG_PER_BOHR
    if not np.all(np.isfinite(out)):
        raise ValueError(
            f"non-finite TS/MBIS radius for Z={Z.tolist()} ratios={ratios.tolist()}; "
            "R_vdw(TS) is NaN for ionic rows and some pickles store NaN monomer "
            "volume ratios for bare monatomic ions"
        )
    return out


def _radius_thole_edge_param(a_length_ang, alpha_i, alpha_j, damp_mask, bare=100.0):
    """Convert a fixed Thole damping *length* (Angstrom) into the per-edge
    dimensionless Thole parameter apnet_pt expects.
    """
    import torch

    a_M_bohr = torch.as_tensor(
        a_length_ang, dtype=alpha_i.dtype, device=alpha_i.device
    ) / _ANG_PER_BOHR
    a_q = torch.sqrt(alpha_i * alpha_j) / (a_M_bohr ** 3)
    return torch.where(damp_mask, a_q, torch.full_like(a_q, float(bare)))


@functools.lru_cache(maxsize=1)
def _read_full_polarizability_table():
    """Untruncated ``vdw-params.csv`` (includes ionic rows like "Na+", "Cl-"),
    indexed by the ``symbol`` column. apnet_pt's own ``polarizability_table``
    (constants.py) truncates this same file to ``nrows=102`` (neutral
    elements only, Z-indexable); we read it without that limit to reach the
    ion rows, which only have unique ``symbol`` strings (not unique Z)."""
    path = resources.files("apnet_pt").joinpath("data", "vdw-params.csv")
    return pd.read_csv(path, header=0, index_col=0, sep=",")


def compute_ipd_row(row, radius=None, bare_param=100.0, include_H=True,
                    damping_style="mutual", intra_damping_style="mutual",
                    verbose=False):
    """Radius-based Thole IPD induction for a *single* dataframe row.

    Every AB *and* intramolecular edge is damped, and hydrogen is optionally made
    unpolarizable. Returns the results keyed by the same roles :func:`ipd_column_names`
    uses, plus ``iterations`` and ``converged``.

    The row must carry ``qcel_molecule`` (a two-fragment :class:`qcelemental.Molecule`,
    geometry in bohr) and the eight MBIS columns read below.
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

    if radius not in RADIUS_MODES:
        raise ValueError(
            f"radius must be one of {RADIUS_MODES!r}; got {radius!r}. "
            "Note the string 'None' is not accepted -- pass the Python value None."
        )

    def log(message):
        if verbose:
            print(message)

    mol = row["qcel_molecule"]
    log(f"{row['system_id'] = }")

    batch = ap2_fused_collate_update_no_target(
        [qcel_dimer_to_fused_data(mol, r_cut_im=99999.0, dimer_ind=0)]
    )

    def t(col_name):
        return torch.tensor(np.asarray(row[col_name]), dtype=torch.float32)

    qA, qB = t("q hf/adz A"), t("q hf/adz B")
    muA, muB = t("mu hf/adz A"), t("mu hf/adz B")
    quadA, quadB = t("theta hf/adz A"), t("theta hf/adz B")
    hfvr_A, hfvr_B = t("volume ratios A"), t("volume ratios B")

    # Zeroing the H polarizability is NOT done here: hfvr feeds both the radii
    # below and the damping length, which divides by alpha (-> NaN). It happens
    # inside the SCF instead, via include_H, after the tensors are built.
    alpha_A = apnet_constants.polarizability_table[batch.ZA.long()] * hfvr_A.reshape(-1) ** (4 / 3.0)
    alpha_B = apnet_constants.polarizability_table[batch.ZB.long()] * hfvr_B.reshape(-1) ** (4 / 3.0)

    src, tgt = batch.e_ABsr_source, batch.e_ABsr_target

    # Per-edge alphas for the intermolecular damping-length conversion below.
    alpha_i = alpha_A.index_select(0, src)
    alpha_j = alpha_B.index_select(0, tgt)

    # R_vdw(TS) * ratio^(1/3).
    # Getting scaled TS vdw radii with hfvr instead of the ionic radii
    R_A = _ts_mbis_radius_ang(batch.ZA.numpy(), hfvr_A.numpy())
    R_B = _ts_mbis_radius_ang(batch.ZB.numpy(), hfvr_B.numpy())

    # Building the thole damping parameters for each intramolecular edge, this is for monomer A, monomer A damping.
    e_AA_source = batch.e_AA_source
    e_AA_target = batch.e_AA_target

    a_len_AA = torch.tensor(
        np.hypot(R_A[e_AA_source.numpy()], R_A[e_AA_target.numpy()]), dtype=torch.float32
    )
    log(f"{a_len_AA = }")

    # Per-edge alphas for the AA damping-length conversion. distance_tensors gathers
    # alpha by (e_source, e_target) in this same order, so the resulting (n_AA,)
    # tensor lines up edge-for-edge with what the kernel sees.
    alpha_AA_i = alpha_A.index_select(0, e_AA_source)
    alpha_AA_j = alpha_A.index_select(0, e_AA_target)
    a_edge_AA = _radius_thole_edge_param(
        a_len_AA, alpha_AA_i, alpha_AA_j,
        damp_mask=torch.ones_like(a_len_AA, dtype=torch.bool),
        bare=bare_param,
    )
    log(f"{a_edge_AA = }")

    # Building the thole damping parameters for each intramolecular edge, this is for monomer B, monomer B damping.
    e_BB_source = batch.e_BB_source
    e_BB_target = batch.e_BB_target

    a_len_BB = torch.tensor(
        np.hypot(R_B[e_BB_source.numpy()], R_B[e_BB_target.numpy()]), dtype=torch.float32
    )
    log(f"{a_len_BB = }")

    alpha_BB_i = alpha_B.index_select(0, e_BB_source)
    alpha_BB_j = alpha_B.index_select(0, e_BB_target)
    a_edge_BB = _radius_thole_edge_param(
        a_len_BB, alpha_BB_i, alpha_BB_j,
        damp_mask=torch.ones_like(a_len_BB, dtype=torch.bool),
        bare=bare_param,
    )
    log(f"{a_edge_BB = }")

    # Symmetric per-edge form: a = sqrt(R_i^2 + R_j^2) with both radii from, this is for the intermolecular edges
    a_len = torch.tensor(
        np.hypot(R_A[src.numpy()], R_B[tgt.numpy()]), dtype=torch.float32
    )
    log(f"{a_len = }")

    # No damp_selection here: every intermolecular edge gets the radius-based
    # length, so the mask is all-True and bare_param is never reached. It exists
    # only to satisfy _radius_thole_edge_param's signature.
    damp_mask = torch.ones_like(a_len, dtype=torch.bool)
    # Getting thole damping parameter per edge
    a_edge = _radius_thole_edge_param(a_len, alpha_i, alpha_j, damp_mask=damp_mask, bare=bare_param)

    # If we do not want to apply the masia style damping where we use the vdw radii to get the thole damping value, then restoring
    # all damping values to 0.39
    if not radius:
        a_edge = 0.39
        a_edge_AA = 0.39
        a_edge_BB = 0.39
    elif radius == "inter":
        a_edge_AA = 0.39
        a_edge_BB = 0.39
    elif radius == "intra":
        a_edge = 0.39

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
            e_AA_source=batch.e_AA_source, e_BB_source=batch.e_BB_source,
            e_AA_target=batch.e_AA_target, e_BB_target=batch.e_BB_target,
            max_iterations=MAX_SCF_ITERATIONS,
            thole_damping_param=0.39,            # unused fallback; AA/BB set below
            thole_damping_param_direct=a_edge,   # AB permanent->induced + energy
            thole_damping_param_mutual_AB=a_edge,  # AB induced<->induced
            thole_damping_param_AA=a_edge_AA,    # intra-A induced<->induced
            thole_damping_param_BB=a_edge_BB,    # intra-B induced<->induced
            # Damping *form* on both intermolecular channels: "mutual" (the
            # 1-exp(-a u^3) default) or "linear" (Thole's original piecewise
            # form, Masia Eqs. 7-8).
            direct_damping_style=damping_style,
            mutual_AB_damping_style=damping_style,
            # AA/BB now carry radius-based per-edge lengths rather than the
            # constant 0.39; keep them on "mutual" so a_M stays a true length.
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
        # Direct check on the include_H contract rather than trusting it: alpha_H
        # is zeroed inside the SCF, so every H row of the converged induced
        # dipoles must be exactly zero (not merely small).
        assert not muiA[batch.ZA == 1].any(), "nonzero induced dipole on an A hydrogen"
        assert not muiB[batch.ZB == 1].any(), "nonzero induced dipole on a B hydrogen"

    # The history's leading axis is the pre-SCF seed plus one entry per iteration run, so
    # it is also the iteration count -- and the SCF breaks on convergence, which makes
    # "ran all the way to the cap" exactly the non-converged case. apnet_pt can report
    # this directly via return_iterations, but asking for it would change the shape of
    # the tuple unpacked above.
    iterations = int(np.asarray(muiA_hist).shape[0]) - 1

    return {
        "energy": ind.sum().item(),
        "energy_qu": ind_qu.sum().item(),
        "energy_uu": ind_uu.sum().item(),
        # Damping length actually applied (Angstrom). Min over *every* AB edge, not
        # heavy-only: the contact pair can be an H edge (Cl-...H in cl_water_dimers).
        "a_ang": float(a_len.min()) if a_len.numel() else float("nan"),
        # Converged induced dipoles per atom (a.u.) -- lets the molecular-dipole
        # comparison (plot_na_water_induction.plot_dipole_turnover) be rebuilt for
        # this damping variant, not just its energy.
        "mu_ind_A": muiA.numpy(),
        "mu_ind_B": muiB.numpy(),
        # Per-iteration SCF trajectory (a.u.), shape (n_iter+1, n_atoms, 3); entry 0
        # is the pre-SCF direct-field seed. Powers pymol_dipole_movie.py-style
        # convergence animations, and the browser visualizer.
        "mu_hist_A": muiA_hist,
        "mu_hist_B": muiB_hist,
        # Per-edge Thole lambda_3/lambda_5 for each channel (AB, and the two
        # intramolecular AA/BB channels) -- see induced_dipole_induction_optimized_no_correction's
        # return_lambdas for what these mean.
        "lam3_AB": lam3_AB.numpy(),
        "lam5_AB": lam5_AB.numpy(),
        "lam3_AA": lam3_AA.numpy(),
        "lam5_AA": lam5_AA.numpy(),
        "lam3_BB": lam3_BB.numpy(),
        "lam5_BB": lam5_BB.numpy(),
        "iterations": iterations,
        "converged": iterations < MAX_SCF_ITERATIONS,
    }


def compute_ipd_radius_thole_all(filename, a_length_ang=None,
                                 bare_param=100.0, suffix="", include_H=True,
                                 radius_source="ts_mbis",
                                 damping_style="mutual",
                                 intra_damping_style="mutual",
                                 radius=None,
                                 verbose=True,
                                 ):
    """Run :func:`compute_ipd_row` over every row of a pickled dataframe, in place.

    ``radius_source`` defaults to ``"ts_mbis"`` because :func:`compute_ipd_row` computes
    radii with :func:`_ts_mbis_radius_ang` unconditionally.
    """
    df = pd.read_pickle(filename)
    for idx, row in df.iterrows():
        result = compute_ipd_row(
            row, radius,
            bare_param=bare_param,
            include_H=include_H,
            damping_style=damping_style,
            intra_damping_style=intra_damping_style,
            verbose=verbose,
        )
        write_ipd_row(df, idx, result, radius, suffix)
    df.to_pickle(filename)
    return df


def plot_damping_functions():
    from apnet_pt.pt_datasets.ap2_fused_ds import (
        qcel_dimer_to_fused_data,
        ap2_fused_collate_update_no_target,
    )
    from apnet_pt import multipole
    import torch

    df = pd.read_pickle("na_water_dimers.pkl")
    row = df.iloc[0]
    fragments = row['qcel_molecule'].fragments
    batch = ap2_fused_collate_update_no_target(
            [qcel_dimer_to_fused_data(row['qcel_molecule'], r_cut_im=99999.0, dimer_ind=0)]
    )
    e_AB_src = batch.e_ABsr_source
    e_AB_tgt = batch.e_ABsr_target

    ZA = row['qcel_molecule'].get_fragment(0).atomic_numbers
    RA = row['qcel_molecule'].get_fragment(0).geometry
    ZB = row['qcel_molecule'].get_fragment(1).atomic_numbers
    RB = row['qcel_molecule'].get_fragment(1).geometry
    tab = _read_full_polarizability_table()
    # ZA = np.asarray(ZA).astype(int).ravel()
    # ZB = np.asarray(ZB).astype(int).ravel()

    hfvr_A = row['volume ratios A']
    assert np.allclose(hfvr_A, np.array([1.39085945, 0.18787692, 0.19180905])), "mismatch in water hfvr"
    hfvr_B = row['volume ratios B']
    assert np.allclose(hfvr_B, np.array([0.06302156])), "mismatch in sodium ion hfvr"

    # Neutral rows only: "Na" and "Na+" are distinct index strings, so looking up
    # the bare element symbol never hits an ionic row (which are all NaN here).
    alpha_free_A = np.array(
        [float(tab.loc[qcel.periodictable.to_E(int(z)), "alpha_0(BG)"]) for z in ZA]
    )
    print(f"{alpha_free_A = }")
    alpha_free_B = np.array(
        [float(tab.loc[qcel.periodictable.to_E(int(z)), "alpha_0(BG)"]) for z in ZB]
    )

    assert np.allclose(alpha_free_A, np.array([5.200326538, 4.499159818, 4.499159818])), "Mismatch from expected free atom polarizabilities"
    assert alpha_free_B == [163.0799998], "Mismatch from expected free atom polarizabilities"

    # free atom polarizabilities are in bohr, hfvr need to be raised to 4/3 power not 1/3
    # always did wonder about the 4/3
    alpha_A = alpha_free_A * hfvr_A ** (4.0 / 3.0)
    assert np.allclose(alpha_A, np.array([8.07374339, 0.48413074, 0.49768765])), "Mismatch from expected scaled hfvr atom polarizabilities for monomer A"
    print(f"{alpha_A = }")

    alpha_B = alpha_free_B * hfvr_B ** (4.0 / 3.0)

    print(f"{alpha_B = }")
    assert np.allclose(alpha_B, np.array([4.08996498])), "Mismatch from expected scaled hfvr atom polarizabilities for monomer A"

    alpha_A_AB = alpha_A[np.array(e_AB_src)]
    print(f"{alpha_A_AB = }")
    alpha_B_AB = alpha_B[np.array(e_AB_tgt)]
    print(f"{alpha_B_AB = }")
    print(f"{np.sqrt(alpha_A_AB * alpha_B_AB) = }")
    prefactor_pol = np.sqrt(alpha_A_AB * alpha_B_AB)
    assert np.allclose(prefactor_pol, np.array([5.74641869, 1.40715236, 1.42671829]))
    prefactor_pol = 1 / prefactor_pol

    vdw_free_A = np.array(
        [float(tab.loc[qcel.periodictable.to_E(int(z)), "R_vdw(TS)"]) for z in ZA]
    )
    assert np.allclose(vdw_free_A, np.array([3.19, 3.1, 3.1])), "mismatch in free vdw radii for water"
    vdw_A = vdw_free_A * (hfvr_A ** (1/3))
    print(f"{vdw_A = }")
    assert np.allclose(vdw_A, np.array([3.5608343 , 1.7754952 , 1.78779639]))
    vdw_free_B = np.array(
        [float(tab.loc[qcel.periodictable.to_E(int(z)), "R_vdw(TS)"]) for z in ZB]
    )
    vdw_B = vdw_free_B * (hfvr_B ** (1/3))
    assert np.allclose(vdw_free_B, np.array([3.73])), "mismatch in free vdw radii for sodium ion"
    print(f"{vdw_B = }")
    assert np.allclose(vdw_B, np.array([1.48435764]))

    vdw_A_AB = vdw_A[np.array(e_AB_src)]
    print(f"{vdw_A_AB = }")
    vdw_B_AB = vdw_B[np.array(e_AB_tgt)]
    print(f"{vdw_B_AB = }")
    print(f"{((vdw_A_AB ** 2 + vdw_B_AB ** 2) ** 0.5) ** 3 = }")
    prefactor_radii = 1 / (((vdw_A_AB ** 2 + vdw_B_AB ** 2) ** 0.5) ** 3)

    # The prefactors associated with the vdw radii are smaller, an order of 10 orders of magnitude smaller, this indicates that the exponential function decays slower than the one with polarizabilities
    assert np.allclose(prefactor_radii, np.array([0.01741688, 0.08068179, 0.07970134])), "expected prefactors for Masia model differ than expected"

    assert np.allclose(prefactor_pol, np.array([0.17402143, 0.7106551 , 0.70090922]))
    print(f"{prefactor_pol * 0.39 = }")

    print(f"{prefactor_radii = }")

    print(f"{RA = }")
    RA_AB = RA[e_AB_src]
    print(f"{RA_AB = }")
    RB_AB = RB[e_AB_tgt]
    print(f"{RB_AB = }")
    R_ij = (torch.sum(torch.tensor((RA_AB - RB_AB)), dim=1) ** 2) ** 0.5
    print(f"{torch.sum(torch.tensor((RA_AB - RB_AB)), dim=1) = }")
    print(f"{R_ij = }")
    au3, l3, l5 = multipole.thole_damping_mutual_torch(R_ij, torch.tensor(alpha_A), torch.tensor(alpha_B), a=0.39)
    print(f"{l3 = }")
    au3, l3, l5 = multipole.thole_damping_mutual_torch_masia(R_ij, torch.tensor(vdw_A), torch.tensor(vdw_B), a=1.0)
    print(f"{l3 = }")

    # multipole.thole_damping_mutual_torch()


def main():
    filename = "01_water-water_dimer_multipoles_hippo.pkl"
    filename = "na_water_dimers.pkl"
    compute_ipd_radius_thole_all(filename, radius=None)
    compute_ipd_radius_thole_all(filename, radius="intra")
    compute_ipd_radius_thole_all(filename, radius="inter")
    compute_ipd_radius_thole_all(filename, radius="both")
    df = pd.read_pickle(filename)
    pp(df.columns.tolist())
    print(df[['system_id', 'lam3 AA (radius thole ts_mbis intra only)', 'IPD (radius thole ts_mbis intra only)','lam3 AA (MBIS) 0.39 all','IPD (MBIS) 0.39 all',]].to_string())
    return


if __name__ == "__main__":
    main()
