from apnet_pt.multipole import (
    multipoles_elst_ind_dimer,
)

def compute_mtp_elst_ind(filename):
    """Add ``mtp elst`` / ``mtp elst dimer`` / ``mtp ind`` columns to a pickle."""
    df = pd.read_pickle(filename)

    mols, dimer, monA, monB = [], [], [], []
    for _, r in df.iterrows():
        mols.append(r["qcel_molecule"])
        monA.append(
            (
                np.asarray(r["q hf/adz A"]),
                np.asarray(r["mu hf/adz A"]),
                np.asarray(r["theta hf/adz A"]),
            )
        )
        monB.append(
            (
                np.asarray(r["q hf/adz B"]),
                np.asarray(r["mu hf/adz B"]),
                np.asarray(r["theta hf/adz B"]),
            )
        )
        dimer.append(
            (
                np.asarray(r["q hf/adz dimer"]),
                np.asarray(r["mu hf/adz dimer"]),
                np.asarray(r["theta hf/adz dimer"]),
            )
        )

    E_elst, E_elst_dimer, E_ind = multipoles_elst_ind_dimer(mols, dimer, monA, monB)

    df["mtp elst"] = E_elst  # elst from isolated-monomer multipoles
    df["mtp elst dimer"] = E_elst_dimer  # elst from dimer multipoles
    df["mtp ind"] = E_ind  # induction = dimer elst - monomer elst
    # Contracted (variational-1/2) induction: Delta MTP is an un-halved mu_ind.F
    # estimate, so 1/2 . Delta MTP is the physically-scaled induction that
    # collapses onto SAPT induction at long range.
    df["mtp ind / 2"] = np.asarray(E_ind) / 2.0

    df.to_pickle(filename)
    return df

