"""The uploaded column vocabulary: which names we recognise, and what each feature needs.

Names only. Whether the values are valid belongs to ``dataframe_validation``, what can be
derived from them to ``system_processing``, and what gets persisted to
``system_serialization``. Nothing here opens a dataframe.

Only ``qcel_molecule`` is required. Everything else is a capability: a frame carrying
nothing but molecules still animates, and a tab that cannot run should say what it is
missing rather than disappear.

Two things are deliberately *not* recognised, because the app derives them itself and an
uploaded column claiming otherwise would be believed:

* Monomer membership always comes from ``qcel_molecule``'s fragments. Atom-count columns
  such as "# of atoms in Monomer A" are not authoritative.
* The closest intermolecular contact is always computed from geometry, so "closest contact
  ang" and per-pair distances like "O-Na dist ang" are ignored.

IPD result columns are not listed here either. ``thole_damping.ipd_column_names`` is the
source of truth for those; code that needs to detect a stored result asks it directly, via
``ipd_results``.

``ENERGY_DEFINITIONS`` and :func:`classify_energy_definition` are the exception to "names
only" in spirit but not in mechanism: they attach meaning to accepted source names -- family,
level, category and role -- so the browser can group and label curves without parsing column
strings. Still no dataframe is opened. Turning that vocabulary into a catalog and per-frame
numbers is ``energies``'s job.
"""

# 2: frames carry a "multipoles" block.
# 3: frames carry an "ipd" block -- per-mode availability and iteration counts.
# 4: systems carry an "energy_catalog", and frames an "energies" block keyed by energy id.
# 5: each stored IPD mode carries max_abs_mu for one mode-wide scale across a trajectory.
# 6: energy catalogs recognize the two valid AMOEBA+ polarization parameterizations.
# 7: catalogs include raw SAPT interaction components and SAPT(DFT) entries.
# 8: frame-level IPD computability also rejects missing/non-finite required cell values.
# 9: frames carry Delta-MTP computability/result metadata and catalogs recognize all four
#    Delta-MTP outputs.
# Bumping this is what makes an already-stored collection reprocess instead of being returned
# untouched -- see upload_system.process_upload.
SCHEMA_VERSION = 9

REQUIRED_COLUMNS = {
    "qcel_molecule",
}

# Spellings seen in real uploads, mapped to the name the rest of the app uses.
# 131_Na-benzene.pkl writes the monomer molecules with a space.
COLUMN_ALIASES = {
    "system id": "system_id",
    "qcel molecule": "qcel_molecule",
    "qcel molecule A": "qcel_molecule A",
    "qcel molecule B": "qcel_molecule B",
}

MBIS_COLUMNS = {
    "q hf/adz A",
    "q hf/adz B",
    "q hf/adz dimer",

    # The *permanent* atomic dipole, an input. Not mu_ind, the induced dipole the viewer
    # draws, which is an IPD result and named by thole_damping.
    "mu hf/adz A",
    "mu hf/adz B",
    "mu hf/adz dimer",

    "theta hf/adz A",
    "theta hf/adz B",
    "theta hf/adz dimer",

    "volume ratios A",
    "volume ratios B",
    "volume ratios dimer",
}

# The multipole table's quantities, each naming the three columns it is assembled from: the
# two isolated-monomer arrays, and the dimer array covering every atom.
#
# Names only -- no shapes. What each array should look like belongs with the code that
# validates it, in system_serialization.
#
# theta is deliberately absent. The schema recognises it (MBIS_COLUMNS), but the table does
# not show quadrupoles, and serializing a (n, 3, 3) per atom per side for something unread
# would roughly triple the payload.
MULTIPOLE_QUANTITIES = {
    "charges": {
        "A": "q hf/adz A",
        "B": "q hf/adz B",
        "dimer": "q hf/adz dimer",
    },
    "dipoles": {
        "A": "mu hf/adz A",
        "B": "mu hf/adz B",
        "dimer": "mu hf/adz dimer",
    },
    "volume_ratios": {
        "A": "volume ratios A",
        "B": "volume ratios B",
        "dimer": "volume ratios dimer",
    },
}

# Every plottable uploaded energy, in display order and keyed by a stable output id rather
# than by an input spelling. ``source_columns`` is a priority tuple: one definition emits at
# most one catalog entry even when an upload carries more than one accepted source name.
#
# Values are scientific metadata the frontend groups, labels and styles by. Recognition is by
# exact source name, never by a "sapt" substring heuristic. IPD results remain absent because
# thole_damping constructs their names and energies builds those entries from ipd_results.MODES.
ENERGY_DEFINITIONS = (
    # SAPT0 interaction total, its four raw category totals, then induction breakdown.
    {
        "id": "sapt0_total",
        "source_columns": ("SAPT0 TOTAL kcalmol", "SAPT0 TOTAL ENERGY adz"),
        "label": "Total",
        "category": "interaction",
        "family": "sapt",
        "level": "SAPT0",
        "role": "total",
    },
    {
        "id": "sapt0_elst",
        "source_columns": ("SAPT0 ELST kcalmol", "SAPT0 ELST ENERGY adz"),
        "label": "Electrostatics",
        "category": "electrostatics",
        "parent_category": "interaction",
        "family": "sapt",
        "level": "SAPT0",
        "role": "total",
    },
    {
        "id": "sapt0_exch",
        "source_columns": ("SAPT0 EXCH kcalmol", "SAPT0 EXCH ENERGY adz"),
        "label": "Exchange",
        "category": "exchange",
        "parent_category": "interaction",
        "family": "sapt",
        "level": "SAPT0",
        "role": "total",
    },
    {
        "id": "sapt0_ind",
        "source_columns": ("SAPT0 IND kcalmol", "SAPT0 INDU ENERGY adz"),
        "label": "Induction total",
        "category": "induction",
        "parent_category": "interaction",
        "family": "sapt",
        "level": "SAPT0",
        "role": "total",
    },
    {
        "id": "sapt0_disp",
        "source_columns": (
            "SAPT0 DISP kcalmol",
            "SAPT0 DISP ENERGY adz",
            "D4 ENERGY adz",
        ),
        "label": "Dispersion",
        "category": "dispersion",
        "parent_category": "interaction",
        "family": "sapt",
        "level": "SAPT0",
        "role": "total",
    },
    {
        "id": "sapt0_ind20r",
        "source_columns": ("SAPT ind20,r kcalmol", "ind20,r_sapt0"),
        "label": "ind20,r",
        "category": "induction",
        "family": "sapt",
        "level": "SAPT0",
        "role": "component",
    },
    {
        "id": "sapt0_exch_ind20r",
        "source_columns": (
            "SAPT exch-ind20,r kcalmol",
            "exch-ind20,r_sapt0",
        ),
        "label": "exch-ind20,r",
        "category": "induction",
        "family": "sapt",
        "level": "SAPT0",
        "role": "component",
    },
    {
        "id": "sapt0_dhf",
        "source_columns": ("SAPT dHF ind kcalmol", "delta hf correction"),
        "label": "δHF",
        "category": "induction",
        "family": "sapt",
        "level": "SAPT0",
        "role": "component",
    },

    # SAPT0/cc-pVDZ interaction decomposition and induction breakdown.
    {
        "id": "sapt0_ccpvdz_total",
        "source_columns": ("SAPT0/cc-pVDZ TOTAL kcalmol",),
        "label": "Total",
        "category": "interaction",
        "family": "sapt",
        "level": "SAPT0/cc-pVDZ",
        "role": "total",
    },
    {
        "id": "sapt0_ccpvdz_elst",
        "source_columns": ("SAPT0/cc-pVDZ ELST kcalmol",),
        "label": "Electrostatics",
        "category": "electrostatics",
        "parent_category": "interaction",
        "family": "sapt",
        "level": "SAPT0/cc-pVDZ",
        "role": "total",
    },
    {
        "id": "sapt0_ccpvdz_exch",
        "source_columns": ("SAPT0/cc-pVDZ EXCH kcalmol",),
        "label": "Exchange",
        "category": "exchange",
        "parent_category": "interaction",
        "family": "sapt",
        "level": "SAPT0/cc-pVDZ",
        "role": "total",
    },
    {
        "id": "sapt0_ccpvdz_ind",
        "source_columns": ("SAPT0/cc-pVDZ IND kcalmol",),
        "label": "Induction total",
        "category": "induction",
        "parent_category": "interaction",
        "family": "sapt",
        "level": "SAPT0/cc-pVDZ",
        "role": "total",
    },
    {
        "id": "sapt0_ccpvdz_disp",
        "source_columns": ("SAPT0/cc-pVDZ DISP kcalmol",),
        "label": "Dispersion",
        "category": "dispersion",
        "parent_category": "interaction",
        "family": "sapt",
        "level": "SAPT0/cc-pVDZ",
        "role": "total",
    },
    {
        "id": "sapt0_ccpvdz_ind20r",
        "source_columns": ("SAPT/cc-pVDZ ind20,r kcalmol",),
        "label": "ind20,r",
        "category": "induction",
        "family": "sapt",
        "level": "SAPT0/cc-pVDZ",
        "role": "component",
    },
    {
        "id": "sapt0_ccpvdz_exch_ind20r",
        "source_columns": ("SAPT/cc-pVDZ exch-ind20,r kcalmol",),
        "label": "exch-ind20,r",
        "category": "induction",
        "family": "sapt",
        "level": "SAPT0/cc-pVDZ",
        "role": "component",
    },
    {
        "id": "sapt0_ccpvdz_dhf",
        "source_columns": ("SAPT/cc-pVDZ dHF ind kcalmol",),
        "label": "δHF",
        "category": "induction",
        "family": "sapt",
        "level": "SAPT0/cc-pVDZ",
        "role": "component",
    },

    # SAPT2+/aDZ raw interaction decomposition; no induction breakdown is supplied.
    {
        "id": "sapt2p_adz_total",
        "source_columns": ("SAPT2+/aDZ Tot",),
        "label": "Total",
        "category": "interaction",
        "family": "sapt",
        "level": "SAPT2+/aDZ",
        "role": "total",
    },
    {
        "id": "sapt2p_adz_elst",
        "source_columns": ("SAPT2+/aDZ Elst",),
        "label": "Electrostatics",
        "category": "electrostatics",
        "parent_category": "interaction",
        "family": "sapt",
        "level": "SAPT2+/aDZ",
        "role": "total",
    },
    {
        "id": "sapt2p_adz_exch",
        "source_columns": ("SAPT2+/aDZ Exch",),
        "label": "Exchange",
        "category": "exchange",
        "parent_category": "interaction",
        "family": "sapt",
        "level": "SAPT2+/aDZ",
        "role": "total",
    },
    {
        "id": "sapt2p_adz_ind",
        "source_columns": ("SAPT2+/aDZ Ind",),
        "label": "Induction total",
        "category": "induction",
        "parent_category": "interaction",
        "family": "sapt",
        "level": "SAPT2+/aDZ",
        "role": "total",
    },
    {
        "id": "sapt2p_adz_disp",
        "source_columns": ("SAPT2+/aDZ Disp",),
        "label": "Dispersion",
        "category": "dispersion",
        "parent_category": "interaction",
        "family": "sapt",
        "level": "SAPT2+/aDZ",
        "role": "total",
    },

    # SAPT(DFT) raw interaction decomposition.
    {
        "id": "saptdft_total",
        "source_columns": ("F-Total",),
        "label": "Total",
        "category": "interaction",
        "family": "sapt",
        "level": "SAPT(DFT)",
        "role": "total",
    },
    {
        "id": "saptdft_elst",
        "source_columns": ("F-Electrostatics",),
        "label": "Electrostatics",
        "category": "electrostatics",
        "parent_category": "interaction",
        "family": "sapt",
        "level": "SAPT(DFT)",
        "role": "total",
    },
    {
        "id": "saptdft_exch",
        "source_columns": ("F-Exchange",),
        "label": "Exchange",
        "category": "exchange",
        "parent_category": "interaction",
        "family": "sapt",
        "level": "SAPT(DFT)",
        "role": "total",
    },
    {
        "id": "saptdft_ind",
        "source_columns": ("F-Induction",),
        "label": "Induction total",
        "category": "induction",
        "parent_category": "interaction",
        "family": "sapt",
        "level": "SAPT(DFT)",
        "role": "total",
    },
    {
        "id": "saptdft_disp",
        "source_columns": ("F-Dispersion",),
        "label": "Dispersion",
        "category": "dispersion",
        "parent_category": "interaction",
        "family": "sapt",
        "level": "SAPT(DFT)",
        "role": "total",
    },
    {
        "id": "saptdft_ind20r",
        "source_columns": ("ind20,r_dft",),
        "label": "ind20,r",
        "category": "induction",
        "family": "sapt",
        "level": "SAPT(DFT)",
        "role": "component",
    },
    {
        "id": "saptdft_exch_ind20r",
        "source_columns": ("exch-ind20,r_dft",),
        "label": "exch-ind20,r",
        "category": "induction",
        "family": "sapt",
        "level": "SAPT(DFT)",
        "role": "component",
    },
    {
        "id": "saptdft_dhf",
        "source_columns": ("delta hf correction",),
        "label": "δHF",
        "category": "induction",
        "family": "sapt",
        "level": "SAPT(DFT)",
        "role": "component",
    },

    # Classical undamped Delta-MTP energies. The raw induction is the dimer-multipole
    # electrostatics minus isolated-monomer electrostatics; its half-scaled form is the
    # physically meaningful variational induction estimate.
    {
        "id": "mtp_elst",
        "source_columns": ("mtp elst",),
        "label": "Delta-MTP electrostatics (monomer)",
        "category": "electrostatics",
        "family": "mtp",
        "level": None,
        "role": "model",
    },
    {
        "id": "mtp_elst_dimer",
        "source_columns": ("mtp elst dimer",),
        "label": "Delta-MTP electrostatics (dimer)",
        "category": "electrostatics",
        "family": "mtp",
        "level": None,
        "role": "model",
    },
    {
        "id": "mtp_ind",
        "source_columns": ("mtp ind",),
        "label": "Delta-MTP induction (unscaled)",
        "category": "induction",
        "family": "mtp",
        "level": None,
        "role": "model",
    },
    {
        "id": "mtp_ind_half",
        "source_columns": ("mtp ind / 2",),
        "label": "Delta-MTP induction",
        "category": "induction",
        "family": "mtp",
        "level": None,
        "role": "model",
    },

    # AMOEBA+ polarization under the two valid direct/mutual damping combinations.
    {
        "id": "amoeba_pol_3900_7000",
        "source_columns": ("amoebaplus_pol_3900_7000",),
        "label": "Polarization (0.70 direct, 0.39 mutual)",
        "category": "induction",
        "family": "amoeba",
        "level": None,
        "role": "component",
        "variant": "direct_070_mutual_039",
    },
    {
        "id": "amoeba_pol_3900",
        "source_columns": ("amoebaplus_pol_3900",),
        "label": "Polarization (0.39 direct, 0.39 mutual)",
        "category": "induction",
        "family": "amoeba",
        "level": None,
        "role": "component",
        "variant": "direct_039_mutual_039",
    },

    # Deliberate exclusions from the same S66x8 upload: charge-penetration polarization is
    # invalid, CT is not polarization, mtp is electrostatics, rep/disp are other categories,
    # and HIPPO remains a separate unsupported force-field family.
)

# All accepted input spellings, derived so schema reporting cannot drift from classification.
ENERGY_COLUMNS = frozenset(
    source
    for definition in ENERGY_DEFINITIONS
    for source in definition["source_columns"]
)

# system_id is optional: without it every row is its own single-frame system, so only the
# "trajectory" feature is lost, not the upload.
FEATURE_REQUIREMENTS = {
    "geometry_viewer": {
        "qcel_molecule",
    },

    "trajectory": {
        "system_id",
    },

    "mbis_charges": {
        "q hf/adz A",
        "q hf/adz B",
        "q hf/adz dimer",
    },

    "atomic_dipoles": {
        "mu hf/adz A",
        "mu hf/adz B",
    },

    "quadrupoles": {
        "theta hf/adz A",
        "theta hf/adz B",
    },

    "volume_ratios": {
        "volume ratios A",
        "volume ratios B",
    },

    # The monomer inputs an IPD run reads. Column presence only -- whether apnet_pt is
    # installed is a separate question, answered by capability.capability(). Independent
    # of mbis_charges: that needs the dimer column, this does not.
    "ipd_computable": {
        "q hf/adz A",
        "q hf/adz B",
        "mu hf/adz A",
        "mu hf/adz B",
        "theta hf/adz A",
        "theta hf/adz B",
        "volume ratios A",
        "volume ratios B",
    },

    # Delta-MTP compares isolated-monomer electrostatics with electrostatics from the dimer
    # multipoles. It therefore needs A, B, and dimer q/mu/theta arrays, but no volume ratios.
    "delta_mtp_computable": {
        "q hf/adz A",
        "q hf/adz B",
        "q hf/adz dimer",
        "mu hf/adz A",
        "mu hf/adz B",
        "mu hf/adz dimer",
        "theta hf/adz A",
        "theta hf/adz B",
        "theta hf/adz dimer",
    },
}


DELTA_MTP_OUTPUT_COLUMNS = (
    "mtp elst",
    "mtp elst dimer",
    "mtp ind",
    "mtp ind / 2",
)


# Every column the app understands. A union of the sets above rather than a fourth list, so
# it cannot drift from them. Used to report what an upload carried and what it did not.
KNOWN_COLUMNS = REQUIRED_COLUMNS | MBIS_COLUMNS | ENERGY_COLUMNS | {
    column for columns in FEATURE_REQUIREMENTS.values() for column in columns
}


def canonical_name(name):
    """The name the app uses for a column, given whatever the upload called it."""
    return COLUMN_ALIASES.get(name, name)


def rename_map(columns):
    """A mapping suitable for ``df.rename(columns=...)``; aliases only."""
    return {
        name: canonical_name(name)
        for name in columns
        if canonical_name(name) != name
    }


def classify_energy_definition(definition, columns):
    """Catalog metadata for one definition's first available source, or None.

    A priority tuple belongs to one stable output id, so alternate upload spellings can never
    create duplicate curves. Different definitions may still intentionally select the same
    source when one raw quantity has more than one scientific role.
    """
    columns = {canonical_name(name) for name in columns}
    source = next(
        (name for name in definition["source_columns"] if name in columns),
        None,
    )
    if source is None:
        return None
    return {
        "source_column": source,
        "units": "kcal/mol",
        **{
            key: value
            for key, value in definition.items()
            if key != "source_columns"
        },
    }


def feature_availability(columns):
    """Which features these columns support, reporting every feature true or false.

    A false entry is what lets a tab explain itself instead of vanishing. Says nothing
    about stored IPD history, whose column names come from thole_damping -- the IPD code
    detects that where the question matters.
    """
    columns = {canonical_name(name) for name in columns}

    features = {
        feature: required.issubset(columns)
        for feature, required in FEATURE_REQUIREMENTS.items()
    }

    # Any one level is enough, so this is a plain intersection rather than a subset test.
    features["energy_plot"] = bool(ENERGY_COLUMNS & columns)

    return features
