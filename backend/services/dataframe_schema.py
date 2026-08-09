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

``INDUCTION_ENERGY_COLUMNS`` and :func:`classify_energy_column` are the exception to "names
only" in spirit but not in mechanism: they attach *meaning* to a name -- which family, which
level of theory, total or component -- so that the browser can group and label curves without
parsing column strings. Still no dataframe is opened. Turning that vocabulary into a catalog
and per-frame numbers is ``energies``'s job.
"""

# 2: frames carry a "multipoles" block.
# 3: frames carry an "ipd" block -- per-mode availability and iteration counts.
# 4: systems carry an "energy_catalog", and frames an "energies" block keyed by energy id.
# Bumping this is what makes an already-stored collection reprocess instead of being returned
# untouched -- see upload_system.process_upload.
SCHEMA_VERSION = 4

REQUIRED_COLUMNS = {
    "qcel_molecule",
}

# Spellings seen in real uploads, mapped to the name the rest of the app uses.
# 131_Na-benzene.pkl writes the monomer molecules with a space.
COLUMN_ALIASES = {
    "system id": "system_id",
    "qcel molecule A": "qcel_molecule A",
    "qcel molecule B": "qcel_molecule B",
}

MBIS_COLUMNS = {
    "q hf/adz A",
    "q hf/adz B",
    "q hf/adz dimer",

    # The *permanent* atomic dipole, an input. Not mu_ind, the induced dipole the viewer
    # draws, which is an IPD result and named by radius_thole.
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

# Every energy column the app knows how to plot, keyed by the exact name an upload carries.
#
# Values are the scientific metadata the frontend groups, labels and styles by. The point of
# stating it here is that the browser never has to infer meaning from a column name: it reads
# `family`, `level` and `role` as fields.
#
# `level` is part of a curve's identity, not decoration. The water fixtures carry both SAPT0
# and SAPT0/cc-pVDZ induction, and both label their total "Induction total" -- keyed on the
# term alone the two benchmarks would collapse into one curve, silently, with the second
# level's points appended to the first's.
#
# A level is named after what its columns literally say. The unqualified "SAPT0 IND kcalmol"
# family records no basis set anywhere in the dataframe, so it is called "SAPT0" rather than
# guessed at as aDZ.
#
# `id` is written out rather than derived from the column name: "SAPT ind20,r kcalmol" and
# "SAPT/cc-pVDZ ind20,r kcalmol" would slug to the same thing, and ten literal strings need no
# collision handling.
#
# Recognition is by exact name, with no "sapt" in name.lower() fallback. A substring rule would
# sweep in "SAPT0 ELST kcalmol" and "SAPT0 TOTAL kcalmol" -- electrostatics, and a total
# interaction energy -- both of which are simply wrong curves on an induction axis. An
# unrecognised column returning None is the safe failure; a misrecognised one is a plausible
# lie.
#
# IPD results are deliberately absent. Their names are thole_damping's to construct, and
# `energies` builds those catalog entries from ipd_results.MODES instead.
INDUCTION_ENERGY_COLUMNS = {
    # SAPT0. Total plus the three terms decomposing it.
    "SAPT0 IND kcalmol": {
        "id": "sapt0_ind",
        "label": "Induction total",
        "family": "sapt",
        "level": "SAPT0",
        "role": "total",
    },
    "SAPT ind20,r kcalmol": {
        "id": "sapt0_ind20r",
        "label": "ind20,r",
        "family": "sapt",
        "level": "SAPT0",
        "role": "component",
    },
    "SAPT exch-ind20,r kcalmol": {
        "id": "sapt0_exch_ind20r",
        "label": "exch-ind20,r",
        "family": "sapt",
        "level": "SAPT0",
        "role": "component",
    },
    "SAPT dHF ind kcalmol": {
        "id": "sapt0_dhf",
        "label": "δHF",
        "family": "sapt",
        "level": "SAPT0",
        "role": "component",
    },

    # SAPT0/cc-pVDZ. The same four quantities at a second level of theory.
    "SAPT0/cc-pVDZ IND kcalmol": {
        "id": "sapt0_ccpvdz_ind",
        "label": "Induction total",
        "family": "sapt",
        "level": "SAPT0/cc-pVDZ",
        "role": "total",
    },
    "SAPT/cc-pVDZ ind20,r kcalmol": {
        "id": "sapt0_ccpvdz_ind20r",
        "label": "ind20,r",
        "family": "sapt",
        "level": "SAPT0/cc-pVDZ",
        "role": "component",
    },
    "SAPT/cc-pVDZ exch-ind20,r kcalmol": {
        "id": "sapt0_ccpvdz_exch_ind20r",
        "label": "exch-ind20,r",
        "family": "sapt",
        "level": "SAPT0/cc-pVDZ",
        "role": "component",
    },
    "SAPT/cc-pVDZ dHF ind kcalmol": {
        "id": "sapt0_ccpvdz_dhf",
        "label": "δHF",
        "family": "sapt",
        "level": "SAPT0/cc-pVDZ",
        "role": "component",
    },

    # SAPT2+/aDZ, in 131_Na-benzene.pkl. A total with no breakdown alongside it, which is a
    # complete benchmark rather than a degraded one -- so it is a `total` and draws solid.
    "SAPT2+/aDZ Ind": {
        "id": "sapt2p_adz_ind",
        "label": "Induction total",
        "family": "sapt",
        "level": "SAPT2+/aDZ",
        "role": "total",
    },

    # Classical undamped multipole induction. A model, not a benchmark, and it belongs to no
    # level of theory -- hence `level: None`, which is what keeps it out of the SAPT headings.
    "mtp ind": {
        "id": "mtp_ind",
        "label": "Multipole induction",
        "family": "mtp",
        "level": None,
        "role": "model",
    },
    "mtp ind / 2": {
        "id": "mtp_ind_half",
        "label": "Multipole induction / 2",
        "family": "mtp",
        "level": None,
        "role": "model",
    },
}

# Derived from the map rather than listed again, so the two cannot drift. Used by
# feature_availability and KNOWN_COLUMNS; a level is available when any of these is present.
ENERGY_COLUMNS = frozenset(INDUCTION_ENERGY_COLUMNS)

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
}


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


def classify_energy_column(name):
    """The scientific metadata for a recognised energy column, or None for anything else.

    None means "not an energy this app plots", which covers the overwhelming majority of
    columns -- including energies it deliberately does not plot yet, such as electrostatics
    and total interaction energies.

    ``category`` is carried on every entry even though only one value exists today. It is what
    lets a later electrostatics tab filter the same catalog without a second metadata scheme.
    """
    name = canonical_name(name)
    entry = INDUCTION_ENERGY_COLUMNS.get(name)
    if entry is None:
        return None
    return {
        "source_column": name,
        "category": "induction",
        "units": "kcal/mol",
        **entry,
    }


def feature_availability(columns):
    """Which features these columns support, reporting every feature true or false.

    A false entry is what lets a tab explain itself instead of vanishing. Says nothing
    about stored IPD history, whose column names come from radius_thole -- the IPD code
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
