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

IPD result columns are not listed here either. ``radius_thole.ipd_column_names`` is the
source of truth for those; code that needs to detect a stored result asks it directly.
"""

# 2: frames carry a "multipoles" block. Bumping this is what makes an already-stored
# collection reprocess instead of being returned untouched -- see upload_system.process_upload.
SCHEMA_VERSION = 2

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

# Induction totals only. A level is available when its total is present; the optional
# breakdown terms alongside it are not a gate.
ENERGY_COLUMNS = {
    "SAPT2+/aDZ Ind",
    "SAPT0 IND kcalmol",
    "SAPT0/cc-pVDZ IND kcalmol",
}

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
    # installed is a separate question, answered by ipd_compute.capability(). Independent
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
