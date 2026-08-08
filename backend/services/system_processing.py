"""Derived quantities: geometry fields, contact distance, equilibrium ratio, trajectories.

Answers "what useful values can we derive from an upload?". Everything here is computed
from what a frame already carries -- nothing is read back from a column that claims to
state it, and nothing an uploader supplied is overwritten.

Self-contained -- numpy and pandas only. It does not know about IPD results, apnet_pt, or
how any of this is persisted.

Two things are load-bearing and easy to break by accident:

* ``n_atoms_A`` is the *only* encoding of monomer membership -- atom ``i`` is in A iff
  ``i < n_atoms_A``. **This module assumes validation has already guaranteed that the two
  fragments are contiguous and ordered monomer-A-then-B**; nothing here re-checks it, and
  fragments like ``[[0, 2, 4], [1, 3, 5]]`` would pass silently and mean something else.
  The check belongs in ``dataframe_validation``.
* A trajectory's separation source is chosen **once for the whole group**, never per row.
  A group mixing an Re ratio with an Angstrom distance sorts into nonsense.

``add_derived_columns`` must run before ``group_trajectories``: grouping reads the
``eq_ratio`` and ``contact_distance_ang`` columns it writes rather than recomputing them.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

BOHR2ANG = 0.52917720859

CONTACT_COLUMN = "contact_distance_ang"
EQ_RATIO_COLUMN = "eq_ratio"

# The trailing token of "01_Na-Water_1.05" or "131_Na-benzene_0.70", read as a multiple of
# the equilibrium separation. One pattern serves both the group name and the separation, so
# the two can never disagree about where a group name ends.
SEPARATION_TOKEN = re.compile(r"[_-](\d+(?:\.\d+)?)$")


# --- geometry ---------------------------------------------------------------


def fragments(molecule):
    """The monomer index arrays of a dimer, or None if the molecule has none."""
    found = getattr(molecule, "fragments", None)
    if found is None:
        return None
    return [np.asarray(fragment) for fragment in found]


def geometry_fields(molecule):
    """Symbols, coordinates in Angstrom, atom counts and an xyz string, or None.

    ``n_atoms_A`` is taken as the size of the first fragment, which is only meaningful
    because validation has already guaranteed contiguous monomer-A-then-B ordering. Nothing
    here re-checks it.

    Never raises: this feeds listings, and one odd molecule must not take a whole dataset
    down with it.
    """
    parts = fragments(molecule)
    if parts is None or len(parts) != 2:
        return None
    try:
        # qcelemental keeps geometry in bohr; everything downstream reports Angstrom.
        coords = np.asarray(molecule.geometry, dtype=float) * BOHR2ANG
        symbols = [str(symbol) for symbol in molecule.symbols]
    except (AttributeError, TypeError, ValueError):
        return None
    if coords.ndim != 2 or coords.shape[1] != 3 or len(symbols) != len(coords):
        return None

    n_atoms_a = len(parts[0])
    return {
        "symbols": symbols,
        "coords": coords,
        "n_atoms": len(coords),
        "n_atoms_A": n_atoms_a,
        "xyz": build_xyz(symbols, coords),
    }


def build_xyz(symbols, coords, comment=""):
    """Format an XYZ string for ``viewer.addModel(xyz, "xyz")``."""
    lines = [str(len(symbols)), comment or "ipd-visualizer"]
    for symbol, (x, y, z) in zip(symbols, coords):
        lines.append(f"{symbol} {x:.8f} {y:.8f} {z:.8f}")
    return "\n".join(lines) + "\n"


def contact_distance(coords, n_atoms_a):
    """Shortest distance between any atom of A and any atom of B, in Angstrom, or None.

    The one separation measure that means the same thing for every dimer. All atoms count,
    hydrogens included: for a hydrogen-bonded dimer the H...O contact *is* the interaction,
    and skipping it would report the much longer heavy-atom separation.

    ``coords`` must already be Angstrom and ordered monomer-A-then-B.
    """
    coords = np.asarray(coords, dtype=float)
    n_atoms_a = int(n_atoms_a)
    if coords.ndim != 2 or coords.shape[1] != 3:
        return None
    if not 0 < n_atoms_a < len(coords):
        return None
    if not np.isfinite(coords).all():
        return None

    # (n_a, n_b) pairwise distances by broadcasting. These are dimers -- a few dozen atoms at
    # most -- so the full matrix is cheaper than anything cleverer.
    deltas = coords[:n_atoms_a, None, :] - coords[None, n_atoms_a:, :]
    return float(np.linalg.norm(deltas, axis=-1).min())


def contact_distance_for(molecule):
    """Closest A-to-B contact for a row's molecule, in Angstrom, or None."""
    if molecule is None:
        return None
    geometry = geometry_fields(molecule)
    if geometry is None:
        return None
    return contact_distance(geometry["coords"], geometry["n_atoms_A"])


# --- separation -------------------------------------------------------------


def separation_from_system_id(system_id):
    """The separation encoded in a system_id's trailing token, if it has one."""
    match = SEPARATION_TOKEN.search(str(system_id))
    return float(match.group(1)) if match else None


def system_group(system_id):
    """The trajectory a system_id belongs to: the id with its separation token stripped.

    A dataset not following the convention loses nothing -- nothing is stripped, and every
    group holds one geometry.
    """
    stripped = SEPARATION_TOKEN.sub("", str(system_id))
    return stripped or str(system_id)


def _finite_float(row, column):
    """``row[column]`` as a finite float, or None if absent or unusable."""
    try:
        value = float(row[column])
    except (TypeError, ValueError, KeyError, IndexError):
        return None
    return value if np.isfinite(value) else None


def _as_ratios(values):
    """Every value formatted as a multiple of the equilibrium separation."""
    return [
        {
            "separation": value,
            "separation_units": "Re",
            "separation_label": f"{value:.2f} Re",
        }
        for value in values
    ]


def derive_separations(frames):
    """Give every frame in **one trajectory** a separation, its units and its label.

    The source is chosen once for the whole trajectory, in this order of preference, and a
    source qualifies only if it describes *every* frame. A group that mixed an Re ratio for
    one frame with an Angstrom distance for the next would sort into nonsense and animate in
    the wrong order.

        1. eq_ratio column          -> Re
        2. system_id trailing token -> Re
        3. contact_distance_ang     -> Angstrom
        4. nothing                  -> unavailable, labelled by system_id

    The first two are dimensionless multiples of equilibrium; the third is a real distance
    and is stamped as one. Reads the columns ``add_derived_columns`` writes -- it must have
    run first.
    """
    frames = list(frames)

    ratios = [_finite_float(frame["_row"], EQ_RATIO_COLUMN) for frame in frames]
    if frames and all(value is not None for value in ratios):
        return _as_ratios(ratios)

    ratios = [separation_from_system_id(frame["system_id"]) for frame in frames]
    if frames and all(value is not None for value in ratios):
        return _as_ratios(ratios)

    # Available for any dimer whose geometry parses, so a dataframe declaring nothing at all
    # still gets an ordered, honestly-labelled scan.
    contacts = [_finite_float(frame["_row"], CONTACT_COLUMN) for frame in frames]
    if frames and all(value is not None for value in contacts):
        return [
            {
                "separation": value,
                "separation_units": "angstrom",
                "separation_label": f"{value:.3f} Å",
            }
            for value in contacts
        ]

    return [
        {
            "separation": None,
            "separation_units": None,
            "separation_label": frame["system_id"],
        }
        for frame in frames
    ]


# --- trajectories -----------------------------------------------------------


def _system_id_of(row, position):
    """A row's system_id, or "" when it has none.

    Blank is treated exactly like absent. A frame that labels some rows and not others
    should not collect the unlabelled ones into a single nameless trajectory.
    """
    value = row.get("system_id")
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def group_trajectories(df):
    """Rows collected into trajectories, ordered by separation and frame-indexed.

    A row with no usable system_id becomes its own single-frame trajectory named
    ``row <position>`` -- absent grouping means one system per row, not one system holding
    everything.

    **Run ``add_derived_columns`` first.** Separations are read off the columns it writes,
    so grouping an unprocessed frame leaves every separation unavailable: frames keep their
    dataframe order and label themselves by system_id.

    Returns ``[{"name", "frames"}]`` sorted by name, each frame carrying ``row_index``,
    ``system_id``, ``frame_index``, ``separation``, ``separation_units`` and
    ``separation_label``.
    """
    groups = {}
    for position, (_, row) in enumerate(df.iterrows()):
        system_id = _system_id_of(row, position)
        name = system_group(system_id) if system_id else f"row {position}"
        groups.setdefault(name, []).append(
            {
                "row_index": position,
                "system_id": system_id,
                # Kept only long enough to derive the separation below.
                "_row": row,
            }
        )

    trajectories = []
    for name in sorted(groups):
        frames = groups[name]
        derived = derive_separations(frames)
        for frame, values in zip(frames, derived):
            frame.pop("_row")
            frame.update(values)

        # A dataframe's rows arrive in build order -- s66x8 interleaves all 66 trajectories --
        # so a scan only reads as one once sorted. Rows with no separation keep their
        # dataframe position and fall after those that have one.
        frames.sort(
            key=lambda frame: (
                frame["separation"] is None,
                frame["separation"] if frame["separation"] is not None else 0.0,
                frame["row_index"],
            )
        )
        # Assigned after sorting: this is a position in the animation, not in the dataframe.
        for index, frame in enumerate(frames):
            frame["frame_index"] = index

        trajectories.append({"name": name, "frames": frames})

    return trajectories


# --- derived columns --------------------------------------------------------


def add_derived_columns(df):
    """Write the derivable metadata columns in place; return the names written.

    The two columns are owned differently, on purpose:

    * ``contact_distance_ang`` belongs to the application. It is recomputed from
      ``qcel_molecule`` every time, overwriting whatever was there -- an uploaded column of
      that name is not authoritative. Uploads that carry their own contact distance spell it
      differently anyway ("closest contact ang"), which is why this name was chosen.
    * ``eq_ratio`` belongs to the uploader. It is filled only when absent; if they supplied
      one, theirs wins. The point is to fill gaps, not to correct their science.
    """
    written = [CONTACT_COLUMN]

    df[CONTACT_COLUMN] = [
        contact_distance_for(row.get("qcel_molecule")) for _, row in df.iterrows()
    ]

    if EQ_RATIO_COLUMN not in df.columns:
        ratios = [
            separation_from_system_id(_system_id_of(row, position))
            for position, (_, row) in enumerate(df.iterrows())
        ]
        # Written only where a ratio is genuinely recoverable. Where only a contact distance
        # exists the cell stays NaN, and where *no* row yields one the column is never created
        # at all -- an Angstrom distance is not a ratio, and writing it as one would be a lie
        # every later reader believes.
        if any(ratio is not None for ratio in ratios):
            df[EQ_RATIO_COLUMN] = [
                np.nan if ratio is None else ratio for ratio in ratios
            ]
            written.append(EQ_RATIO_COLUMN)

    return written
