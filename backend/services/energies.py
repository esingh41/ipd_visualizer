"""Plottable energies: what each one means, once, and its value per frame.

The fourth responsibility alongside the three ``ipd_results`` names::

    dataframe_schema  -> uploaded energy column vocabulary and its meaning
    thole_damping     -> IPD result column names
    ipd_results       -> the damping modes the UI offers
    this module       -> one catalog + per-frame values, keyed by energy id

Pure translation, like ``ipd_results``: it reads column names and dataframe rows and returns
dictionaries. It opens no files, imports no Flask and runs no science, which is what lets
``system_serialization`` call it while saving a collection without a cycle back.

**Metadata once, numbers per frame.** The catalog is a list of what every plottable energy *is*
-- family, level of theory, total or component -- written into a system's payload a single time.
Each frame then carries only ``{energy_id: value}``. Repeating the metadata on all fourteen
frames of a scan would multiply it by the frame count for no information gained, and would let
two frames disagree about what the same curve means.

The point of the split is that the browser never decodes a column name. It receives
``family="sapt"``, ``level="SAPT0"``, ``role="component"`` as fields and groups on them; the one
thing it still owns is how a curve *looks*.

Two sources feed the catalog, kept apart on purpose:

* **Uploaded columns**, selected from ordered ``dataframe_schema.ENERGY_DEFINITIONS``.
* **IPD results**, which have no fixed column name to recognise -- ``thole_damping`` constructs
  it from the parameterization. So those entries are built from ``ipd_results.MODES``, asking
  ``thole_damping`` for the name, and carry the parameterization as fields rather than as text
  the frontend would have to parse back out.

Both are **detected, never declared**: an energy reaches the catalog only when its column is
actually present. A dataset with one benchmark and no IPD is a complete catalog of one entry,
not a degraded one.

Only IPD results this app computed become curves. Uploads may carry induction energies under
``radius_thole``'s older names -- ``IPD (MBIS) 0.39 all`` and friends -- and those are ignored,
for the reason ``ipd_results`` states: a displayed result should be traceable to the choices
that produced it, and those names do not say what they were.
"""

from __future__ import annotations

import numpy as np

from backend.services import dataframe_schema, ipd_results, thole_damping


def _finite(value):
    """A float, or None when the cell holds anything that is not a finite number.

    Absent, NaN, and a string someone put in a numeric column all collapse to None, which is
    the single "no value here" the frontend has to handle -- and which ``json.dump`` will
    accept, unlike a bare NaN.
    """
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _ipd_entries(columns):
    """A catalog entry per damping mode whose energy column is present.

    The column name comes from ``thole_damping``, never from matching an "IPD (...)" string:
    that module constructs these names and is the only thing that should know their shape.

    The parameterization is splatted in as fields -- ``inter_thole_parameterization`` and
    ``intra_thole_parameterization`` -- so a frontend that wants to say "radius inter, 0.39
    intra" can, without decoding the label or the column.
    """
    entries = []
    for mode_id, mode in ipd_results.MODES.items():
        parameterization = mode["parameterization"]
        column = thole_damping.ipd_column_names(**parameterization)["energy"]
        if column not in columns:
            continue
        entries.append(
            {
                # The same id the compute endpoints use, so a curve and the Trajectory tab's
                # damping-mode dropdown name one thing rather than two.
                "id": f"ipd_{mode_id}",
                "source_column": column,
                "label": mode["label"],
                "category": "induction",
                "family": "ipd",
                # An IPD result is this app's own prediction, not a benchmark at some level of
                # theory. None is what keeps it out of the SAPT level headings.
                "level": None,
                "role": "model",
                "units": "kcal/mol",
                **parameterization,
            }
        )
    return entries


def energy_catalog(columns):
    """Every energy these columns can plot, in display order.

    Takes column names rather than a dataframe: what a dataset *can* plot is a question about
    its vocabulary, and answering it should not require loading rows.

    Order is meaningful and is not the dataframe's. ``ENERGY_DEFINITIONS`` declares each
    level's display order and chooses at most one accepted source per stable id. Then families
    are laid out reference-first: SAPT benchmarks, this app's IPD predictions, and anything
    else.
    """
    columns = {dataframe_schema.canonical_name(name) for name in columns}

    uploaded = [
        entry
        for definition in dataframe_schema.ENERGY_DEFINITIONS
        if (
            entry := dataframe_schema.classify_energy_definition(definition, columns)
        )
        is not None
    ]

    return (
        [entry for entry in uploaded if entry["family"] == "sapt"]
        + _ipd_entries(columns)
        + [entry for entry in uploaded if entry["family"] != "sapt"]
    )


def frame_energies(row, catalog):
    """``{energy_id: float or None}`` for one row -- numbers only, no metadata repeated.

    Every catalog id appears, with None where this row has no usable value. Present-and-null
    rather than absent because the two mean different things to a plot: an id missing from the
    catalog has no curve at all, while a null at one frame is a gap in a curve that exists.
    A partly computed IPD mode is exactly that, and drawing it as a gap rather than joining
    across the missing frames is what keeps it from reading as a finished calculation.
    """
    return {
        entry["id"]: _finite(row.get(entry["source_column"])) for entry in catalog
    }
