// Potential energy curves: induction energy against dimer separation.
//
// The whole tab is a re-slicing of /api/ipd/systems. That endpoint is oriented by
// geometry -- one entry per (system, separation), each carrying every energy at that
// separation. A curve is the transpose: one model or reference term, read across all
// separations. Hence no endpoint of its own and no backend work; see buildSeries.

import { callJson } from "/main.js";

const chartEl = document.getElementById("plotChart");
const datasetEl = document.getElementById("plotDatasetSelect");
const systemEl = document.getElementById("plotSystemSelect");
const curveListEl = document.getElementById("plotCurveList");
const statusEl = document.getElementById("plotStatus");

// null means "not fetched yet", which is what makes the fetch-once check below work. An
// empty array would be indistinguishable from a server that legitimately has no systems,
// and the tab would refetch on every activation forever.
let catalog = null;
let series = [];
let currentAxis = null;
let selected = new Set();
let drawn = false;

// --- styling --------------------------------------------------------------

// Two independent channels, so the plot survives being read quickly: colour says which
// energy component, dash says total vs breakdown. Both resolve to custom properties in
// styles.css, so the sidebar swatches and the lines cannot drift apart.
const IPD_COLORS = ["--ipd-1", "--ipd-2", "--ipd-3", "--ipd-4"];

// Display names for the IPD models, in the order they should be listed.
//
// The catalog's own `label` is the source DataFrame's column name minus the monomer suffix
// -- "mu_ind hist (MBIS) 0.39 all" -- which names a *dipole history*. That is right on the
// Visualize tab, which animates exactly that, and wrong here, where the curve is the
// energy. radius_thole.ipd_column_names already draws the same distinction: it writes
// "IPD (X)" for the energy and derives "mu_ind hist A (X)" from it.
//
// "universal" is not a rewording of "MBIS 0.39 all": radius=None restores a constant Thole
// parameter of 0.39 on every edge, where the other three derive it from TS/MBIS vdW radii
// and differ only in which edges are damped (radius_thole.py:85-94, :342-346).
//
// Keyed on slug rather than label, so uploaded datasets are covered too -- they reuse
// these same four slugs via ipd_compute.MODEL_FOR_RADIUS.
const IPD_MODELS = [
  ["mbis_0_39_all", "IPD (universal 0.39)"],
  ["radius_thole_ts_mbis_intra_inter", "IPD (radius, intra + inter)"],
  ["radius_thole_ts_mbis_intra_only", "IPD (radius, intra)"],
  ["radius_thole_ts_mbis_inter_only", "IPD (radius, inter)"],
];

// Keyed by the reference label the catalog uses. Only the induction family appears in the
// bundled data; the others are here so an S66x8 import needs no changes.
const SAPT_STYLE = {
  "Induction total": { color: "--sapt-ind", dash: "solid" },
  "ind20,r": { color: "--sapt-ind", dash: "dash" },
  "exch-ind20,r": { color: "--sapt-ind", dash: "dot" },
  "δHF": { color: "--sapt-ind", dash: "dashdot" },
};

// Fallback for labels the map above has not seen, matched on the conventional SAPT
// component names so a new decomposition term lands on the right colour unassisted.
const SAPT_FALLBACK = [
  [/elst|electrostat/i, "--sapt-elst"],
  [/exch/i, "--sapt-exch"],
  [/disp/i, "--sapt-disp"],
  [/ind/i, "--sapt-ind"],
];

// Plotly needs a literal colour string; it cannot resolve var(--x) itself.
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function styleFor(entry) {
  if (entry.group === "ipd") {
    // Keyed on the series' own rank rather than its position in whatever array the caller
    // happens to be walking. Position varies by dataset -- the bundled catalog lists
    // intra_inter first, an upload lists mbis_0_39_all first -- so a positional colour
    // would change under a model when you switched systems, which is exactly when you are
    // trying to compare them.
    //
    // Solid throughout: these are this app's own predictions, and a prediction is always
    // a total. Dashes stay reserved for decomposing a reference into its terms.
    return { color: cssVar(IPD_COLORS[entry.rank % IPD_COLORS.length]), dash: "solid" };
  }

  const known = SAPT_STYLE[entry.label];
  if (known) {
    return { color: cssVar(known.color), dash: known.dash };
  }
  const matched = SAPT_FALLBACK.find(([pattern]) => pattern.test(entry.label));
  return { color: cssVar(matched ? matched[1] : "--muted"), dash: "dash" };
}

// --- data -----------------------------------------------------------------

function unique(values) {
  return [...new Set(values)];
}

// Local rather than imported from main.js: the codebase keeps helpers this small private
// per module (there is a matching copy at ipd_compute.js:62) so modules stay independent.
function fillSelect(selectEl, options, preferred) {
  selectEl.replaceChildren();
  for (const value of options) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    selectEl.append(option);
  }
  selectEl.value = options.includes(preferred) ? preferred : (options[0] ?? "");
  // A one-option selector is a label wearing a dropdown's clothes. Today the dataset
  // selector is exactly that; it stops being disabled the moment a second one exists.
  selectEl.disabled = options.length < 2;
  return selectEl.value;
}

// The two ways a system's geometries can be laid along x. Angstrom is preferred; Re is the
// fallback for data with no contact distance. Compared by identity, so these are the only
// two instances that ever exist.
const X_CONTACT = {
  title: "Closest contact (Å)",
  value: (row) => row.contact_distance_ang,
  format: ".3f",
  suffix: " Å",
  // The ratio to the equilibrium geometry, carried per point so it can be read off the
  // hover without giving up an Angstrom axis.
  extra: (row) => (row.separation != null ? `${row.separation.toFixed(2)} Rₑ` : ""),
};

const X_SEPARATION = {
  title: "Separation (Rₑ)",
  value: (row) => row.separation,
  format: ".2f",
  suffix: "",
  extra: () => "",
};

// Pure: catalog in, plot-ready series out. Nothing here touches the DOM or module state,
// which makes it the one part of this file checkable by pasting it into a console.
function buildSeries(entries, datasetName, systemName) {
  const rows = entries
    .filter((e) => e.dataset === datasetName && e.system_name === systemName)
    // Sorted on separation rather than on the x value: it is the field every entry is
    // guaranteed to have, and the two order a system identically. Plotly draws points in
    // array order, so an unsorted x doubles the line back on itself -- no error, just a
    // wrong chart.
    .sort((a, b) => a.separation - b.separation);

  // Angstrom is the better axis -- it is a real distance rather than a ratio, so curves
  // from different systems are on the same footing. But it is decided once per system,
  // never per point: an axis that silently switched units partway along would be worse
  // than one in the units the whole system shares.
  const axis = rows.every((row) => typeof row.contact_distance_ang === "number")
    ? X_CONTACT
    : X_SEPARATION;

  const byId = new Map();

  const push = (id, label, group, rank, row, y) => {
    // Skips the `model` and `ref` slugs without naming them: they carry a null energy
    // because they are dipole snapshots, not energies. Any future slug that does the
    // same is handled the same way.
    if (y === null || y === undefined) {
      return;
    }
    if (!byId.has(id)) {
      byId.set(id, { id, label, group, rank, x: [], y: [], customdata: [] });
    }
    const entry = byId.get(id);
    entry.x.push(axis.value(row));
    entry.y.push(y);
    entry.customdata.push(axis.extra(row));
  };

  // Slugs IPD_MODELS does not name still get a series and a rank -- the curve list is
  // deliberately data-driven, so a damping mode added later has to appear on its own. It
  // just appears after the known four, under whatever name the catalog gave it.
  let unranked = 0;

  for (const row of rows) {
    for (const model of row.models ?? []) {
      const known = IPD_MODELS.findIndex(([slug]) => slug === model.slug);
      const seen = byId.has(model.slug);
      push(
        model.slug,
        known === -1 ? model.label || model.slug : IPD_MODELS[known][1],
        "ipd",
        known === -1 ? IPD_MODELS.length + unranked : known,
        row,
        model.energy_kcalmol,
      );
      // Counted only on the row that actually created the series, so a model appearing at
      // every separation does not advance the counter fourteen times.
      if (known === -1 && !seen && byId.has(model.slug)) {
        unranked += 1;
      }
    }
    for (const ref of row.reference_energies ?? []) {
      // Prefixed so a reference label can never collide with a model slug. No rank: SAPT
      // styling keys off the label, and the catalog's order already carries meaning.
      push(`sapt:${ref.label}`, ref.label, "sapt", null, row, ref.kcalmol);
    }
  }

  // IPD models first and in IPD_MODELS order, so the list reads the same whichever dataset
  // is selected -- the bundled catalog and an upload emit their models in different orders.
  // Array#sort is stable, so returning 0 leaves SAPT in catalog order, which is meaningful
  // there: the total comes first, then the three terms that sum to it.
  const ordered = [...byId.values()].sort((a, b) => {
    if (a.group !== b.group) {
      return a.group === "ipd" ? -1 : 1;
    }
    return a.group === "ipd" ? a.rank - b.rank : 0;
  });

  return { series: ordered, axis };
}

// --- curve selector -------------------------------------------------------

const GROUP_LABELS = { ipd: "IPD models", sapt: "SAPT reference" };

function buildCurveList() {
  curveListEl.replaceChildren();

  const legend = document.createElement("legend");
  legend.textContent = "Curves";
  curveListEl.append(legend);

  for (const group of ["ipd", "sapt"]) {
    const members = series.filter((entry) => entry.group === group);
    if (members.length === 0) {
      continue;
    }

    const heading = document.createElement("p");
    heading.className = "plot-curve-group";
    heading.textContent = GROUP_LABELS[group];
    curveListEl.append(heading);

    for (const entry of members) {
      const style = styleFor(entry);

      const row = document.createElement("label");
      row.className = "plot-curve";

      const box = document.createElement("input");
      box.type = "checkbox";
      box.value = entry.id;
      box.checked = selected.has(entry.id);
      box.addEventListener("change", () => {
        if (box.checked) {
          selected.add(entry.id);
        } else {
          selected.delete(entry.id);
        }
        draw();
      });

      const swatch = document.createElement("span");
      swatch.className = "plot-curve-swatch";
      swatch.style.color = style.color;
      if (style.dash !== "solid") {
        swatch.dataset.dash = style.dash;
      }

      const text = document.createElement("span");
      text.textContent = entry.label;

      row.append(box, swatch, text);
      curveListEl.append(row);
    }
  }
}

// --- drawing --------------------------------------------------------------

function isLaidOut() {
  // A chart built inside a hidden panel measures 0x0 and stays collapsed after the panel
  // is shown, so every draw and resize checks first. Same hazard the 3Dmol viewer has;
  // see the matching guard at main.js:653.
  return (
    Boolean(chartEl) &&
    !chartEl.closest("[hidden]") &&
    chartEl.clientWidth > 0 &&
    chartEl.clientHeight > 0
  );
}

function draw() {
  if (!isLaidOut()) {
    return;
  }

  const xAxis = currentAxis ?? X_SEPARATION;
  // Only worth printing when the axis is in Angstrom; on an Rₑ axis the ratio is the x
  // value already showing in the hover header.
  const showRatio = xAxis === X_CONTACT;

  const traces = [];
  for (const entry of series) {
    const style = styleFor(entry);
    if (!selected.has(entry.id)) {
      continue;
    }
    traces.push({
      type: "scatter",
      mode: "lines+markers",
      name: entry.label,
      x: entry.x,
      y: entry.y,
      customdata: entry.customdata,
      // In unified mode Plotly already prints the trace name as the row label, so the
      // template supplies only the value. <extra> would otherwise repeat the name in a
      // second box alongside.
      hovertemplate: showRatio
        ? "%{y:.3f} kcal/mol · %{customdata}<extra></extra>"
        : "%{y:.3f} kcal/mol<extra></extra>",
      line: { color: style.color, width: 2, dash: style.dash },
      marker: { color: style.color, size: 6 },
    });
  }

  const layout = {
    margin: { l: 66, r: 16, t: 12, b: 48 },
    font: { family: '"IBM Plex Sans", "Segoe UI", sans-serif', color: cssVar("--ink") },
    // Transparent so the .panel card provides the background; a white plot on an
    // off-white card reads as a misaligned rectangle.
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: {
      title: { text: xAxis.title },
      gridcolor: cssVar("--line"),
      zeroline: false,
      hoverformat: xAxis.format,
      ticksuffix: xAxis.suffix,
    },
    yaxis: {
      title: { text: "Induction energy (kcal/mol)" },
      gridcolor: cssVar("--line"),
      // The sign of an interaction energy is the whole story, so the axis has to show
      // where zero is.
      zeroline: true,
      zerolinecolor: cssVar("--muted"),
    },
    // One hover box listing every curve at that separation -- the comparison this tab
    // exists for, without chasing individual points.
    hovermode: "x unified",
    legend: { orientation: "h", y: -0.18 },
    showlegend: traces.length > 0,
  };

  const config = {
    responsive: true,
    displaylogo: false,
    // Wheel zoom, on top of Plotly's built-in drag-to-pan and box-zoom; double-click
    // resets. That is the entire zoom and pan requirement, for one config flag.
    scrollZoom: true,
    toImageButtonOptions: { filename: "ipd-curves", scale: 2 },
  };

  // react() rather than newPlot(): it diffs against what is already rendered, so toggling
  // a curve keeps the zoom the user set instead of snapping back to the full range.
  Plotly.react(chartEl, traces, layout, config);
  drawn = true;

  // "Nothing to show" and "nothing ticked" are different problems with different fixes,
  // so they get different sentences. The synthetic demo system has a single geometry and
  // no energies at all, which is the case that makes this worth distinguishing.
  if (series.length === 0) {
    statusEl.className = "status";
    statusEl.textContent = "This system has no energy curves.";
  } else {
    statusEl.className = "status";
    statusEl.textContent = traces.length
      ? `Showing ${traces.length} of ${series.length} curves.`
      : "No curves selected.";
  }
}

// --- selector cascade -----------------------------------------------------

function rebuildSeries() {
  ({ series, axis: currentAxis } = buildSeries(catalog, datasetEl.value, systemEl.value));

  // Keep whatever was checked if those curves still exist, so switching system compares
  // like with like. Falls back to the IPD models -- the tab is called Plot IPD.
  const available = new Set(series.map((entry) => entry.id));
  const kept = [...selected].filter((id) => available.has(id));
  selected = new Set(
    kept.length ? kept : series.filter((e) => e.group === "ipd").map((e) => e.id),
  );

  buildCurveList();
  draw();
}

function populateSystems() {
  const names = unique(
    catalog.filter((e) => e.dataset === datasetEl.value).map((e) => e.system_name),
  );
  fillSelect(systemEl, names, systemEl.value);
  rebuildSeries();
}

function populateDatasets() {
  fillSelect(datasetEl, unique(catalog.map((e) => e.dataset)), datasetEl.value);
  populateSystems();
}

datasetEl.addEventListener("change", populateSystems);
systemEl.addEventListener("change", rebuildSeries);

// --- boot -----------------------------------------------------------------

async function loadCatalog() {
  statusEl.className = "status";
  statusEl.textContent = "Loading curves…";
  try {
    catalog = await callJson("/api/ipd/systems");
    populateDatasets();
  } catch (err) {
    // callJson throws on a failed request. Uncaught, that is an unhandled rejection
    // visible only in devtools, and the user gets a blank panel with no explanation.
    // Left null so the next activation retries rather than sitting on the failure.
    catalog = null;
    statusEl.className = "status error";
    statusEl.textContent = err.message;
  }
}

document.addEventListener("app:tabchange", (event) => {
  if (event.detail.name !== "plot") {
    return;
  }
  // Fetched on first activation rather than at load: the Visualize tab already pulls this
  // same endpoint at boot, and this tab may never be opened at all.
  if (catalog === null) {
    loadCatalog();
    return;
  }
  // Layout has not settled at the instant the event fires, so measuring now reads the
  // geometry from before the panel was shown.
  requestAnimationFrame(() => {
    if (!isLaidOut()) {
      return;
    }
    if (drawn) {
      Plotly.Plots.resize(chartEl);
    } else {
      draw();
    }
  });
});

window.addEventListener("resize", () => {
  if (drawn && isLaidOut()) {
    Plotly.Plots.resize(chartEl);
  }
});
