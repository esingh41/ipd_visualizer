// Energy curves against dimer separation. Today: induction.
//
// The whole tab is a re-slicing of the trajectory payload viewer.js already loaded. That
// payload is oriented by geometry -- one frame per separation, each carrying every energy at
// that separation. A curve is the transpose: one energy read across all frames. Hence no
// endpoint of its own and no second fetch; see buildSeries.
//
// **Nothing here knows what a dataframe column is called.** The payload's `energy_catalog`
// states what each energy *is* -- category, family, level of theory, total or component -- and
// every frame carries `energies[id]`. This file answers three questions and no others: which
// curves exist, which did the user pick, and how should they be drawn. Adding a damping mode
// or a level of theory is a backend change; it appears here on its own.

const chartEl = document.getElementById("plot-chart");
const curveListEl = document.getElementById("plot-curves");
const statusEl = document.getElementById("plot-status");

// The category this tab draws. Electrostatics would be a second value here and a second
// heading; the rest of the file is already indifferent to which one it is.
const CATEGORY = "induction";

let trajectory = null;
let series = [];
let groups = [];
let currentAxis = null;
let equilibrium = null;
let selected = new Set();
let drawn = false;

// Built once per trajectory and then *mutated*, never rebuilt per draw. Plotly writes the
// user's zoom and pan into the layout object it was handed, so handing it a fresh object on
// every curve toggle throws that away and the chart snaps back to the full range -- which is
// the whole thing Plotly.react is being used to avoid. Rebuilding it per system is what makes
// the view reset when the data underneath it changes, which is the one time that is wanted.
let layout = null;

// --- styling ---------------------------------------------------------------
//
// Frontend-owned, on two independent channels, so the plot survives being read quickly:
// colour says which family of energy, dash says total vs breakdown. Both resolve to custom
// properties in viewer.css, so the sidebar swatches and the lines cannot drift apart.

// Reference colours follow the conventional SAPT decomposition scheme, keyed on category so
// an electrostatics tab lands on red without touching this map.
const CATEGORY_COLORS = {
  induction: "--sapt-ind",
  electrostatics: "--sapt-elst",
  exchange: "--sapt-exch",
  dispersion: "--sapt-disp",
};

// IPD curves are this app's own predictions rather than a benchmark, so they get a family of
// their own. Indexed by rank within the IPD group, not by position in `series`: the catalog
// emits only the modes a dataset actually has, and a positional colour would change under a
// mode when you switched datasets -- exactly when you are trying to compare them.
const IPD_COLORS = ["--ipd-1", "--ipd-2", "--ipd-3", "--ipd-4"];

// Dash per decomposition term, keyed on the label the backend gave it -- which is the term
// itself, so both SAPT levels are covered by three entries and a third level needs none.
// A component this does not name still draws; see styleFor.
const COMPONENT_DASH = {
  "ind20,r": "dash",
  "exch-ind20,r": "dot",
  "δHF": "dashdot",
};

// Headings for families that are not a level of theory. A family missing from here still gets
// a heading -- see groupHeading -- so a model added to the backend needs no entry.
const FAMILY_LABELS = {
  ipd: "IPD",
  mtp: "Multipole",
};

// Plotly needs a literal colour string; it cannot resolve var(--x) itself.
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function styleFor(entry) {
  // Solid means "this is a whole energy", dashed "this is one term of one". Read off `role`
  // rather than recognised from a label: SAPT2+/aDZ reports a total and no breakdown, and
  // guessing from its name would draw the one complete benchmark it has as a component.
  const dash = entry.role === "component" ? COMPONENT_DASH[entry.label] ?? "dash" : "solid";

  if (entry.family === "ipd") {
    return { color: cssVar(IPD_COLORS[entry.rank % IPD_COLORS.length]), dash };
  }
  if (entry.family === "sapt") {
    return { color: cssVar(CATEGORY_COLORS[entry.category] ?? "--sapt-ind"), dash };
  }
  // Any other family -- one this file has never heard of included. It gets a colour, a
  // checkbox and a line rather than being dropped for being unrecognised.
  return { color: cssVar("--model-other"), dash };
}

// --- data ------------------------------------------------------------------

// The two ways a system's geometries can be laid along x. Angstrom is preferred; Re is the
// fallback for data with no contact distance. Compared by identity, so these are the only two
// instances that ever exist.
const X_CONTACT = {
  title: "Closest contact (Å)",
  value: (frame) => frame.contact_distance_ang,
  format: ".3f",
  suffix: " Å",
  // The ratio to the equilibrium geometry, carried per point so it can be read off the hover
  // without giving up an Angstrom axis.
  extra: (frame) =>
    typeof frame.separation === "number" ? `${frame.separation.toFixed(2)} Rₑ` : "",
};

const X_SEPARATION = {
  title: "Separation (Rₑ)",
  value: (frame) => frame.separation,
  format: ".2f",
  suffix: "",
  extra: () => "",
};

// Pure: catalog and frames in, plot-ready series out. Nothing here touches the DOM or module
// state, which makes it the one part of this file checkable by pasting it into a console.
function buildSeries(frames, energyCatalog, category) {
  // ?? [] so a collection stored before energy_catalog existed renders an explanation rather
  // than throwing on the first filter.
  const entries = (energyCatalog ?? []).filter((entry) => entry.category === category);

  // Sorted on separation rather than on the x value: it is the field every frame is
  // guaranteed to have, and the two order a system identically. Plotly draws points in array
  // order, so an unsorted x doubles the line back on itself -- no error, just a wrong chart.
  const rows = [...frames].sort((a, b) => a.separation - b.separation);

  // Angstrom is the better axis -- a real distance rather than a ratio, so curves from
  // different systems are on the same footing. But it is decided once per system, never per
  // point: an axis that silently switched units partway along would be worse than one in the
  // units the whole system shares.
  const axis = rows.every((row) => typeof row.contact_distance_ang === "number")
    ? X_CONTACT
    : X_SEPARATION;

  let ipdRank = 0;
  const built = entries.map((energy) => ({
    id: energy.id,
    label: energy.label,
    category: energy.category,
    family: energy.family,
    level: energy.level,
    role: energy.role,
    // Rank within its own family, assigned in catalog order, so a curve keeps its colour when
    // a dataset carries a different subset of modes.
    rank: energy.family === "ipd" ? ipdRank++ : 0,
    // Two levels can both label their total "Induction total", so the legend and the unified
    // hover need the level to tell them apart. The curve list strips it back off, since there
    // the group heading is already saying it.
    traceName: energy.level ? `${energy.level} ${energy.label}` : energy.label,
    x: rows.map(axis.value),
    // null, not omitted. A mode computed for some frames and not others is an ordinary state,
    // and Plotly's default connectgaps:false breaks the line across the nulls. Dropping the
    // points instead would draw one straight segment over the gap, which reads as a finished
    // calculation.
    y: rows.map((row) => row.energies?.[energy.id] ?? null),
    customdata: rows.map((row) => axis.extra(row)),
  }));

  // An energy in the catalog with no value at any frame is not a curve.
  const kept = built.filter((entry) => entry.y.some((value) => value !== null));

  return { series: kept, axis, groups: buildGroups(kept) };
}

// One group per family, and per level within the SAPT family, in the order the catalog first
// mentioned them. Built from the payload rather than declared, so a family or a level this app
// has never seen still gets its own headed section.
function buildGroups(entries) {
  const groups = [];
  for (const entry of entries) {
    const key = groupKey(entry);
    if (!groups.some((group) => group.key === key)) {
      groups.push({ key, heading: groupHeading(entry) });
    }
  }
  return groups;
}

function groupKey(entry) {
  return entry.level ? `${entry.family}:${entry.level}` : entry.family;
}

function groupHeading(entry) {
  if (entry.level) {
    // "SAPT0 reference", "SAPT2+/aDZ reference" -- built, not looked up.
    return `${entry.level} reference`;
  }
  return (
    FAMILY_LABELS[entry.family] ??
    entry.family.charAt(0).toUpperCase() + entry.family.slice(1)
  );
}

// x of the equilibrium geometry, or null when the system has no 1.00 Rₑ frame. In Angstrom it
// is that frame's own contact distance; on a ratio axis it is 1 by definition.
function equilibriumMark(frames, axis) {
  const frame = frames.find(
    (candidate) =>
      typeof candidate.eq_ratio === "number" && Math.abs(candidate.eq_ratio - 1) < 1e-6,
  );
  if (!frame) {
    return null;
  }
  const x = axis === X_CONTACT ? frame.contact_distance_ang : 1;
  return typeof x === "number" ? x : null;
}

// A readable opening state rather than every curve at once: one reference to compare against,
// and the predictions being compared. The decomposition terms are deliberately off -- at
// 0.70 Rₑ ind20,r and exch-ind20,r are an order of magnitude larger than the total they sum
// to, and on a shared axis they flatten everything else into a line.
function defaultSelection(entries) {
  const reference = entries.find(
    (entry) => entry.family === "sapt" && entry.role === "total",
  );
  const models = entries.filter((entry) => entry.family === "ipd");
  return new Set([
    ...(reference ? [reference.id] : []),
    ...models.map((entry) => entry.id),
  ]);
}

// --- curve selector --------------------------------------------------------

function buildCurveList() {
  curveListEl.replaceChildren();

  const legend = document.createElement("legend");
  legend.textContent = "Curves";
  curveListEl.append(legend);

  for (const group of groups) {
    const members = series.filter((entry) => groupKey(entry) === group.key);
    if (members.length === 0) {
      continue;
    }

    const heading = document.createElement("p");
    heading.className = "plot-curve-group";
    heading.textContent = group.heading;
    curveListEl.append(heading);

    for (const entry of members) {
      curveListEl.append(curveRow(entry));
    }
  }
}

function curveRow(entry) {
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
  // The bare label: the group heading above already names the level, and repeating it would
  // make four near-identical lines of "SAPT0/cc-pVDZ ...".
  text.textContent = entry.label;

  row.append(box, swatch, text);
  return row;
}

// --- drawing ---------------------------------------------------------------

function isLaidOut() {
  // A chart built inside a hidden panel measures 0x0 and stays collapsed after the panel is
  // shown, so every draw and resize checks first. Same hazard the 3Dmol viewer has; see the
  // matching guard, isLaidOut, in viewer.js.
  return (
    Boolean(chartEl) &&
    !chartEl.closest("[hidden]") &&
    chartEl.clientWidth > 0 &&
    chartEl.clientHeight > 0
  );
}

function draw() {
  if (!isLaidOut() || layout === null) {
    return;
  }

  const axis = currentAxis ?? X_SEPARATION;
  // Only worth printing when the axis is in Angstrom; on an Rₑ axis the ratio is the x value
  // already showing in the hover header.
  const showRatio = axis === X_CONTACT;

  const traces = series
    .filter((entry) => selected.has(entry.id))
    .map((entry) => {
      const style = styleFor(entry);
      return {
        type: "scatter",
        mode: "lines+markers",
        name: entry.traceName,
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
      };
    });

  // The only field that changes between draws. Everything else about the layout -- including
  // whatever range the user has zoomed to -- is left exactly as Plotly last left it.
  layout.showlegend = traces.length > 0;

  Plotly.react(chartEl, traces, layout, CONFIG);
  drawn = true;
  renderStatus();
}

function buildLayout(axis) {
  const muted = cssVar("--muted");
  return {
    margin: { l: 66, r: 16, t: 20, b: 48 },
    font: { family: "system-ui, -apple-system, 'Segoe UI', sans-serif", color: cssVar("--text") },
    // Transparent so the .plot-panel card provides the background; a white plot on an
    // off-white card reads as a misaligned rectangle.
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: {
      title: { text: axis.title },
      gridcolor: cssVar("--border"),
      zeroline: false,
      hoverformat: axis.format,
      ticksuffix: axis.suffix,
    },
    yaxis: {
      title: { text: "Induction energy (kcal/mol)" },
      gridcolor: cssVar("--border"),
      // The sign of an interaction energy is the whole story, so the axis has to show where
      // zero is.
      zeroline: true,
      zerolinecolor: muted,
    },
    // The equilibrium geometry is the one physically privileged point on the scan. Marking it
    // makes that visible without hovering for it.
    shapes:
      equilibrium === null
        ? []
        : [
            {
              type: "line",
              x0: equilibrium,
              x1: equilibrium,
              yref: "paper",
              y0: 0,
              y1: 1,
              line: { color: muted, width: 1, dash: "dot" },
            },
          ],
    annotations:
      equilibrium === null
        ? []
        : [
            {
              x: equilibrium,
              yref: "paper",
              y: 1,
              yanchor: "bottom",
              text: "Rₑ",
              showarrow: false,
              font: { size: 11, color: muted },
            },
          ],
    // One hover box listing every curve at that separation -- the comparison this tab exists
    // for, without chasing individual points.
    hovermode: "x unified",
    legend: { orientation: "h", y: -0.18 },
    showlegend: false,
  };
}

function selectedCount() {
  return series.filter((entry) => selected.has(entry.id)).length;
}

const CONFIG = {
  responsive: true,
  displaylogo: false,
  // Wheel zoom, on top of Plotly's built-in drag-to-pan and box-zoom; double-click resets.
  // That is the entire zoom and pan requirement, for one config flag.
  scrollZoom: true,
  toImageButtonOptions: { filename: "induction-curves", scale: 2 },
};

// Four different situations with four different fixes, so they get four different sentences.
// Collapsing them into "nothing to show" would leave the reader unable to tell a dataset that
// carries no energies from one whose curves are all unticked.
//
// Counts what is *selected*, not what was last drawn. The two agree once the panel is visible,
// but this also runs while it is still hidden and nothing has been drawn at all -- reporting
// the trace count there would announce "no curves selected" over a full set of ticked boxes.
function renderStatus() {
  statusEl.className = "status";

  if (trajectory === null) {
    statusEl.textContent = "Select a system to plot.";
    return;
  }
  if (trajectory.energy_catalog === undefined) {
    statusEl.textContent =
      "This collection was registered before energy curves existed. Re-upload the dataframe to plot it.";
    return;
  }
  if (series.length === 0) {
    statusEl.textContent = "This dataset carries no induction energies.";
    return;
  }

  const visible = selectedCount();
  if (visible === 0) {
    statusEl.textContent = "No curves selected.";
    return;
  }

  const parts = [`Showing ${visible} of ${series.length} curves.`];
  if (!series.some((entry) => entry.family === "ipd")) {
    parts.push("No IPD curves yet — compute them on the Trajectory tab.");
  }
  statusEl.textContent = parts.join(" ");
}

// --- entry points ----------------------------------------------------------

// Called by viewer.js whenever it loads a trajectory. No fetch of its own: this is the very
// payload viewer.js just received, and re-requesting it would be a second copy that could
// disagree with what the 3D viewer is showing.
export function setTrajectory(loaded) {
  trajectory = loaded;

  ({
    series,
    groups,
    axis: currentAxis,
  } = buildSeries(loaded.frames, loaded.energy_catalog, CATEGORY));
  equilibrium = equilibriumMark(loaded.frames, currentAxis ?? X_SEPARATION);
  // A new object, so the incoming system is framed by its own data rather than inheriting the
  // zoom the last one was left at. Within a system the same object survives every toggle.
  layout = buildLayout(currentAxis ?? X_SEPARATION);

  // Keep whatever was ticked if those curves still exist, so switching system compares like
  // with like. Only when nothing survives does the readable default come back.
  const available = new Set(series.map((entry) => entry.id));
  const kept = [...selected].filter((id) => available.has(id));
  selected = kept.length ? new Set(kept) : defaultSelection(series);

  buildCurveList();
  draw();
  // draw() only reports status once it has a laid-out chart, and this runs while the Plots
  // panel is usually still hidden.
  if (!isLaidOut()) {
    renderStatus(0);
  }
}

// Called by viewer.js when the Plots tab becomes visible.
export function activate() {
  // Layout has not settled at the instant the tab is switched, so measuring now reads the
  // geometry from before the panel was shown.
  requestAnimationFrame(() => {
    if (!isLaidOut()) {
      return;
    }
    if (drawn) {
      Plotly.Plots.resize(chartEl);
    }
    // Unconditional: a redraw after a resize is cheap, and it is what fills the chart the
    // first time the tab is opened.
    draw();
  });
}

window.addEventListener("resize", () => {
  if (drawn && isLaidOut()) {
    Plotly.Plots.resize(chartEl);
  }
});
