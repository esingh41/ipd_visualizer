// Metadata-driven energy curves against dimer separation.
//
// The trajectory payload is geometry-oriented: one frame per separation, with all energies
// at that point. Each plot view transposes it into one series per catalog entry. Both charts
// use the same loaded payload, so there is no plot-specific endpoint or second fetch.

// --- styling ---------------------------------------------------------------

const CATEGORY_COLORS = {
  interaction: "--sapt-total",
  induction: "--sapt-ind",
  electrostatics: "--sapt-elst",
  exchange: "--sapt-exch",
  dispersion: "--sapt-disp",
};

const IPD_COLORS = ["--ipd-1", "--ipd-2", "--ipd-3", "--ipd-4"];

const AMOEBA_COLORS = {
  direct_070_mutual_039: "--amoeba-1",
  direct_039_mutual_039: "--amoeba-2",
};

const COMPONENT_DASH = {
  "ind20,r": "dash",
  "exch-ind20,r": "dot",
  "δHF": "dashdot",
};

const FAMILY_LABELS = {
  ipd: "IPD",
  mtp: "Multipole",
  amoeba: "AMOEBA+",
};

const SAPT_TERM_LABELS = {
  interaction: "Total",
  electrostatics: "Electrostatics",
  exchange: "Exchange",
  induction: "Induction",
  dispersion: "Dispersion",
};

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function styleFor(entry) {
  // Role remains category-relative: each raw SAPT category total is solid, while an
  // induction term such as ind20,r is a component and receives its distinct dash.
  const dash = entry.role === "component" ? COMPONENT_DASH[entry.label] ?? "dash" : "solid";

  if (entry.family === "ipd") {
    return { color: cssVar(IPD_COLORS[entry.rank % IPD_COLORS.length]), dash };
  }
  if (entry.family === "sapt") {
    return { color: cssVar(CATEGORY_COLORS[entry.category] ?? "--sapt-ind"), dash };
  }
  if (entry.family === "amoeba") {
    return { color: cssVar(AMOEBA_COLORS[entry.variant] ?? "--model-other"), dash };
  }
  return { color: cssVar("--model-other"), dash };
}

// --- data ------------------------------------------------------------------

const X_CONTACT = {
  title: "Closest contact (Å)",
  value: (frame) => frame.contact_distance_ang,
  format: ".3f",
  suffix: " Å",
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

// Pure: catalog and frames in, plot-ready series out. The predicate is the only scientific
// difference between the two views; neither one knows dataframe source-column names.
function buildSeries(
  frames,
  energyCatalog,
  entryFilter,
  labelFor = (entry) => entry.label,
) {
  const entries = (energyCatalog ?? []).filter(entryFilter);
  const rows = [...frames].sort((a, b) => a.separation - b.separation);
  const axis = rows.every((row) => typeof row.contact_distance_ang === "number")
    ? X_CONTACT
    : X_SEPARATION;

  let ipdRank = 0;
  const built = entries.map((energy) => {
    const label = labelFor(energy);
    return {
      id: energy.id,
      label,
      category: energy.category,
      family: energy.family,
      level: energy.level,
      role: energy.role,
      variant: energy.variant ?? null,
      rank: energy.family === "ipd" ? ipdRank++ : 0,
      traceName: energy.level ? `${energy.level} ${label}` : label,
      x: rows.map(axis.value),
      // Preserve null points so Plotly breaks partly computed curves rather than joining gaps.
      y: rows.map((row) => row.energies?.[energy.id] ?? null),
      customdata: rows.map((row) => axis.extra(row)),
    };
  });

  const kept = built.filter((entry) => entry.y.some((value) => value !== null));
  return { series: kept, axis, groups: buildGroups(kept) };
}

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
    return `${entry.level} reference`;
  }
  return (
    FAMILY_LABELS[entry.family] ??
    entry.family.charAt(0).toUpperCase() + entry.family.slice(1)
  );
}

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

function inductionDefaultSelection(entries) {
  const reference = entries.find(
    (entry) => entry.family === "sapt" && entry.role === "total",
  );
  const models = entries.filter((entry) => entry.family === "ipd");
  return new Set([
    ...(reference ? [reference.id] : []),
    ...models.map((entry) => entry.id),
  ]);
}

function saptDefaultSelection(entries) {
  const defaultLevel = entries.some((entry) => entry.level === "SAPT0")
    ? "SAPT0"
    : entries[0]?.level;
  return new Set(
    entries.filter((entry) => entry.level === defaultLevel).map((entry) => entry.id),
  );
}

function isRawSaptTerm(entry) {
  return (
    entry.family === "sapt" &&
    (entry.category === "interaction" || entry.parent_category === "interaction")
  );
}

// --- shared layout/config --------------------------------------------------

function buildLayout(axis, equilibrium, yAxisTitle) {
  const muted = cssVar("--muted");
  return {
    margin: { l: 66, r: 16, t: 20, b: 48 },
    font: { family: "system-ui, -apple-system, 'Segoe UI', sans-serif", color: cssVar("--text") },
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
      title: { text: yAxisTitle },
      gridcolor: cssVar("--border"),
      zeroline: true,
      zerolinecolor: muted,
    },
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
    hovermode: "x unified",
    legend: { orientation: "h", y: -0.18 },
    showlegend: false,
  };
}

function plotConfig(exportFilename) {
  return {
    responsive: true,
    displaylogo: false,
    scrollZoom: true,
    toImageButtonOptions: { filename: exportFilename, scale: 2 },
  };
}

// --- one stateful plot view ------------------------------------------------

function createPlotView({
  chartEl,
  curveListEl,
  statusEl,
  entryFilter,
  labelFor,
  defaultSelection,
  emptyMessage,
  statusExtra,
  yAxisTitle,
  exportFilename,
}) {
  let trajectory = null;
  let series = [];
  let groups = [];
  let currentAxis = null;
  let selected = new Set();
  let drawn = false;
  let layout = null;
  const config = plotConfig(exportFilename);

  function isLaidOut() {
    return (
      Boolean(chartEl) &&
      !chartEl.closest("[hidden]") &&
      chartEl.clientWidth > 0 &&
      chartEl.clientHeight > 0
    );
  }

  function selectedCount() {
    return series.filter((entry) => selected.has(entry.id)).length;
  }

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
      statusEl.textContent = emptyMessage;
      return;
    }

    const visible = selectedCount();
    if (visible === 0) {
      statusEl.textContent = "No curves selected.";
      return;
    }

    const parts = [`Showing ${visible} of ${series.length} curves.`];
    const extra = statusExtra(series);
    if (extra) {
      parts.push(extra);
    }
    statusEl.textContent = parts.join(" ");
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
    text.textContent = entry.label;
    row.append(box, swatch, text);
    return row;
  }

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

  function draw() {
    if (!isLaidOut() || layout === null) {
      return;
    }

    const axis = currentAxis ?? X_SEPARATION;
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
          hovertemplate: showRatio
            ? "%{y:.3f} kcal/mol · %{customdata}<extra></extra>"
            : "%{y:.3f} kcal/mol<extra></extra>",
          line: { color: style.color, width: 2, dash: style.dash },
          marker: { color: style.color, size: 6 },
        };
      });

    layout.showlegend = traces.length > 0;
    Plotly.react(chartEl, traces, layout, config);
    drawn = true;
    renderStatus();
  }

  function setTrajectory(loaded) {
    trajectory = loaded;
    ({ series, groups, axis: currentAxis } = buildSeries(
      loaded.frames,
      loaded.energy_catalog,
      entryFilter,
      labelFor,
    ));
    const equilibrium = equilibriumMark(loaded.frames, currentAxis ?? X_SEPARATION);
    layout = buildLayout(currentAxis ?? X_SEPARATION, equilibrium, yAxisTitle);

    // Preserve comparable selections across systems. Restore defaults only when none survive.
    const available = new Set(series.map((entry) => entry.id));
    const kept = [...selected].filter((id) => available.has(id));
    selected = kept.length ? new Set(kept) : defaultSelection(series);

    buildCurveList();
    draw();
    if (!isLaidOut()) {
      renderStatus();
    }
  }

  function activate() {
    if (!isLaidOut()) {
      return;
    }
    if (drawn) {
      Plotly.Plots.resize(chartEl);
    }
    draw();
  }

  function resize() {
    if (drawn && isLaidOut()) {
      Plotly.Plots.resize(chartEl);
    }
  }

  return { setTrajectory, activate, resize };
}

// --- the two plot instances -----------------------------------------------

const inductionView = createPlotView({
  chartEl: document.getElementById("plot-chart"),
  curveListEl: document.getElementById("plot-curves"),
  statusEl: document.getElementById("plot-status"),
  entryFilter: (entry) => entry.category === "induction",
  labelFor: (entry) => entry.label,
  defaultSelection: inductionDefaultSelection,
  emptyMessage: "This dataset carries no induction energies.",
  statusExtra: (entries) =>
    entries.some((entry) => entry.family === "ipd")
      ? ""
      : "No IPD curves yet — compute them on the IPD tab.",
  yAxisTitle: "Induction energy (kcal/mol)",
  exportFilename: "induction-curves",
});

const saptComponentsView = createPlotView({
  chartEl: document.getElementById("sapt-components-chart"),
  curveListEl: document.getElementById("sapt-components-curves"),
  statusEl: document.getElementById("sapt-components-status"),
  entryFilter: isRawSaptTerm,
  labelFor: (entry) => SAPT_TERM_LABELS[entry.category] ?? entry.label,
  defaultSelection: saptDefaultSelection,
  emptyMessage: "This dataset carries no SAPT interaction components.",
  statusExtra: () => "",
  yAxisTitle: "SAPT energy (kcal/mol)",
  exportFilename: "sapt-components",
});

const views = [inductionView, saptComponentsView];

export function setTrajectory(loaded) {
  views.forEach((view) => view.setTrajectory(loaded));
}

export function activate() {
  requestAnimationFrame(() => views.forEach((view) => view.activate()));
}

window.addEventListener("resize", () => views.forEach((view) => view.resize()));
