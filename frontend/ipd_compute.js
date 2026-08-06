// On-demand IPD panel: upload a dimer dataframe, pick a geometry and a damping mode, run
// the SCF on the server, and load the result into the viewer that main.js already owns.
//
// A second module rather than more of main.js, because the two have almost no shared
// state: this one talks to /api/datasets and knows nothing about arrows, frames or the
// camera. The seam is deliberately three imports wide.

import {
  callJson,
  refreshCatalog,
  selectCatalogSystem,
  setStatus,
  showComputedSystem,
} from "/main.js";
import { showTab } from "/tabs.js";

// Mirrors MODEL_FOR_RADIUS in backend/services/ipd_compute.py, which in turn matches the
// slugs scripts/build_ipd_dataset.py bakes into the bundled catalog. Used to point the
// viewer's Model selector at the mode that was just computed.
const MODEL_SLUG_FOR_RADIUS = {
  none: "mbis_0_39_all",
  intra: "radius_thole_ts_mbis_intra_only",
  inter: "radius_thole_ts_mbis_inter_only",
  both: "radius_thole_ts_mbis_intra_inter",
};

const badgeEl = document.getElementById("ipdBadge");
const detailsEl = document.getElementById("ipdDetails");
const fileEl = document.getElementById("ipdFile");
const datasetEl = document.getElementById("ipdDatasetSelect");
const systemEl = document.getElementById("ipdSystemSelect");
const geometryEl = document.getElementById("ipdGeometrySelect");
const radiusEl = document.getElementById("ipdRadius");
const runBtn = document.getElementById("ipdRunBtn");
const runGroupBtn = document.getElementById("ipdRunGroupBtn");
const recomputeBtn = document.getElementById("ipdRecomputeBtn");
const recomputeGroupBtn = document.getElementById("ipdRecomputeGroupBtn");
const resultEl = document.getElementById("ipdResult");
const exportBtn = document.getElementById("ipdExportBtn");
const restoreBtn = document.getElementById("ipdRestoreBtn");
const viewBtn = document.getElementById("ipdViewBtn");
const statusEl = document.getElementById("ipdStatus");

// One object holds everything the panel renders from, and renderPanel() is the only thing
// that writes to the DOM. Every handler mutates state then calls it once, so there is no
// path where two controls disagree about what is selected.
const ipd = {
  phase: "idle", // idle | uploading | validating | ready | running | error
  capability: null,
  dataset: null, // the full /api/datasets/<id> response
  selection: { system: null, rowId: null, radius: "both" },
  result: null,
  // Where "View in visualizer" goes after a group run: {rowId, label, modelSlug}. Held
  // separately from `result` because a group run produces a summary, not one result, and
  // cleared whenever it could point at something stale (see clearViewTarget callers).
  viewTarget: null,
  error: null,
};

// --- helpers ---------------------------------------------------------------

function fillSelect(selectEl, options, preferred) {
  selectEl.replaceChildren();
  for (const { value, label, title, disabled } of options) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    if (title) option.title = title;
    if (disabled) option.disabled = true;
    selectEl.append(option);
  }
  const values = options.map((option) => option.value);
  selectEl.value = values.includes(preferred) ? preferred : (values[0] ?? "");
  return selectEl.value;
}

function setBadge(text, kind) {
  badgeEl.textContent = text;
  badgeEl.className = `status-badge ${kind}`;
}

function setPanelStatus(message, kind = "") {
  statusEl.textContent = message;
  statusEl.classList.remove("error", "ok");
  if (kind) statusEl.classList.add(kind);
}

function currentSystem() {
  return (ipd.dataset?.systems ?? []).find(
    (entry) => entry.group === ipd.selection.system,
  );
}

function currentGeometry() {
  return (currentSystem()?.geometries ?? []).find(
    (geometry) => geometry.row_id === ipd.selection.rowId,
  );
}

// A geometry's option text carries its stored-result state, so the user can see which
// damping modes are already computed without selecting each one in turn. The modes come
// from the server's own has_ipd_result check, never from a guess made here.
function geometryOptionLabel(geometry) {
  const saved = Object.entries(geometry.stored)
    .filter(([, present]) => present)
    .map(([mode]) => mode);
  if (!geometry.usable) return `${geometry.label} — unusable`;
  return saved.length
    ? `${geometry.label} — ${saved.join(", ")} saved`
    : `${geometry.label} — not calculated`;
}

// --- rendering -------------------------------------------------------------

function renderBadge() {
  if (!ipd.capability) {
    setBadge("Checking server…", "");
    return;
  }
  if (!ipd.capability.available) {
    setBadge(
      ipd.capability.code === "apnet_pt_missing"
        ? "Server missing apnet_pt"
        : "apnet_pt unusable",
      "bad",
    );
    return;
  }
  if (!ipd.dataset) {
    setBadge("IPD ready — no dataset", "ok");
    return;
  }
  const compat = ipd.dataset.compatibility ?? {};
  if (!compat.ok) {
    setBadge(
      compat.missing_columns?.length ? "Missing MBIS data" : "Dataset unusable",
      "warn",
    );
    return;
  }
  setBadge("IPD ready", "ok");
}

function renderDetails() {
  const lines = [];
  if (ipd.capability && !ipd.capability.available) {
    lines.push(ipd.capability.reason);
    if (ipd.capability.missing_kwargs?.length) {
      lines.push(`Missing arguments: ${ipd.capability.missing_kwargs.join(", ")}`);
    }
  }
  if (ipd.dataset) {
    const compat = ipd.dataset.compatibility ?? {};
    lines.push(
      `${ipd.dataset.n_systems} system(s), ${ipd.dataset.n_geometries} geometries, ` +
        `${ipd.dataset.n_with_results} with saved results`,
    );
    if (compat.missing_columns?.length) {
      lines.push(`Missing columns: ${compat.missing_columns.join(", ")}`);
    }
    if (compat.ok === false && !compat.missing_columns?.length) {
      lines.push(`No usable rows (${compat.usable_rows}/${compat.total_rows}).`);
    }
    // The working copy carries columns the upload did not, and the export hands them back,
    // so it is said out loud rather than left as a surprise in the downloaded file.
    if (ipd.dataset.added_columns?.length) {
      lines.push(
        `Derived for you: ${ipd.dataset.added_columns.join(", ")} — included in the export.`,
      );
    }
  }

  detailsEl.replaceChildren();
  for (const line of lines.filter(Boolean)) {
    const item = document.createElement("li");
    item.textContent = line;
    detailsEl.append(item);
  }
}

function renderSelectors() {
  const systems = ipd.dataset?.systems ?? [];
  ipd.selection.system = fillSelect(
    systemEl,
    systems.map((entry) => ({
      value: entry.group,
      label: `${entry.group} (${entry.geometries.length})`,
    })),
    ipd.selection.system,
  );

  const geometries = currentSystem()?.geometries ?? [];
  ipd.selection.rowId = fillSelect(
    geometryEl,
    geometries.map((geometry) => ({
      value: geometry.row_id,
      label: geometryOptionLabel(geometry),
      title: geometry.system_id,
      disabled: !geometry.usable,
    })),
    ipd.selection.rowId,
  );

  // Locked while a run is in flight as well as when empty: changing the dataset, system
  // or geometry mid-run would retarget a calculation that has already been dispatched.
  const busy = ipd.phase === "running" || ipd.phase === "uploading";
  systemEl.disabled = systems.length === 0 || busy;
  geometryEl.disabled = geometries.length === 0 || busy;
  datasetEl.disabled = datasetEl.options.length === 0 || busy;
  fileEl.disabled = busy;

  for (const button of radiusEl.querySelectorAll("button")) {
    const active = button.dataset.radius === ipd.selection.radius;
    button.setAttribute("aria-checked", String(active));
    button.classList.toggle("active", active);
    button.disabled = busy;
  }
}

function renderRunControl() {
  const geometry = currentGeometry();
  const system = currentSystem();
  const stored = geometry?.stored?.[ipd.selection.radius] ?? false;
  const busy = ipd.phase === "running";
  const ready =
    Boolean(ipd.capability?.available) &&
    Boolean(ipd.dataset?.compatibility?.ok) &&
    Boolean(geometry?.usable);

  if (busy) {
    runBtn.textContent = "Running IPD…";
  } else if (stored) {
    runBtn.textContent = "Load saved IPD";
  } else {
    // No separate "save result" control: every calculation is persisted, and the label
    // is what says so.
    runBtn.textContent = "Run and save IPD";
  }
  runBtn.disabled = !ready || busy;

  // How much of this trajectory is still missing for the selected radius mode, so the
  // button can say what pressing it will actually do.
  const geometries = system?.geometries ?? [];
  const missing = geometries.filter(
    (candidate) => candidate.usable && !candidate.stored?.[ipd.selection.radius],
  ).length;
  runGroupBtn.textContent = busy
    ? "Running…"
    : missing === 0
      ? `All ${geometries.length} saved`
      : `Run all ${missing} remaining`;
  runGroupBtn.disabled = !ready || busy || missing === 0;

  // Recompute only means something when there is something to discard.
  recomputeBtn.hidden = !stored;
  recomputeBtn.disabled = !ready || busy;
  recomputeGroupBtn.hidden = geometries.length === 0 || missing === geometries.length;
  recomputeGroupBtn.disabled = !ready || busy;

  exportBtn.disabled = !ipd.dataset;
  restoreBtn.disabled = !ipd.dataset || (ipd.dataset?.n_with_results ?? 0) === 0;

  // Names the geometry it will open, so the destination is never a guess.
  viewBtn.hidden = !ipd.viewTarget;
  viewBtn.disabled = busy;
  if (ipd.viewTarget) {
    viewBtn.textContent = `View ${ipd.viewTarget.label} in visualizer`;
  }
}

// Called wherever the stored target could stop matching what is on screen: a different
// dataset, a different system, a fresh upload, or a restore that deleted the results.
function clearViewTarget() {
  ipd.viewTarget = null;
}

function renderResult() {
  resultEl.replaceChildren();
  if (!ipd.result) return;

  const rows = [
    ["Source", ipd.result.result_source === "stored" ? "Saved result" : "Newly computed"],
    ["Radius", ipd.result.radius],
    [
      "Converged",
      ipd.result.converged
        ? `yes, in ${ipd.result.iterations} iterations`
        : `no (${ipd.result.iterations} iterations)`,
    ],
  ];
  if (ipd.result.energy_kcalmol !== null && ipd.result.energy_kcalmol !== undefined) {
    rows.push(["Induction energy", `${ipd.result.energy_kcalmol.toFixed(3)} kcal/mol`]);
  }
  if (ipd.result.dataset_updated) {
    rows.push(["Saved", `${ipd.result.columns_written.length} columns written`]);
  }

  for (const [term, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = value;
    resultEl.append(dt, dd);
  }

  for (const warning of ipd.result.warnings ?? []) {
    const note = document.createElement("dd");
    note.className = "system-note warn";
    note.textContent = warning;
    resultEl.append(note);
  }
}

function renderPanel() {
  renderBadge();
  renderDetails();
  renderSelectors();
  renderRunControl();
  renderResult();
}

// --- data flow -------------------------------------------------------------

function reportError(err) {
  ipd.phase = "error";
  ipd.error = err;
  const details = err.payload?.details ?? {};
  const extra = details.missing?.length ? ` (${details.missing.join(", ")})` : "";
  const retry = err.payload?.retryable ? " You can try again." : "";
  setPanelStatus(`${err.message}${extra}${retry}`, "error");
  renderPanel();
}

async function refreshDatasetList(preferred) {
  const datasets = await callJson("/api/datasets");
  fillSelect(
    datasetEl,
    datasets.map((entry) => ({
      value: entry.dataset_id,
      label: `${entry.name} (${entry.n_geometries ?? entry.n_rows} geometries)`,
      title: entry.registered_at ?? undefined,
    })),
    preferred ?? datasetEl.value,
  );
  // Disabled state is owned by renderSelectors, which also accounts for a run in flight.
  return datasetEl.value;
}

async function loadDataset(datasetId) {
  if (!datasetId) {
    ipd.dataset = null;
    renderPanel();
    return;
  }
  ipd.phase = "validating";
  setPanelStatus("Loading dataset…");
  renderPanel();

  ipd.dataset = await callJson(`/api/datasets/${encodeURIComponent(datasetId)}`);
  // A newly loaded dataset invalidates any previous selection, so let renderSelectors
  // fall back to the first option of each rather than keeping a stale row_id.
  ipd.selection.system = null;
  ipd.selection.rowId = null;
  ipd.result = null;
  clearViewTarget();
  ipd.phase = "ready";
  setPanelStatus(
    ipd.dataset.compatibility?.ok
      ? "Dataset ready. Choose a geometry and a radius mode."
      : "Dataset registered, but IPD cannot run against it.",
    ipd.dataset.compatibility?.ok ? "ok" : "error",
  );
  renderPanel();
}

async function runCalculation(forceRecompute) {
  const geometry = currentGeometry();
  if (!geometry) return;

  ipd.phase = "running";
  setPanelStatus(
    forceRecompute ? "Recomputing induced dipoles…" : "Running induced-dipole SCF…",
  );
  renderPanel();

  const payload = await callJson(
    `/api/datasets/${encodeURIComponent(ipd.dataset.dataset_id)}/ipd`,
    "POST",
    {
      row_id: geometry.row_id,
      radius: ipd.selection.radius,
      force_recompute: Boolean(forceRecompute),
    },
  );

  ipd.result = payload;
  ipd.phase = "ready";

  // Catalog selection and viewer state are settled *before* the page is revealed, so the
  // Visualize tab is already showing the right geometry when it appears.
  await syncAfterRun(payload.row_id, payload);
  setPanelStatus(
    payload.result_source === "stored"
      ? "Loaded the saved result."
      : payload.converged
        ? "Calculation finished and saved."
        : "Calculation saved, but the SCF did not converge.",
    payload.converged ? "ok" : "error",
  );
  renderPanel();

  // Running one geometry means "show me this one". Focus travels with the switch: the
  // button that was just pressed is about to be hidden, and leaving focus on a hidden
  // element strands keyboard and screen-reader users.
  //
  // A throw from any of the above skips this, so a failed run stays put -- which is what
  // reportError relies on.
  showTab("visualize", { focusTab: true });
}

// A result is only really "in" the app once the main catalog knows about it. Refreshing
// both and then selecting the geometry there means the viewer, the four catalog selectors
// and this panel all describe the same thing -- rather than the viewer showing one
// geometry while the selectors still name another.
async function syncAfterRun(rowId, payload = null) {
  const datasetId = ipd.dataset.dataset_id;
  ipd.dataset = await callJson(`/api/datasets/${encodeURIComponent(datasetId)}`);

  const modelSlug = MODEL_SLUG_FOR_RADIUS[ipd.selection.radius];
  await refreshCatalog();
  const selected = rowId
    ? await selectCatalogSystem(`${datasetId}/${rowId}`, modelSlug)
    : false;

  // Only reachable if the catalog somehow lacks the geometry we just computed; render the
  // payload directly rather than leaving the viewer on stale data.
  if (!selected && payload) {
    showComputedSystem(payload);
  }
}

async function runGroup(forceRecompute) {
  const system = currentSystem();
  if (!system) return;

  ipd.phase = "running";
  setPanelStatus(
    `${forceRecompute ? "Recomputing" : "Computing"} ${system.geometries.length} ` +
      `geometries in ${system.group}…`,
  );
  renderPanel();

  const summary = await callJson(
    `/api/datasets/${encodeURIComponent(ipd.dataset.dataset_id)}/ipd/group`,
    "POST",
    {
      system: system.group,
      radius: ipd.selection.radius,
      force_recompute: Boolean(forceRecompute),
    },
  );

  ipd.phase = "ready";
  ipd.result = null; // a group summary is not a single-geometry result
  await syncAfterRun(summary.first_row_id);

  // Deterministic target: run_group returns first_row_id as the first geometry it touched
  // in *separation* order, so this is always the lowest-separation result of the batch --
  // never "whichever finished last". Labelled with the geometry so it is not a mystery.
  const landed = (currentSystem()?.geometries ?? []).find(
    (geometry) => geometry.row_id === summary.first_row_id,
  );
  ipd.viewTarget = summary.first_row_id
    ? {
        rowId: summary.first_row_id,
        label: landed?.label ?? summary.first_row_id,
        modelSlug: MODEL_SLUG_FOR_RADIUS[ipd.selection.radius],
      }
    : null;

  const parts = [`${summary.computed} computed`, `${summary.reused} already saved`];
  if (summary.failed.length) parts.push(`${summary.failed.length} failed`);
  if (summary.not_converged.length) {
    parts.push(`${summary.not_converged.length} did not converge`);
  }
  setPanelStatus(
    `${system.group}: ${parts.join(", ")} in ${summary.elapsed_s}s.`,
    summary.failed.length ? "error" : "ok",
  );

  // Per-geometry failures are listed rather than summarised away: which geometry failed
  // and why is the whole point of not aborting the batch.
  resultEl.replaceChildren();
  for (const failure of summary.failed) {
    const dt = document.createElement("dt");
    dt.textContent = failure.row_id;
    const dd = document.createElement("dd");
    dd.textContent = failure.error;
    resultEl.append(dt, dd);
  }
  renderPanel();
}

// --- listeners -------------------------------------------------------------

fileEl.addEventListener("change", async () => {
  const file = fileEl.files?.[0];
  if (!file) return;
  ipd.phase = "uploading";
  clearViewTarget();
  setPanelStatus(`Uploading ${file.name}…`);
  renderPanel();

  try {
    const form = new FormData();
    form.append("file", file);
    const registered = await callJson("/api/datasets", "POST", form);
    await refreshDatasetList(registered.dataset_id);
    await loadDataset(registered.dataset_id);
    // An uploaded dataframe may already contain results from a previous run, in which case
    // those geometries are viewable immediately -- so the catalog has to be re-read even
    // though nothing was computed here.
    await refreshCatalog();
  } catch (err) {
    reportError(err);
  } finally {
    // Clearing the input means re-picking the same file fires `change` again, which is
    // what a user expects after a failed upload.
    fileEl.value = "";
  }
});

datasetEl.addEventListener("change", () => {
  loadDataset(datasetEl.value).catch(reportError);
});

systemEl.addEventListener("change", () => {
  ipd.selection.system = systemEl.value;
  ipd.selection.rowId = null; // the old row belongs to a different trajectory
  clearViewTarget(); // ... and so does any geometry a previous batch left pointed at
  renderPanel();
});

geometryEl.addEventListener("change", () => {
  ipd.selection.rowId = geometryEl.value;
  renderPanel();
});

radiusEl.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-radius]");
  if (!button) return;
  ipd.selection.radius = button.dataset.radius;
  renderPanel();
});

runBtn.addEventListener("click", () => {
  runCalculation(false).catch(reportError);
});

recomputeBtn.addEventListener("click", () => {
  runCalculation(true).catch(reportError);
});

runGroupBtn.addEventListener("click", () => {
  runGroup(false).catch(reportError);
});

recomputeGroupBtn.addEventListener("click", () => {
  const system = currentSystem();
  if (!system) return;
  if (
    !window.confirm(
      `Recompute all ${system.geometries.length} geometries in ${system.group}?`,
    )
  ) {
    return;
  }
  runGroup(true).catch(reportError);
});

exportBtn.addEventListener("click", () => {
  // A plain navigation rather than fetch: the response is an attachment, and letting the
  // browser handle it is what produces the save dialog.
  window.location.href = `/api/datasets/${encodeURIComponent(
    ipd.dataset.dataset_id,
  )}/export`;
});

restoreBtn.addEventListener("click", async () => {
  if (!window.confirm("Discard every computed IPD result in this dataset?")) return;
  try {
    ipd.dataset = await callJson(
      `/api/datasets/${encodeURIComponent(ipd.dataset.dataset_id)}/restore`,
      "POST",
    );
    ipd.result = null;
    clearViewTarget(); // the geometry it pointed at no longer has a stored result
    await refreshCatalog(); // ... and it has to leave the viewer's selectors too
    setPanelStatus("Restored the originally uploaded dataframe.", "ok");
    renderPanel();
  } catch (err) {
    reportError(err);
  }
});

viewBtn.addEventListener("click", () => {
  const target = ipd.viewTarget;
  if (!target || !ipd.dataset) return;
  selectCatalogSystem(
    `${ipd.dataset.dataset_id}/${target.rowId}`,
    target.modelSlug,
  )
    .then(() => showTab("visualize", { focusTab: true }))
    .catch(reportError);
});

// --- boot ------------------------------------------------------------------

async function boot() {
  // Also warms the server's torch/apnet_pt import, so the first calculation pays only
  // for the science.
  ipd.capability = await callJson("/api/ipd/capability");
  const datasetId = await refreshDatasetList();
  renderPanel();
  if (datasetId) {
    await loadDataset(datasetId);
  } else {
    setPanelStatus(
      ipd.capability.available
        ? "Upload a dimer dataframe to compute induced dipoles."
        : "IPD computation is unavailable on this server.",
      ipd.capability.available ? "" : "error",
    );
  }
}

boot().catch((err) => {
  reportError(err);
  setStatus(`IPD panel failed to start: ${err.message}`, "error");
});
