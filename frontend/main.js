import {
  DEFAULT_ARROW_LEN,
  addDipoleArrows,
  applyMoleculeStyle,
  isLaidOut,
  stripIntermonomerBonds,
} from "/molview.js";

const statusEl = document.getElementById("status");
const frameLabelEl = document.getElementById("frameLabel");
const timelineEl = document.getElementById("timeline");
const scaleSliderEl = document.getElementById("scaleSlider");
const scaleReadoutEl = document.getElementById("scaleReadout");
const systemInfoEl = document.getElementById("systemInfo");

const datasetEl = document.getElementById("datasetSelect");
const systemEl = document.getElementById("systemSelect");
const separationEl = document.getElementById("separationSelect");
const modelEl = document.getElementById("modelSelect");

const playBtn = document.getElementById("playBtn");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const restartBtn = document.getElementById("restartBtn");

const viewer = $3Dmol.createViewer("viewer", { backgroundColor: "white" });

// The arrow convention and the nuclei style live in molview.js, shared with the Trajectory
// tab. What stays here is the playback timing, which is this tab's alone.
const PLAYBACK_MS = 125; // DEFAULT_FPS = 8
const MAX_DOTS = 40; // beyond this the timeline degrades to a range input

// The four <select> elements hold the current selection; nothing here mirrors their
// values, so there is only ever one source of truth for what is displayed.
let catalog = [];
let currentSystem = null;
let currentFrame = 0;
let playTimer = null;
let baseScale = 1.0; // DEFAULT_ARROW_LEN / max|mu|, one value for all frames, atoms and models

export function setStatus(message, kind = "") {
  statusEl.textContent = message;
  statusEl.classList.remove("error", "ok");
  if (kind) {
    statusEl.classList.add(kind);
  }
}

export async function callJson(url, method = "GET", body = null) {
  // FormData carries its own multipart boundary, so the JSON content type must not be
  // set for an upload -- doing so makes the server parse the body as JSON and find no file.
  const isForm = body instanceof FormData;
  const res = await fetch(url, {
    method,
    headers: isForm ? {} : { "Content-Type": "application/json" },
    body: isForm ? body : body ? JSON.stringify(body) : null,
  });
  const rawText = await res.text();
  let data = null;
  try {
    data = rawText ? JSON.parse(rawText) : {};
  } catch {
    data = null;
  }
  if (!res.ok) {
    if (data && typeof data === "object") {
      // The structured fields (code, details, retryable) ride along on the Error so the
      // IPD panel can render more than a sentence. Plain callers keep using .message.
      const err = new Error(data.error || `Request failed: ${res.status}`);
      err.payload = data;
      throw err;
    }
    const fallback = rawText?.trim() || `Request failed: ${res.status}`;
    throw new Error(`Server error (${res.status}): ${fallback}`);
  }
  if (!data) {
    throw new Error("Server returned a non-JSON response.");
  }
  return data;
}

// --- arrows ---------------------------------------------------------------

function currentScale() {
  return baseScale * Number(scaleSliderEl.value);
}

// --- frame lifecycle ------------------------------------------------------

function frameLabel(index) {
  // The converted dataset files hold converged dipoles only, so a one-frame history is
  // not an SCF run and must not be labelled as iteration zero of one.
  if (currentSystem.n_frames === 1) {
    return "Converged induced dipoles";
  }
  if (index === 0) {
    return "Initial induced dipoles (direct-field seed)";
  }
  return `SCF iteration ${index} of ${currentSystem.n_frames - 1}`;
}

function frameTooltip(index) {
  if (currentSystem.n_frames === 1) {
    return "Converged induced dipoles";
  }
  return index === 0 ? "Frame 0 (initial guess)" : `SCF iteration ${index}`;
}

function updateFrameLabel() {
  frameLabelEl.textContent = frameLabel(currentFrame);
}

function updateTimeline() {
  const range = timelineEl.querySelector(".timeline-range");
  if (range) {
    range.value = String(currentFrame);
    range.setAttribute("aria-valuetext", frameTooltip(currentFrame));
    return;
  }
  timelineEl.querySelectorAll(".timeline-dot").forEach((dot) => {
    const isCurrent = Number(dot.dataset.frame) === currentFrame;
    if (isCurrent) {
      dot.setAttribute("aria-current", "true");
    } else {
      dot.removeAttribute("aria-current");
    }
  });
}

function updateScaleReadout() {
  const multiplier = Number(scaleSliderEl.value);
  scaleReadoutEl.textContent = `${multiplier.toFixed(1)}× (${currentScale().toFixed(
    1,
  )} Å per a.u.)`;
}

// Rebuilds only the arrows. The model, its styling, and the camera are left alone --
// no clear(), no removeAllModels(), no zoomTo().
function showFrame(index) {
  if (!currentSystem) {
    return;
  }
  currentFrame = Math.max(0, Math.min(index, currentSystem.n_frames - 1));
  viewer.removeAllShapes();
  addDipoleArrows(
    viewer,
    currentSystem.coords,
    currentSystem.mu_history[currentFrame],
    currentScale(),
  );
  updateFrameLabel();
  updateTimeline();
  viewer.render();
}

// --- timeline -------------------------------------------------------------

function buildTimeline() {
  timelineEl.replaceChildren();

  if (currentSystem.n_frames > MAX_DOTS) {
    const range = document.createElement("input");
    range.type = "range";
    range.className = "timeline-range";
    range.min = "0";
    range.max = String(currentSystem.n_frames - 1);
    range.step = "1";
    range.value = "0";
    range.setAttribute("aria-label", "SCF iteration");
    range.addEventListener("input", () => {
      stopPlayback();
      showFrame(Number(range.value));
    });
    timelineEl.append(range);
    return;
  }

  for (let i = 0; i < currentSystem.n_frames; i += 1) {
    const dot = document.createElement("button");
    dot.type = "button";
    dot.className = "timeline-dot";
    dot.dataset.frame = String(i);
    dot.title = frameTooltip(i);
    dot.setAttribute("aria-label", frameTooltip(i));
    dot.addEventListener("click", () => {
      stopPlayback();
      showFrame(i);
    });
    timelineEl.append(dot);
  }
}

// --- playback -------------------------------------------------------------

function startPlayback() {
  if (playTimer !== null || !currentSystem) {
    return;
  }
  // Playing from the final frame restarts the run rather than doing nothing.
  if (currentFrame >= currentSystem.n_frames - 1) {
    showFrame(0);
  }
  playBtn.textContent = "Pause";
  playTimer = setInterval(() => {
    if (currentFrame >= currentSystem.n_frames - 1) {
      stopPlayback();
      return;
    }
    showFrame(currentFrame + 1);
  }, PLAYBACK_MS);
}

function stopPlayback() {
  if (playTimer === null) {
    return;
  }
  clearInterval(playTimer);
  playTimer = null;
  playBtn.textContent = "Play";
}

function togglePlayback() {
  if (playTimer === null) {
    startPlayback();
  } else {
    stopPlayback();
  }
}

// --- selectors ------------------------------------------------------------

// Rebuilds a <select> from scratch, keeping the current choice when it is still offered
// and falling back to the first option when it is not. Returns the resulting value.
function fillSelect(selectEl, options, preferred) {
  selectEl.replaceChildren();
  for (const { value, label, title } of options) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    if (title) {
      option.title = title;
    }
    selectEl.append(option);
  }
  const values = options.map((option) => option.value);
  selectEl.value = values.includes(preferred) ? preferred : (values[0] ?? "");
  return selectEl.value;
}

function unique(values) {
  return [...new Set(values)];
}

function selectedSystems() {
  return catalog.filter(
    (entry) =>
      entry.dataset === datasetEl.value && entry.system_name === systemEl.value,
  );
}

// The separation option value *is* the system_id, so resolving a selection to a unique
// row is an exact lookup rather than a float comparison.
function resolveSelectedSystemId() {
  return separationEl.value;
}

function selectedCatalogEntry() {
  return catalog.find((entry) => entry.system_id === resolveSelectedSystemId()) ?? null;
}

function populateDatasetSelector() {
  fillSelect(
    datasetEl,
    unique(catalog.map((entry) => entry.dataset)).map((dataset) => ({
      value: dataset,
      label: dataset,
    })),
    datasetEl.value,
  );
}

function populateSystemSelector() {
  const names = unique(
    catalog
      .filter((entry) => entry.dataset === datasetEl.value)
      .map((entry) => entry.system_name),
  );
  fillSelect(
    systemEl,
    names.map((name) => ({ value: name, label: name })),
    systemEl.value,
  );
}

function populateSeparationSelector() {
  const options = selectedSystems()
    .slice()
    .sort((a, b) => a.separation - b.separation)
    .map((entry) => ({
      value: entry.system_id,
      label: entry.separation_label ?? String(entry.separation),
      title: entry.separation_alt_label ?? undefined,
    }));
  fillSelect(separationEl, options, separationEl.value);
}

function populateModelSelector() {
  const entry = selectedCatalogEntry();
  const models = entry?.models ?? [];
  fillSelect(
    modelEl,
    models.map((model) => ({ value: model.slug, label: model.label })),
    modelEl.value,
  );
}

// --- system information ---------------------------------------------------

// Reference energies as <dl> rows, one headed group per level of theory. A dataframe can
// carry several -- na-water has both SAPT0 and SAPT0/cc-pVDZ, computed in different basis
// sets -- and those are different benchmarks, so they get separate headings rather than being
// run together into one undifferentiated list.
//
// A row is [term, value]; a lone string is a full-width group heading, which is the shape
// both this tab and the Trajectory tab render. Exported for that second caller.
export function referenceRows(referenceEnergies = []) {
  const rows = [];
  let heading = null;
  for (const entry of referenceEnergies) {
    // Catalog order carries the meaning: each level's total comes first, then the terms that
    // decompose it. A level with no breakdown -- SAPT2+/aDZ gives only a total -- is a
    // one-row group, not a defect. An entry with no `level` at all can only come from a
    // catalog built before levels existed; it reads as one unnamed group rather than
    // "undefined reference".
    const level = entry.level ?? "SAPT";
    if (level !== heading) {
      rows.push(`${level} reference`);
      heading = level;
    }
    rows.push([entry.label, `${entry.kcalmol.toFixed(3)} kcal/mol`]);
  }
  return rows;
}

function setPlaybackEnabled(enabled) {
  for (const button of [playBtn, prevBtn, nextBtn, restartBtn]) {
    button.disabled = !enabled;
  }
}

function updateSystemInfo() {
  const {
    dataset,
    system_name: systemName,
    separation_label: separationLabel,
    separation_alt_label: separationAlt,
    n_atoms: nAtoms,
    n_atoms_A: nAtomsA,
    n_frames: nFrames,
    max_abs_mu: maxAbsMu,
    energy_kcalmol: energy,
    reference_energies: referenceEnergies = [],
  } = currentSystem;

  // A row is [term, value]; a lone string is a full-width group heading. The first three
  // come from the catalog, so they are absent for an on-demand result and are dropped
  // rather than rendered as "undefined".
  const rows = [
    ...(dataset ? [["Dataset", dataset]] : []),
    ...(systemName ? [["System", systemName]] : []),
    ...(currentSystem.system_id && !systemName
      ? [["Geometry", currentSystem.system_id]]
      : []),
    ...(separationLabel ? [["Separation", separationLabel]] : []),
    ...(separationAlt ? [["", separationAlt]] : []),
    ["Atoms", `${nAtoms}`],
    ...(nAtomsA ? [["Monomers", `${nAtomsA} + ${nAtoms - nAtomsA}`]] : []),
    ["SCF frames", nFrames === 1 ? "1 (converged only)" : `${nFrames}`],
    ["max |μ|", `${maxAbsMu.toFixed(4)} a.u.`],
    // Only present when the catalog could map this model to an energy column unambiguously.
    ...(energy === null || energy === undefined
      ? []
      : [["Polarization energy", `${energy.toFixed(3)} kcal/mol`]]),
    // Properties of the geometry, not of the selected model: they stay put while
    // "Polarization energy" above them follows the model selector. The headings are what say
    // so, which is why they are grouped rather than appended as plain rows.
    ...referenceRows(referenceEnergies),
  ];

  systemInfoEl.replaceChildren();
  for (const row of rows) {
    if (typeof row === "string") {
      const heading = document.createElement("dt");
      heading.className = "system-group";
      heading.textContent = row;
      systemInfoEl.append(heading);
      continue;
    }
    const [term, value] = row;
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = value;
    systemInfoEl.append(dt, dd);
  }

  if (currentSystem.note) {
    const note = document.createElement("dd");
    note.className = "system-note";
    note.textContent = currentSystem.note;
    systemInfoEl.append(note);
  }
}

// --- data loading ---------------------------------------------------------

// Re-reads the catalog and rebuilds the four selectors. Each one keeps its current choice
// when that choice still exists, so refreshing after a calculation adds the new geometry
// without moving the user's selection.
export async function refreshCatalog() {
  catalog = await callJson("/api/ipd/systems");
  populateDatasetSelector();
  populateSystemSelector();
  populateSeparationSelector();
  populateModelSelector();
  return catalog;
}

// Points the cascade at one catalog entry and loads it. Used by the compute panel so a
// freshly calculated geometry becomes the *selected* one rather than something the viewer
// shows while the selectors still name a different system.
export async function selectCatalogSystem(systemId, model = null) {
  const entry = catalog.find((candidate) => candidate.system_id === systemId);
  if (!entry) {
    return false;
  }
  // Each populate* reads the level above it, so the cascade has to be walked top-down:
  // setting a value before its options exist would be discarded by the rebuild.
  datasetEl.value = entry.dataset;
  populateSystemSelector();
  systemEl.value = entry.system_name;
  populateSeparationSelector();
  separationEl.value = systemId;
  populateModelSelector();
  if (model && [...modelEl.options].some((option) => option.value === model)) {
    modelEl.value = model;
  }
  await loadSelectedSystem();
  return true;
}

async function loadCatalog() {
  setStatus("Loading system catalog…");
  await refreshCatalog();
  if (catalog.length === 0) {
    throw new Error("The system catalog is empty. Run scripts/build_ipd_dataset.py.");
  }
}

// Renders a history payload, whatever produced it: the bundled catalog or an on-demand
// calculation. Both sources share this one path, so they share one camera policy, one
// global arrow scale and one timeline -- there is no second way to display a history.
export function showComputedSystem(data) {
  stopPlayback();

  // Only a different geometry justifies touching the model or the camera. Changing the
  // dipole model alone leaves the nuclei -- and therefore the view -- exactly as they are.
  const geometryChanged =
    currentSystem === null || currentSystem.system_id !== data.system_id;
  const frameCountChanged =
    currentSystem === null || currentSystem.n_frames !== data.n_frames;

  currentSystem = data;
  baseScale = data.max_abs_mu > 0 ? DEFAULT_ARROW_LEN / data.max_abs_mu : 1.0;

  if (geometryChanged) {
    // The geometry is fixed for the whole SCF, so the model is loaded and styled once
    // per system rather than once per frame.
    viewer.clear();
    const model = viewer.addModel(data.xyz, "xyz");
    if (!stripIntermonomerBonds(model, data.n_atoms_A, data.n_atoms)) {
      setStatus(
        "Skipped intermonomer bond removal: the viewer and the data disagree on atom count.",
        "error",
      );
    }
    applyMoleculeStyle(viewer);
    currentFrame = 0;
  }

  if (geometryChanged || frameCountChanged) {
    buildTimeline();
    currentFrame = Math.min(currentFrame, data.n_frames - 1);
  }

  setPlaybackEnabled(data.n_frames > 1);
  updateScaleReadout();
  showFrame(currentFrame);

  if (geometryChanged) {
    viewer.zoomTo(); // the only zoomTo in this file; never called on a frame or model change
    viewer.render();
  }

  updateSystemInfo();

  // A computed payload carries no catalog metadata, so the caption falls back to the
  // system id rather than printing "undefined undefined".
  const name = [data.system_name, data.separation_label]
    .filter(Boolean)
    .join(" ") || data.system_id || "System";
  setStatus(
    data.n_frames === 1
      ? `${name}: converged dipoles, ${data.n_atoms} atoms`
      : `${name}: ${data.n_frames - 1} SCF iterations, ${data.n_atoms} atoms`,
    "ok",
  );
}

async function loadSelectedSystem() {
  const systemId = resolveSelectedSystemId();
  if (!systemId) {
    return;
  }

  setStatus("Loading system…");
  showComputedSystem(
    await callJson(
      `/api/ipd/system?system_id=${encodeURIComponent(systemId)}` +
        `&model=${encodeURIComponent(modelEl.value)}`,
    ),
  );
}

// Every selector change ends in the same load, differing only in how far down the
// dataset -> system -> separation -> model cascade it has to repopulate first.
function reload() {
  loadSelectedSystem().catch((err) => setStatus(err.message, "error"));
}

// --- listeners ------------------------------------------------------------

datasetEl.addEventListener("change", () => {
  stopPlayback();
  populateSystemSelector();
  populateSeparationSelector();
  populateModelSelector();
  reload();
});

systemEl.addEventListener("change", () => {
  stopPlayback();
  populateSeparationSelector();
  populateModelSelector();
  reload();
});

separationEl.addEventListener("change", () => {
  stopPlayback();
  populateModelSelector();
  reload();
});

modelEl.addEventListener("change", () => {
  stopPlayback();
  reload();
});

playBtn.addEventListener("click", togglePlayback);

prevBtn.addEventListener("click", () => {
  stopPlayback();
  showFrame(currentFrame - 1);
});

nextBtn.addEventListener("click", () => {
  stopPlayback();
  showFrame(currentFrame + 1);
});

restartBtn.addEventListener("click", () => {
  stopPlayback();
  showFrame(0);
});

scaleSliderEl.addEventListener("input", () => {
  updateScaleReadout();
  showFrame(currentFrame);
});

// --- viewer sizing --------------------------------------------------------

// Safe to call from anywhere, including while the Visualize tab is hidden -- isLaidOut is
// what makes it so; see the note on it in molview.js.
export function resizeViewer() {
  if (!isLaidOut(document.getElementById("viewer"))) {
    return;
  }
  viewer.resize();
  viewer.render();
}

// Resize the canvas only -- re-rendering the model here would reset the camera.
window.addEventListener("resize", resizeViewer);

// tabs.js announces the switch rather than calling in here, which keeps it free of any
// dependency on this module. One animation frame so the revealed panel has been laid out
// before 3Dmol measures it.
document.addEventListener("app:tabchange", (event) => {
  if (event.detail.name !== "visualize") {
    return;
  }
  requestAnimationFrame(resizeViewer);
});

// The readout is not primed here: baseScale is only known once a system loads, and
// loadSelectedSystem() calls updateScaleReadout() itself.
loadCatalog()
  .then(loadSelectedSystem)
  .catch((err) => setStatus(err.message, "error"));
