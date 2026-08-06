// The dimer scan as a movie: one frame per separation, played in order.
//
// The Visualize tab holds the geometry still and animates the SCF. This tab does the
// opposite -- every frame is a converged result, and what moves is the dimer itself. So the
// model is rebuilt each frame where Visualize rebuilds only the arrows, and the camera is
// framed once on the *widest* geometry so nothing walks out of view as the monomers separate.
//
// Fed by GET /api/ipd/trajectory, which is the one thing /api/ipd/systems deliberately does
// not carry: coordinates. The selector cascade above it is pure catalog, exactly as on the
// Visualize and Plot tabs.

import { callJson, referenceRows } from "/main.js";
import {
  DEFAULT_ARROW_LEN,
  addDipoleArrows,
  applyMoleculeStyle,
  isLaidOut,
  stripIntermonomerBonds,
} from "/molview.js";

const viewerEl = document.getElementById("trajViewer");
const statusEl = document.getElementById("trajStatus");
const infoEl = document.getElementById("trajInfo");
const frameLabelEl = document.getElementById("trajFrameLabel");
const timelineEl = document.getElementById("trajTimeline");
const scaleSliderEl = document.getElementById("trajScaleSlider");
const scaleReadoutEl = document.getElementById("trajScaleReadout");

const datasetEl = document.getElementById("trajDatasetSelect");
const systemEl = document.getElementById("trajSystemSelect");
const modelEl = document.getElementById("trajModelSelect");

const playBtn = document.getElementById("trajPlayBtn");
const prevBtn = document.getElementById("trajPrevBtn");
const nextBtn = document.getElementById("trajNextBtn");
const restartBtn = document.getElementById("trajRestartBtn");

// Slower than the Visualize tab's 125 ms. Fourteen separations at 8 fps is over in under two
// seconds, and each frame here reparses a model rather than just swapping shapes.
const PLAYBACK_MS = 250;
const MAX_DOTS = 40; // beyond this the timeline degrades to a range input

// null means "not fetched yet", which is what makes the fetch-once check work. An empty array
// would be indistinguishable from a server with no systems, and the tab would refetch forever.
let catalog = null;
let trajectory = null;
let currentFrame = 0;
let playTimer = null;
let baseScale = 1.0; // DEFAULT_ARROW_LEN / max|mu|, one value for the whole scan

// Built lazily: main.js can call createViewer at module scope because Visualize is the boot
// tab, but this panel boots `hidden`, and a WebGL canvas born in a 0x0 container stays
// collapsed forever after.
let viewer = null;
let currentModel = null;

function setStatus(message, kind = "") {
  statusEl.textContent = message;
  statusEl.classList.remove("error", "ok");
  if (kind) {
    statusEl.classList.add(kind);
  }
}

// --- selectors ------------------------------------------------------------

function unique(values) {
  return [...new Set(values)];
}

// Local rather than imported from main.js: the codebase keeps helpers this small private per
// module (matching copies at main.js and plot.js) so modules stay independent.
function fillSelect(selectEl, options, preferred) {
  selectEl.replaceChildren();
  for (const { value, label } of options) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    selectEl.append(option);
  }
  const values = options.map((option) => option.value);
  selectEl.value = values.includes(preferred) ? preferred : (values[0] ?? "");
  selectEl.disabled = options.length < 2;
  return selectEl.value;
}

function selectedEntries() {
  return catalog
    .filter(
      (entry) =>
        entry.dataset === datasetEl.value && entry.system_name === systemEl.value,
    )
    .sort((a, b) => a.separation - b.separation);
}

function populateDatasets() {
  fillSelect(
    datasetEl,
    unique(catalog.map((entry) => entry.dataset)).map((name) => ({
      value: name,
      label: name,
    })),
    datasetEl.value,
  );
}

function populateSystems() {
  fillSelect(
    systemEl,
    unique(
      catalog
        .filter((entry) => entry.dataset === datasetEl.value)
        .map((entry) => entry.system_name),
    ).map((name) => ({ value: name, label: name })),
    systemEl.value,
  );
}

// The models offered at *every* separation of this scan. An intersection rather than a union,
// for the same reason the server takes one: a model missing from one geometry would leave a
// hole in the middle of the animation. Computing it here as well as there keeps the selector
// honest before the fetch returns, and the two agree because they use the same rule.
function populateModels() {
  const entries = selectedEntries();
  const shared = entries.length
    ? entries
        .map((entry) => new Set((entry.models ?? []).map((model) => model.slug)))
        .reduce((left, right) => new Set([...left].filter((slug) => right.has(slug))))
    : new Set();

  fillSelect(
    modelEl,
    (entries[0]?.models ?? [])
      .filter((model) => shared.has(model.slug))
      .map((model) => ({ value: model.slug, label: model.label })),
    modelEl.value,
  );
}

// --- rendering ------------------------------------------------------------

function currentScale() {
  return baseScale * Number(scaleSliderEl.value);
}

function updateScaleReadout() {
  const multiplier = Number(scaleSliderEl.value);
  scaleReadoutEl.textContent = `${multiplier.toFixed(1)}× (${currentScale().toFixed(
    1,
  )} Å per a.u.)`;
}

// Replaces the nuclei with one frame's geometry. Never zooms: the camera is set once per
// trajectory, in frameCamera().
function renderGeometry(frame) {
  if (currentModel) {
    viewer.removeModel(currentModel);
  }
  currentModel = viewer.addModel(frame.xyz, "xyz");
  // Re-run on every frame, not once per trajectory: 3Dmol re-guesses bonds each time it
  // parses, and at close separations it invents an ion---oxygen stick that would otherwise
  // blink on and off as the distance crosses about 2.5 A.
  if (!stripIntermonomerBonds(currentModel, trajectory.n_atoms_A, trajectory.n_atoms)) {
    setStatus(
      "Skipped intermonomer bond removal: the viewer and the data disagree on atom count.",
      "error",
    );
  }
  applyMoleculeStyle(viewer);
}

// One zoomTo per trajectory, framed on the most separated geometry. Fitting the closest one
// instead would leave the far monomer outside the view for most of the playback, and
// re-zooming per frame would keep the dimer the same size on screen -- which is precisely the
// change this tab exists to show.
function frameCamera() {
  renderGeometry(trajectory.frames[trajectory.frames.length - 1]);
  viewer.zoomTo();
}

function showFrame(index) {
  if (!trajectory || !viewer) {
    return;
  }
  currentFrame = Math.max(0, Math.min(index, trajectory.n_frames - 1));
  const frame = trajectory.frames[currentFrame];

  renderGeometry(frame);
  viewer.removeAllShapes();
  addDipoleArrows(viewer, frame.coords, frame.mu, currentScale());
  viewer.render();

  updateFrameLabel();
  updateTimeline();
  updateInfo();
}

function frameTitle(index) {
  const frame = trajectory.frames[index];
  return frame.separation_label ?? String(frame.separation);
}

function updateFrameLabel() {
  frameLabelEl.textContent =
    trajectory.n_frames === 1
      ? `Single geometry: ${frameTitle(0)}`
      : `${frameTitle(currentFrame)} — geometry ${currentFrame + 1} of ${
          trajectory.n_frames
        }`;
}

// --- timeline -------------------------------------------------------------

function buildTimeline() {
  timelineEl.replaceChildren();

  if (trajectory.n_frames > MAX_DOTS) {
    const range = document.createElement("input");
    range.type = "range";
    range.className = "timeline-range";
    range.min = "0";
    range.max = String(trajectory.n_frames - 1);
    range.step = "1";
    range.value = "0";
    range.setAttribute("aria-label", "Separation");
    range.addEventListener("input", () => {
      stopPlayback();
      showFrame(Number(range.value));
    });
    timelineEl.append(range);
    return;
  }

  for (let i = 0; i < trajectory.n_frames; i += 1) {
    const dot = document.createElement("button");
    dot.type = "button";
    dot.className = "timeline-dot";
    dot.dataset.frame = String(i);
    dot.title = frameTitle(i);
    dot.setAttribute("aria-label", frameTitle(i));
    dot.addEventListener("click", () => {
      stopPlayback();
      showFrame(i);
    });
    timelineEl.append(dot);
  }
}

function updateTimeline() {
  const range = timelineEl.querySelector(".timeline-range");
  if (range) {
    range.value = String(currentFrame);
    range.setAttribute("aria-valuetext", frameTitle(currentFrame));
    return;
  }
  timelineEl.querySelectorAll(".timeline-dot").forEach((dot) => {
    if (Number(dot.dataset.frame) === currentFrame) {
      dot.setAttribute("aria-current", "true");
    } else {
      dot.removeAttribute("aria-current");
    }
  });
}

// --- caption --------------------------------------------------------------

// A row is [term, value]; a lone string is a full-width group heading. Same shape as the
// Visualize tab's system info, so the two read alike.
function updateInfo() {
  const frame = trajectory.frames[currentFrame];
  const modelLabel =
    trajectory.models.find((model) => model.slug === trajectory.model)?.label ??
    trajectory.model;

  const rows = [
    ["System", trajectory.system_name],
    ["Separation", frame.separation_label ?? String(frame.separation)],
    ...(frame.separation_alt_label ? [["", frame.separation_alt_label]] : []),
    ...(typeof frame.contact_distance_ang === "number"
      ? [["Closest contact", `${frame.contact_distance_ang.toFixed(3)} Å`]]
      : []),
    ["Atoms", `${trajectory.n_atoms}`],
    ...(trajectory.n_atoms_A
      ? [["Monomers", `${trajectory.n_atoms_A} + ${trajectory.n_atoms - trajectory.n_atoms_A}`]]
      : []),
    // Absent for the mu_ind model / mu_ind ref slugs, which are dipole snapshots rather than
    // energies, so the row is dropped rather than rendered as "undefined".
    ...(frame.energy_kcalmol === null || frame.energy_kcalmol === undefined
      ? []
      : [[modelLabel, `${frame.energy_kcalmol.toFixed(3)} kcal/mol`]]),
    ...referenceRows(frame.reference_energies),
  ];

  infoEl.replaceChildren();
  for (const row of rows) {
    if (typeof row === "string") {
      const heading = document.createElement("dt");
      heading.className = "system-group";
      heading.textContent = row;
      infoEl.append(heading);
      continue;
    }
    const [term, value] = row;
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = value;
    infoEl.append(dt, dd);
  }
}

// --- playback -------------------------------------------------------------

function setPlaybackEnabled(enabled) {
  for (const button of [playBtn, prevBtn, nextBtn, restartBtn]) {
    button.disabled = !enabled;
  }
}

function startPlayback() {
  if (playTimer !== null || !trajectory || trajectory.n_frames < 2) {
    return;
  }
  // Playing from the last separation restarts the scan rather than doing nothing.
  if (currentFrame >= trajectory.n_frames - 1) {
    showFrame(0);
  }
  playBtn.textContent = "Pause";
  playTimer = setInterval(() => {
    if (currentFrame >= trajectory.n_frames - 1) {
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

// --- loading --------------------------------------------------------------

async function loadTrajectory({ recenter = true } = {}) {
  if (!viewer || !datasetEl.value || !systemEl.value) {
    return;
  }
  stopPlayback();
  setStatus("Loading trajectory…");

  trajectory = await callJson(
    `/api/ipd/trajectory?dataset=${encodeURIComponent(datasetEl.value)}` +
      `&system=${encodeURIComponent(systemEl.value)}` +
      `&model=${encodeURIComponent(modelEl.value)}`,
  );

  baseScale =
    trajectory.max_abs_mu > 0 ? DEFAULT_ARROW_LEN / trajectory.max_abs_mu : 1.0;
  currentFrame = Math.min(currentFrame, trajectory.n_frames - 1);

  buildTimeline();
  setPlaybackEnabled(trajectory.n_frames > 1);
  updateScaleReadout();

  if (recenter) {
    // Only a new scan justifies moving the camera. Changing the dipole model leaves every
    // geometry exactly where it was.
    frameCamera();
    currentFrame = 0;
  }
  showFrame(currentFrame);

  setStatus(
    trajectory.n_frames === 1
      ? `${trajectory.system_name}: one geometry only, nothing to animate.`
      : `${trajectory.system_name}: ${trajectory.n_frames} separations, ` +
          `${trajectory.n_atoms} atoms.`,
    "ok",
  );
}

function reload(options) {
  loadTrajectory(options).catch((err) => setStatus(err.message, "error"));
}

async function loadCatalog() {
  setStatus("Loading system catalog…");
  try {
    catalog = await callJson("/api/ipd/systems");
  } catch (err) {
    // Left null so the next activation retries rather than sitting on the failure.
    catalog = null;
    setStatus(err.message, "error");
    return;
  }
  populateDatasets();
  populateSystems();
  populateModels();
  await loadTrajectory();
}

// --- listeners ------------------------------------------------------------

datasetEl.addEventListener("change", () => {
  stopPlayback();
  populateSystems();
  populateModels();
  reload();
});

systemEl.addEventListener("change", () => {
  stopPlayback();
  populateModels();
  reload();
});

// The geometries are unchanged, so the camera and the current separation stay put and only
// the arrows are recomputed.
modelEl.addEventListener("change", () => {
  stopPlayback();
  reload({ recenter: false });
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

// --- boot -----------------------------------------------------------------

function ensureViewer() {
  if (viewer) {
    return true;
  }
  if (!isLaidOut(viewerEl)) {
    return false;
  }
  viewer = $3Dmol.createViewer("trajViewer", { backgroundColor: "white" });
  return true;
}

document.addEventListener("app:tabchange", (event) => {
  if (event.detail.name !== "trajectory") {
    return;
  }
  // One animation frame so the revealed panel has been laid out before 3Dmol measures it.
  requestAnimationFrame(() => {
    if (!ensureViewer()) {
      return;
    }
    // Fetched on first activation rather than at load: the Visualize tab already pulls this
    // same endpoint at boot, and this tab may never be opened at all.
    if (catalog === null) {
      loadCatalog().catch((err) => setStatus(err.message, "error"));
      return;
    }
    viewer.resize();
    viewer.render();
  });
});

// Resize the canvas only -- re-rendering here would reset the camera.
window.addEventListener("resize", () => {
  if (viewer && isLaidOut(viewerEl)) {
    viewer.resize();
    viewer.render();
  }
});

// Deliberately no work at module load: the panel is hidden, so there is nothing to measure
// and nothing to show. Everything starts from the app:tabchange handler above.
