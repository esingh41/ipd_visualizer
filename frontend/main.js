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

// Ported from pymol_dipole.py / pymol_dipole_movie.py. Changing these changes the
// scientific reading of the picture, so they stay named and grouped.
const DEFAULT_ARROW_LEN = 2.0; // Angstrom, longest arrow in the whole history
const MIN_MU = 1e-6; // below this an arrow is skipped entirely
const CONE_OVERSHOOT = 1.3; // total arrow length is 1.3 * scale * |mu|
const CYL_RADIUS = 0.05;
const CONE_RADIUS = 0.1;
const ARROW_COLOR = 0xbf00bf; // INDUCED_MODEL_COLOR (0.75, 0.0, 0.75)
const SPHERE_SCALE = 0.15;
const STICK_RADIUS = 0.55 * SPHERE_SCALE; // 0.0825
const PLAYBACK_MS = 125; // DEFAULT_FPS = 8
const MAX_DOTS = 40; // beyond this the timeline degrades to a range input

// The four <select> elements hold the current selection; nothing here mirrors their
// values, so there is only ever one source of truth for what is displayed.
let catalog = [];
let currentSystem = null;
let currentFrame = 0;
let playTimer = null;
let baseScale = 1.0; // DEFAULT_ARROW_LEN / max|mu|, one value for all frames, atoms and models

function setStatus(message, kind = "") {
  statusEl.textContent = message;
  statusEl.classList.remove("error", "ok");
  if (kind) {
    statusEl.classList.add(kind);
  }
}

async function callJson(url, method = "GET", body = null) {
  const res = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : null,
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
      throw new Error(data.error || `Request failed: ${res.status}`);
    }
    const fallback = rawText?.trim() || `Request failed: ${res.status}`;
    throw new Error(`Server error (${res.status}): ${fallback}`);
  }
  if (!data) {
    throw new Error("Server returned a non-JSON response.");
  }
  return data;
}

// --- structure ------------------------------------------------------------

// XYZ carries no bond table, so 3Dmol guesses bonds by distance: it bonds two atoms whenever
// they sit closer than r1 + r2 + 0.25 A, which for Na-O is 2.52 A and so invents an Na+---O
// stick at every separation up to 1.10 Re. The two monomers of a dimer are non-bonded by
// construction, so any bond crossing the A/B boundary is an artefact of that guess.
//
// Bonds are assigned once when the model is parsed and re-read from atom.bonds on every
// setStyle, so deleting them here -- before the first setStyle -- is all it takes; there is no
// drawn geometry to fix up afterwards.
function stripIntermonomerBonds(model, nAtomsA) {
  if (!nAtomsA) {
    // No declared monomer split: there is no boundary to enforce, and inventing one would be
    // worse than leaving 3Dmol's guess alone.
    return;
  }

  // An empty selection returns the model's atoms, by reference, in the same index order that
  // atom.bonds refers to. The XYZ parser does not set atom.index, so position *is* the index
  // -- which only holds while the selection is everything.
  const atoms = model.selectedAtoms({});
  if (atoms.length !== currentSystem.n_atoms) {
    setStatus(
      `Skipped intermonomer bond removal: viewer has ${atoms.length} atoms, ` +
        `data has ${currentSystem.n_atoms}.`,
      "error",
    );
    return;
  }

  const monomerOf = (index) => index < nAtomsA;
  atoms.forEach((atom, index) => {
    const bonds = [];
    const bondOrder = [];
    for (let i = 0; i < atom.bonds.length; i += 1) {
      if (monomerOf(atom.bonds[i]) === monomerOf(index)) {
        bonds.push(atom.bonds[i]);
        bondOrder.push(atom.bondOrder[i]);
      }
    }
    // Every atom is filtered, so both halves of a crossing bond go: a bond listed by only one
    // of its two atoms would still render, as a half-length stick.
    atom.bonds = bonds;
    atom.bondOrder = bondOrder;
  });
}

// --- arrows ---------------------------------------------------------------

function currentScale() {
  return baseScale * Number(scaleSliderEl.value);
}

// Reproduces _arrow_cgo (pymol_dipole.py:158-193): the arrow *begins* at the atom
// centre rather than being centred on it, and the cone is added on top of the full
// shaft, so the tip reaches CONE_OVERSHOOT * scale * |mu|.
function addDipoleArrows(frame, scale) {
  const coords = currentSystem.coords;
  for (let i = 0; i < frame.length; i += 1) {
    const mu = frame[i];
    const magnitude = Math.hypot(mu[0], mu[1], mu[2]);
    if (magnitude < MIN_MU) {
      // A zero-length shaft plus a degenerate cone renders as a stray speck that
      // flickers frame to frame.
      continue;
    }
    const [x, y, z] = coords[i];
    const reach = CONE_OVERSHOOT * scale;
    viewer.addArrow({
      start: { x, y, z },
      end: {
        x: x + reach * mu[0],
        y: y + reach * mu[1],
        z: z + reach * mu[2],
      },
      // 3Dmol places the cone base at start + mid * (end - start), so mid pins it to
      // exactly where PyMOL's CYLINDER ends and its CONE begins.
      mid: 1 / CONE_OVERSHOOT,
      radius: CYL_RADIUS,
      radiusRatio: CONE_RADIUS / CYL_RADIUS,
      color: ARROW_COLOR,
    });
  }
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
  addDipoleArrows(currentSystem.mu_history[currentFrame], currentScale());
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

  // A row is [term, value]; a lone string is a full-width group heading.
  const rows = [
    ["Dataset", dataset],
    ["System", systemName],
    ["Separation", separationLabel],
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
    // "Polarization energy" above them follows the model selector. The heading is what says
    // so, which is why they are grouped rather than appended as four more plain rows.
    ...(referenceEnergies.length === 0
      ? []
      : [
          "SAPT0 reference",
          ...referenceEnergies.map(({ label, kcalmol }) => [
            label,
            `${kcalmol.toFixed(3)} kcal/mol`,
          ]),
        ]),
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

async function loadCatalog() {
  setStatus("Loading system catalog…");
  catalog = await callJson("/api/ipd/systems");
  if (catalog.length === 0) {
    throw new Error("The system catalog is empty. Run scripts/build_ipd_dataset.py.");
  }
  populateDatasetSelector();
  populateSystemSelector();
  populateSeparationSelector();
  populateModelSelector();
}

async function loadSelectedSystem() {
  stopPlayback();

  const systemId = resolveSelectedSystemId();
  if (!systemId) {
    return;
  }

  setStatus("Loading system…");
  const data = await callJson(
    `/api/ipd/system?system_id=${encodeURIComponent(systemId)}` +
      `&model=${encodeURIComponent(modelEl.value)}`,
  );

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
    stripIntermonomerBonds(model, data.n_atoms_A);
    viewer.setStyle(
      {},
      {
        sphere: { scale: SPHERE_SCALE, colorscheme: "greenCarbon" },
        stick: { radius: STICK_RADIUS, colorscheme: "greenCarbon" },
      },
    );
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
  setStatus(
    data.n_frames === 1
      ? `${data.system_name} ${data.separation_label}: converged dipoles, ${data.n_atoms} atoms`
      : `${data.system_name} ${data.separation_label}: ${data.n_frames - 1} SCF iterations, ` +
          `${data.n_atoms} atoms`,
    "ok",
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

// Resize the canvas only -- re-rendering the model here would reset the camera.
window.addEventListener("resize", () => {
  viewer.resize();
});

// The readout is not primed here: baseScale is only known once a system loads, and
// loadSelectedSystem() calls updateScaleReadout() itself.
loadCatalog()
  .then(loadSelectedSystem)
  .catch((err) => setStatus(err.message, "error"));
