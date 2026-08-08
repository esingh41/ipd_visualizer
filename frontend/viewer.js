// The trajectory tab.
//
// Reads the JSON written at upload time and nothing else: frames arrive already grouped,
// ordered and frame-indexed, so this file selects, renders and labels. It never asks the
// server for a single frame -- a whole trajectory is fetched once and kept here, which is
// what makes slider movement local rather than a round trip per step.

const SPHERE_SCALE = 0.15;
const STICK_RADIUS = 0.55 * SPHERE_SCALE;
const FRAME_INTERVAL_MS = 500;

// Arrow geometry, ported from molview.js in the previous frontend, where it was in turn ported
// from _arrow_cgo (pymol_dipole.py) and verified to reproduce PyMOL to floating-point
// precision. Changing these changes the scientific reading of the picture.
const DEFAULT_ARROW_LEN = 2.0; // Angstrom, the longest arrow in the whole trajectory
const MIN_MU = 1e-6; // below this an arrow is skipped entirely
const CONE_OVERSHOOT = 1.3; // total arrow length is CONE_OVERSHOOT * scale * |mu|
const CYL_RADIUS = 0.05;
const CONE_RADIUS = 0.1;

// Deliberately not 0xbf00bf -- that is the induced-dipole colour in the old frontend, and
// these are *permanent* MBIS dipoles. Reusing it would say "induced" about an input quantity.
const DIPOLE_COLORS = {
  monomer: 0x1f77b4,
  dimer: 0xff7f0e,
  delta: 0x9467bd,
};

let viewer = null;
let currentTrajectory = null;
let currentFrameIndex = 0;
let playbackTimer = null;

// Which multipole table is on screen, and which atoms it lists. The selected frame stays the
// primary state -- these only decide how that frame is presented.
let currentMultipoleView = "charges";
let currentAtomFilter = "all";

// Viewer overlays: what is drawn *on* the molecule, as opposed to tabulated beneath it.
// Dipoles are three independent layers because arrows of different colours coexist readably;
// charge labels are one exclusive choice because three numbers per atom would not.
let showMonomerDipoles = false;
let showDimerDipoles = false;
let showDeltaDipoles = false;
let chargeLabelMode = "none";

// Handles for what this layer drew, so it can clear its own work and nothing else. The IPD
// layer will keep its own arrays, which is what stops the two from erasing each other.
let multipoleShapes = [];
let chargeLabels = [];

// Angstrom per atomic unit, fitted once per trajectory -- see setArrowScale. The slider is a
// multiplier on top of it, so the automatic fit stays the reference point and "1.0x" always
// means "longest arrow in this trajectory is DEFAULT_ARROW_LEN".
let baseArrowScale = 1.0;

const el = (id) => document.getElementById(id);

function setStatus(message, isError = false) {
  const node = el("status");
  node.textContent = message;
  node.classList.toggle("error", Boolean(isError));
}

async function callJson(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(body?.error || body?.message || `Request failed: ${response.status}`);
  }
  return body;
}

// --- viewer -----------------------------------------------------------------

// XYZ carries no bond table, so 3Dmol guesses bonds by distance: it bonds two atoms whenever
// they sit closer than r1 + r2 + 0.25 A, which for Na-O is 2.52 A and so invents an Na+---O
// stick at every separation up to 1.10 Re. The two monomers of a dimer are non-bonded by
// construction, so any bond crossing the A/B boundary is an artefact of that guess.
//
// Ported from molview.js in the previous frontend, where it was worked out originally.
// Bonds are assigned once when the model is parsed and re-read from atom.bonds on every
// setStyle, so deleting them here -- before the first setStyle -- is all it takes.
function stripIntermonomerBonds(model, nAtomsA, nAtoms) {
  if (!nAtomsA) {
    // No declared monomer split: there is no boundary to enforce, and inventing one would be
    // worse than leaving 3Dmol's guess alone.
    return true;
  }

  // An empty selection returns the model's atoms, by reference, in the same index order that
  // atom.bonds refers to. The XYZ parser does not set atom.index, so position *is* the index.
  const atoms = model.selectedAtoms({});
  if (atoms.length !== nAtoms) {
    return false;
  }

  const inMonomerA = (index) => index < nAtomsA;
  atoms.forEach((atom, index) => {
    const bonds = [];
    const bondOrder = [];
    for (let i = 0; i < atom.bonds.length; i += 1) {
      if (inMonomerA(atom.bonds[i]) === inMonomerA(index)) {
        bonds.push(atom.bonds[i]);
        bondOrder.push(atom.bondOrder[i]);
      }
    }
    // Every atom is filtered, so both halves of a crossing bond go: a bond listed by only one
    // of its two atoms would still render, as a half-length stick.
    atom.bonds = bonds;
    atom.bondOrder = bondOrder;
  });
  return true;
}

function renderFrame(frame, { resetCamera }) {
  if (!viewer || !frame.xyz) {
    return;
  }
  viewer.removeAllModels();
  const model = viewer.addModel(frame.xyz, "xyz");
  stripIntermonomerBonds(model, frame.n_atoms_A, frame.n_atoms);
  viewer.setStyle(
    {},
    { sphere: { scale: SPHERE_SCALE }, stick: { radius: STICK_RADIUS } }
  );
  // Only on the first frame of a trajectory. Zooming on every slider step throws the camera
  // away mid-scrub, which is disorienting and hides what actually changed.
  if (resetCamera) {
    viewer.zoomTo();
  }
  viewer.render();
}

// --- frames -----------------------------------------------------------------

function frameLabel(frame) {
  // separation_label is formatted server-side with its real units -- "0.70 Re" for a ratio,
  // "2.180 A" for a distance -- and is never empty. eq_ratio and contact_distance_ang are
  // both null for frames where they could not be derived, so neither can be formatted here
  // without a guard.
  const parts = [`Frame ${frame.frame_index + 1} / ${currentTrajectory.frames.length}`];
  if (frame.separation_label) {
    parts.push(frame.separation_label);
  }
  if (typeof frame.contact_distance_ang === "number") {
    parts.push(`closest contact ${frame.contact_distance_ang.toFixed(2)} Å`);
  }
  return parts.join("  |  ");
}

// The single entry point for everything that depends on the selected frame. Energies and the
// IPD controls attach here too.
function showFrame(index, { resetCamera = false } = {}) {
  const frame = currentTrajectory?.frames[index];
  if (!frame) {
    return;
  }
  currentFrameIndex = index;
  renderFrame(frame, { resetCamera });
  el("frame-label").textContent = frameLabel(frame);
  renderMultipoleSection(frame);
  updateOverlayControls(frame);
  renderMultipoleOverlays(frame);
}

// Every *user-driven* frame change goes through here: the slider today, clickable timeline
// dots and plot points later. Playback calls showFrame directly instead -- this pauses, so a
// running animation would stop itself on its first tick.
function selectFrame(index) {
  pausePlayback();
  el("frame-slider").value = index;
  showFrame(index);
}

function setTrajectory(trajectory) {
  // Before anything else: a running timer holds an index into the *previous* frames array,
  // and the new one may be shorter.
  pausePlayback();

  currentTrajectory = trajectory;
  currentFrameIndex = 0;
  // Fixed once per trajectory, before the first frame is drawn.
  setArrowScale(trajectory);

  const animatable = trajectory.frames.length > 1;
  const slider = el("frame-slider");
  slider.min = 0;
  slider.max = Math.max(0, trajectory.frames.length - 1);
  slider.value = 0;
  slider.disabled = !animatable;
  el("play-button").disabled = !animatable;

  showFrame(0, { resetCamera: true });
}

// --- playback ---------------------------------------------------------------

function setPlayButton(playing) {
  const button = el("play-button");
  button.textContent = playing ? "❚❚" : "▶";
  button.setAttribute("aria-label", playing ? "Pause" : "Play");
}

function pausePlayback() {
  if (playbackTimer !== null) {
    clearInterval(playbackTimer);
    playbackTimer = null;
  }
  setPlayButton(false);
}

function startPlayback() {
  if (playbackTimer !== null || !currentTrajectory || currentTrajectory.frames.length < 2) {
    return;
  }
  setPlayButton(true);
  playbackTimer = setInterval(() => {
    const next = (currentFrameIndex + 1) % currentTrajectory.frames.length;
    el("frame-slider").value = next;
    showFrame(next);
  }, FRAME_INTERVAL_MS);
}

function togglePlayback() {
  if (playbackTimer === null) {
    startPlayback();
  } else {
    pausePlayback();
  }
}

// --- multipoles -------------------------------------------------------------

// Each frame arrives with its monomer arrays already joined A-then-B and length-checked
// against the atom count, so atom i of the geometry, of the monomer array and of the dimer
// array are the same atom. Nothing here concatenates, and no dataframe column name appears.

const MISSING = "—";

const MULTIPOLE_LABELS = {
  charges: "MBIS charges",
  dipoles: "Atomic dipoles",
  volume_ratios: "Volume ratios",
};

function selectedFrame() {
  return currentTrajectory?.frames[currentFrameIndex] ?? null;
}

function atomLabel(frame, index) {
  return `${frame.symbols[index]}${index + 1}`;
}

// n_atoms_A is the only encoding of monomer membership: atom i is in A iff i < n_atoms_A.
function atomMonomer(frame, index) {
  return index < frame.n_atoms_A ? "A" : "B";
}

function atomVisible(frame, index) {
  if (currentAtomFilter === "all") {
    return true;
  }
  return atomMonomer(frame, index) === currentAtomFilter;
}

function num(value, signed = false) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return MISSING;
  }
  return signed && value >= 0 ? `+${value.toFixed(4)}` : value.toFixed(4);
}

function appendRow(parent, tag, cells) {
  const row = document.createElement("tr");
  for (const cell of cells) {
    const node = document.createElement(tag);
    if (typeof cell === "string") {
      node.textContent = cell;
    } else {
      node.textContent = cell.text;
      if (cell.className) node.className = cell.className;
      if (cell.colSpan) node.colSpan = cell.colSpan;
      if (cell.rowSpan) node.rowSpan = cell.rowSpan;
    }
    row.append(node);
  }
  parent.append(row);
}

function showMultipoleUnavailable(message) {
  const node = el("multipole-message");
  node.textContent = message;
  node.hidden = false;
}

// Charges and volume ratios: one number per atom, so monomer, dimer and their difference are
// each a single column.
function renderScalarTable(frame, data, heading) {
  appendRow(el("multipole-table-head"), "th", [
    "Atom",
    "Monomer",
    { text: `${heading} monomer`, className: "numeric group-start" },
    { text: `${heading} dimer`, className: "numeric" },
    { text: "Δ", className: "numeric" },
  ]);

  const body = el("multipole-table-body");
  for (let i = 0; i < frame.n_atoms; i += 1) {
    if (!atomVisible(frame, i)) {
      continue;
    }
    const monomer = data.monomer?.[i];
    const dimer = data.dimer?.[i];
    const bothPresent = typeof monomer === "number" && typeof dimer === "number";
    appendRow(body, "td", [
      atomLabel(frame, i),
      atomMonomer(frame, i),
      { text: num(monomer), className: "numeric group-start" },
      { text: num(dimer), className: "numeric" },
      { text: bothPresent ? num(dimer - monomer, true) : MISSING, className: "numeric delta" },
    ]);
  }
}

// A dipole is a vector, so each group shows x, y, z and the norm *of those three numbers*.
// That makes the Δ group's norm |Δμ| -- the size of the actual change -- which is not the
// same as the change in size: a dipole that rotates without growing has |Δμ| > 0 while
// |μ_dimer| - |μ_monomer| is ~0. Both readings are on screen because both components and
// norms are.
function vectorCells(vector, signed = false) {
  const components = vector ?? [null, null, null];
  return [
    ...components.map((value, index) => ({
      text: num(value, signed),
      // The x column opens a group, and gets the rule that separates it from the one before.
      className: index === 0 ? "numeric group-start" : "numeric",
    })),
    { text: vector ? num(Math.hypot(...vector)) : MISSING, className: "numeric magnitude" },
  ];
}

function renderDipoleTable(frame, data) {
  const head = el("multipole-table-head");
  appendRow(head, "th", [
    { text: "Atom", rowSpan: 2 },
    { text: "Monomer", rowSpan: 2 },
    { text: "μ monomer (a.u.)", colSpan: 4, className: "group" },
    { text: "μ dimer (a.u.)", colSpan: 4, className: "group" },
    { text: "Δμ = dimer − monomer", colSpan: 4, className: "group" },
  ]);
  appendRow(
    head,
    "th",
    ["x", "y", "z", "|μ|", "x", "y", "z", "|μ|", "x", "y", "z", "|Δμ|"].map((text, index) => ({
      text,
      className: index % 4 === 0 ? "numeric group-start" : "numeric",
    }))
  );

  const body = el("multipole-table-body");
  for (let i = 0; i < frame.n_atoms; i += 1) {
    if (!atomVisible(frame, i)) {
      continue;
    }
    const monomer = data.monomer?.[i] ?? null;
    const dimer = data.dimer?.[i] ?? null;
    const delta = monomer && dimer ? dimer.map((value, k) => value - monomer[k]) : null;
    appendRow(body, "td", [
      atomLabel(frame, i),
      atomMonomer(frame, i),
      ...vectorCells(monomer),
      ...vectorCells(dimer),
      ...vectorCells(delta, true),
    ]);
  }
}

// Decides which table to show. Never touches the viewer or the selected frame.
function renderMultipoleSection(frame) {
  el("multipole-table-head").innerHTML = "";
  el("multipole-table-body").innerHTML = "";
  el("multipole-message").hidden = true;

  if (!frame.n_atoms) {
    showMultipoleUnavailable("This frame has no usable geometry.");
    return;
  }
  if (!frame.multipoles) {
    showMultipoleUnavailable("No multipole data is available for this frame.");
    return;
  }

  const data = frame.multipoles[currentMultipoleView];
  if (!data) {
    showMultipoleUnavailable(
      `${MULTIPOLE_LABELS[currentMultipoleView]} are not available for this frame.`
    );
    return;
  }

  if (currentMultipoleView === "dipoles") {
    renderDipoleTable(frame, data);
  } else {
    renderScalarTable(frame, data, currentMultipoleView === "charges" ? "q" : "ratio");
  }
}

function setActiveButton(group, key, value) {
  for (const button of group.querySelectorAll("button")) {
    button.classList.toggle("active", button.dataset[key] === value);
  }
}

// Switching quantity or filter re-renders the table only -- the molecule is unchanged, and
// re-rendering it would throw the camera away for nothing.
function selectMultipoleView(view) {
  currentMultipoleView = view;
  setActiveButton(el("multipole-views"), "view", view);
  const frame = selectedFrame();
  if (frame) {
    renderMultipoleSection(frame);
  }
}

function setAtomFilter(filter) {
  currentAtomFilter = filter;
  setActiveButton(el("atom-filters"), "filter", filter);
  const frame = selectedFrame();
  if (frame) {
    renderMultipoleSection(frame);
  }
}

// --- viewer overlays --------------------------------------------------------

// What gets drawn *on* the molecule. A separate layer from renderFrame, which stays geometry
// only, and from the table, which stays numbers only. The IPD arrows will be a fourth layer
// with its own handles, so the two never clear each other's work.

function deltaDipoles(frame) {
  const dipoles = frame.multipoles?.dipoles;
  const monomer = dipoles?.monomer;
  const dimer = dipoles?.dimer;
  if (!monomer || !dimer) {
    return null;
  }
  return monomer.map((mu, i) => [dimer[i][0] - mu[0], dimer[i][1] - mu[1], dimer[i][2] - mu[2]]);
}

function dipoleVectors(frame, layer) {
  if (layer === "delta") {
    return deltaDipoles(frame);
  }
  return frame.multipoles?.dipoles?.[layer] ?? null;
}

function chargeValues(frame, mode) {
  const charges = frame.multipoles?.charges;
  if (!charges) {
    return null;
  }
  if (mode === "delta") {
    if (!charges.monomer || !charges.dimer) {
      return null;
    }
    return charges.dimer.map((value, i) => value - charges.monomer[i]);
  }
  return charges[mode] ?? null;
}

// One scale for the whole trajectory, spanning all three layers together, so arrow lengths
// stay comparable between frames *and* between monomer, dimer and delta. Scaling each layer
// to its own maximum would draw a small change at the same length as a large dipole.
//
// Derived here rather than serialized: the whole trajectory is already in the browser.
function setArrowScale(trajectory) {
  let largest = 0;
  for (const frame of trajectory.frames) {
    for (const layer of ["monomer", "dimer", "delta"]) {
      for (const vector of dipoleVectors(frame, layer) ?? []) {
        if (vector) {
          largest = Math.max(largest, Math.hypot(...vector));
        }
      }
    }
  }
  baseArrowScale = largest > 0 ? DEFAULT_ARROW_LEN / largest : 1.0;
  updateArrowScaleReadout();
}

function currentArrowScale() {
  return baseArrowScale * Number(el("arrow-scale").value);
}

// Both numbers, because neither alone is enough: the multiplier says how far from the
// automatic fit you are, and the absolute scale is what makes an arrow length mean something.
function updateArrowScaleReadout() {
  const multiplier = Number(el("arrow-scale").value);
  el("arrow-scale-readout").textContent =
    `${multiplier.toFixed(1)}× (${currentArrowScale().toFixed(1)} Å per a.u.)`;
}

function addDipoleArrows(coords, vectors, color) {
  for (let i = 0; i < vectors.length; i += 1) {
    const vector = vectors[i];
    if (!vector || !coords[i]) {
      continue;
    }
    if (Math.hypot(...vector) < MIN_MU) {
      // Not defensive: an isolated Na+ has no atomic dipole at all, so every water fixture
      // carries one exactly-zero vector per frame. A zero-length shaft plus a degenerate cone
      // renders as a stray speck that flickers frame to frame.
      continue;
    }
    const [x, y, z] = coords[i];
    const reach = CONE_OVERSHOOT * currentArrowScale();
    multipoleShapes.push(
      viewer.addArrow({
        start: { x, y, z },
        end: {
          x: x + reach * vector[0],
          y: y + reach * vector[1],
          z: z + reach * vector[2],
        },
        // 3Dmol places the cone base at start + mid * (end - start), so mid pins it to exactly
        // where PyMOL's CYLINDER ends and its CONE begins.
        mid: 1 / CONE_OVERSHOOT,
        radius: CYL_RADIUS,
        radiusRatio: CONE_RADIUS / CYL_RADIUS,
        color,
      })
    );
  }
}

function formatCharge(value) {
  return value >= 0 ? `+${value.toFixed(2)}` : value.toFixed(2);
}

function renderChargeLabels(frame, mode) {
  const values = chargeValues(frame, mode);
  if (!values) {
    return;
  }
  // frame.coords, not the 3Dmol model's atoms: this is the array the multipole values are
  // indexed against, so label i is guaranteed to sit on the atom whose charge is values[i].
  frame.coords.forEach(([x, y, z], index) => {
    const value = values[index];
    if (typeof value !== "number") {
      return;
    }
    chargeLabels.push(
      viewer.addLabel(
        formatCharge(value),
        {
          position: { x, y, z },
          showBackground: false,
          // 3Dmol's default label text is white, and so is the viewer background.
          fontColor: "black",
          fontOpacity: 1,
          fontSize: 16,
          // Undocumented but real: Label.setContext does `if (style.bold) bold = "bold "` and
          // prepends it to the canvas font string.
          bold: true,
          // Without this the label anchors at its top-left corner and hangs up and to the
          // left of the atom, which at these bond lengths reads as belonging to a neighbour.
          alignment: "center",
          inFront: true,
        },
        undefined,
        true
      )
    );
  });
}

function clearMultipoleOverlays() {
  multipoleShapes.forEach((shape) => viewer.removeShape(shape));
  chargeLabels.forEach((label) => viewer.removeLabel(label));
  multipoleShapes = [];
  chargeLabels = [];
}

function renderMultipoleOverlays(frame) {
  if (!viewer) {
    return;
  }
  clearMultipoleOverlays();

  if (frame?.coords) {
    const layers = [
      ["monomer", showMonomerDipoles],
      ["dimer", showDimerDipoles],
      ["delta", showDeltaDipoles],
    ];
    for (const [layer, visible] of layers) {
      const vectors = visible ? dipoleVectors(frame, layer) : null;
      if (vectors) {
        addDipoleArrows(frame.coords, vectors, DIPOLE_COLORS[layer]);
      }
    }
    if (chargeLabelMode !== "none") {
      renderChargeLabels(frame, chargeLabelMode);
    }
  }

  viewer.render();
}

// A control for data this frame does not have is disabled rather than hidden, so the reason
// the viewer is empty is visible.
function updateOverlayControls(frame) {
  const dipoles = frame?.multipoles?.dipoles;
  el("show-monomer-dipoles").disabled = !dipoles?.monomer;
  el("show-dimer-dipoles").disabled = !dipoles?.dimer;
  el("show-delta-dipoles").disabled = !(dipoles?.monomer && dipoles?.dimer);

  const charges = frame?.multipoles?.charges;
  const available = {
    none: true,
    monomer: Boolean(charges?.monomer),
    dimer: Boolean(charges?.dimer),
    delta: Boolean(charges?.monomer && charges?.dimer),
  };
  for (const option of el("charge-label-mode").options) {
    option.disabled = !available[option.value];
  }
}

// Toggling an overlay redraws overlays only. Going through showFrame would destroy and rebuild
// the molecule, throwing the camera away for a change that does not touch the geometry.
function setMonomerDipolesVisible(visible) {
  showMonomerDipoles = visible;
  renderMultipoleOverlays(selectedFrame());
}

function setDimerDipolesVisible(visible) {
  showDimerDipoles = visible;
  renderMultipoleOverlays(selectedFrame());
}

function setDeltaDipolesVisible(visible) {
  showDeltaDipoles = visible;
  renderMultipoleOverlays(selectedFrame());
}

function setChargeLabelMode(mode) {
  chargeLabelMode = mode;
  renderMultipoleOverlays(selectedFrame());
}

// --- loading ----------------------------------------------------------------

function fillSelect(select, options, { placeholder }) {
  select.innerHTML = "";
  if (!options.length) {
    select.append(new Option(placeholder, ""));
    select.disabled = true;
    return;
  }
  options.forEach(({ label, value }) => select.append(new Option(label, value)));
  select.disabled = false;
}

async function loadTrajectory() {
  const uploadId = el("collection-select").value;
  const slug = el("system-select").value;
  if (!uploadId || !slug) {
    return;
  }
  setStatus("Loading trajectory…");
  const trajectory = await callJson(
    `/api/uploads/${encodeURIComponent(uploadId)}/systems/${encodeURIComponent(slug)}/trajectory`
  );
  setTrajectory(trajectory);
  setStatus(`${trajectory.system} — ${trajectory.n_frames} frames`);
}

async function loadSystems() {
  const uploadId = el("collection-select").value;
  if (!uploadId) {
    return;
  }
  const listing = await callJson(`/api/uploads/${encodeURIComponent(uploadId)}/systems`);
  fillSelect(
    el("system-select"),
    listing.systems.map((system) => ({
      label: `${system.system_id} (${system.frame_count} frames)`,
      value: system.slug,
    })),
    { placeholder: "No systems" }
  );
  await loadTrajectory();
}

async function loadCollections(selectId) {
  const { uploads } = await callJson("/api/uploads");
  fillSelect(
    el("collection-select"),
    uploads.map((upload) => ({
      label: `${upload.display_name} (${upload.frame_count} frames)`,
      value: upload.collection_id,
    })),
    { placeholder: "Upload a dataframe to begin" }
  );
  if (selectId) {
    el("collection-select").value = selectId;
  }
  if (uploads.length) {
    await loadSystems();
  } else {
    setStatus("No uploads yet.");
  }
}

async function uploadFile(file) {
  setStatus(`Uploading ${file.name}…`);
  const body = new FormData();
  body.append("file", file);
  const manifest = await callJson("/api/uploads", { method: "POST", body });

  const dropped = manifest.validation?.dropped_rows ?? [];
  if (dropped.length) {
    setStatus(`${manifest.display_name}: ${dropped.length} row(s) had no usable geometry.`);
  }
  await loadCollections(manifest.collection_id);
}

// --- wiring -----------------------------------------------------------------

function guard(handler) {
  return (event) =>
    Promise.resolve(handler(event)).catch((error) => setStatus(error.message, true));
}

function init() {
  viewer = window.$3Dmol.createViewer(el("viewer"), { backgroundColor: "white" });

  el("file-input").addEventListener(
    "change",
    guard(async (event) => {
      const [file] = event.target.files;
      if (file) {
        await uploadFile(file);
      }
      event.target.value = "";
    })
  );

  el("collection-select").addEventListener("change", guard(loadSystems));
  el("system-select").addEventListener("change", guard(loadTrajectory));
  el("play-button").addEventListener("click", togglePlayback);
  // selectFrame pauses: a timer still advancing under the cursor fights the drag and lands
  // somewhere the user did not choose.
  el("frame-slider").addEventListener("input", (event) => {
    selectFrame(Number(event.target.value));
  });

  el("multipole-views").addEventListener("click", (event) => {
    const { view } = event.target.dataset;
    if (view) {
      selectMultipoleView(view);
    }
  });
  el("atom-filters").addEventListener("click", (event) => {
    const { filter } = event.target.dataset;
    if (filter) {
      setAtomFilter(filter);
    }
  });

  el("show-monomer-dipoles").addEventListener("change", (event) => {
    setMonomerDipolesVisible(event.target.checked);
  });
  el("show-dimer-dipoles").addEventListener("change", (event) => {
    setDimerDipolesVisible(event.target.checked);
  });
  el("show-delta-dipoles").addEventListener("change", (event) => {
    setDeltaDipolesVisible(event.target.checked);
  });
  el("charge-label-mode").addEventListener("change", (event) => {
    setChargeLabelMode(event.target.value);
  });
  // Overlays only. The old page called showFrame here, but it had no overlay layer to redraw
  // on its own; rebuilding the molecule would throw the camera away mid-drag.
  el("arrow-scale").addEventListener("input", () => {
    updateArrowScaleReadout();
    renderMultipoleOverlays(selectedFrame());
  });

  guard(loadCollections)();
}

init();
