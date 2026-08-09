// The trajectory tab, and the page shell around it.
//
// Reads the JSON written at upload time and nothing else: frames arrive already grouped,
// ordered and frame-indexed, so this file selects, renders and labels. It never asks the
// server for a single frame -- a whole trajectory is fetched once and kept here, which is
// what makes slider movement local rather than a round trip per step.
//
// It also owns the collection/system selectors, which sit above the tabs and are shared, and
// hands each loaded trajectory to plots.js. A direct call rather than an event bus: there are
// exactly two participants, and the alternative would be indirection for its own sake.

import * as plots from "/plots.js";

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

// IPD: a secondary analysis of the selected frame, in its own viewer. Everything here is
// scoped to one physical frame and one damping mode; changing either resets all of it.
let ipdViewer = null;
let currentIpdMode = null;
let currentIpdIteration = 0;
let ipdPlaybackTimer = null;
let ipdShapes = [];
let ipdBaseArrowScale = 1.0;

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

// A WebGL canvas measures its container, and a container inside a closed <details> is 0x0.
// Resizing to that collapses the canvas, and it stays collapsed once the panel is opened -- so
// every path that measures has to check first. Ported from molview.js, where the IPD viewer's
// lazy initialisation is exactly the case that needed it.
function isLaidOut(container) {
  return (
    Boolean(container) &&
    !container.closest("[hidden]") &&
    container.clientWidth > 0 &&
    container.clientHeight > 0
  );
}

// One geometry renderer, two named entry points. renderFrame and renderIpdFrame stay distinct
// -- they are the two halves of the design rule, and each one's caller means something
// different by it -- but the body lives once, because two copies of the bond-stripping and
// styling would eventually disagree and the disagreement would be a wrong picture.
function drawGeometry(target, frame, { resetCamera }) {
  if (!target || !frame?.xyz) {
    return;
  }
  target.removeAllModels();
  const model = target.addModel(frame.xyz, "xyz");
  stripIntermonomerBonds(model, frame.n_atoms_A, frame.n_atoms);
  target.setStyle(
    {},
    { sphere: { scale: SPHERE_SCALE }, stick: { radius: STICK_RADIUS } }
  );
  // Only on the first frame of a trajectory. Zooming on every step throws the camera away
  // mid-scrub, which is disorienting and hides what actually changed.
  if (resetCamera) {
    target.zoomTo();
  }
  target.render();
}

// The main viewer: geometry and MBIS analysis.
function renderFrame(frame, { resetCamera }) {
  drawGeometry(viewer, frame, { resetCamera });
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
  // The physical frame is changing, so anything scoped to the *previous* frame's SCF run is
  // now meaningless: a running iteration animation, and the iteration index itself.
  stopIpdPlayback();
  currentFrameIndex = index;
  currentIpdIteration = 0;

  renderFrame(frame, { resetCamera });
  el("frame-label").textContent = frameLabel(frame);
  renderTimelineSelection();
  renderMultipoleSection(frame);
  updateOverlayControls(frame);
  renderMultipoleOverlays(frame);

  // Synchronous, and deliberately: showFrame runs on every playback tick, so anything here
  // that fetched would fire once per frame per lap. renderIpdSection reads frame.ipd, which
  // is already in the browser; histories load only on an explicit action.
  renderIpdSection(frame);
  if (ipdViewer && isIpdSectionOpen()) {
    renderIpdFrame(frame, { resetCamera });
    // The history for the new frame is a fetch, so it is deliberately not awaited here and
    // deliberately not attempted during playback: at 500 ms a frame that would be one request
    // per frame per lap. Scrubbing by hand loads each frame once and then reads the cache.
    if (playbackTimer === null) {
      guard(() => selectIpdMode(currentIpdMode))();
    }
  }
}

// Every *user-driven* frame change goes through here: timeline dots today, plot points later.
// Playback calls showFrame directly instead -- this pauses, so a running animation would stop
// itself on its first tick.
function selectFrame(index) {
  pausePlayback();
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

  el("play-button").disabled = trajectory.frames.length < 2;
  renderTrajectoryTimeline(trajectory);

  showFrame(0, { resetCamera: true });

  // The same payload, re-sliced by separation instead of by frame. Handed over rather than
  // re-fetched, so the two tabs cannot be looking at different copies of one system.
  plots.setTrajectory(trajectory);
}

// --- trajectory timeline ----------------------------------------------------

// A physical geometry per dot. Distinct in meaning from the IPD timeline below, which holds
// geometry fixed and steps through the SCF -- these change what molecule is on screen.
function renderTrajectoryTimeline(trajectory) {
  const timeline = el("frame-timeline");
  timeline.replaceChildren();

  trajectory.frames.forEach((frame, index) => {
    const dot = document.createElement("button");
    dot.type = "button";
    dot.className = "timeline-dot";
    dot.dataset.frame = String(index);
    // separation_label is formatted server-side with its real units and is never empty.
    dot.title = frame.separation_label || `Frame ${index + 1}`;
    dot.setAttribute("aria-label", `Select frame ${index + 1}: ${dot.title}`);
    dot.addEventListener("click", () => selectFrame(index));
    timeline.append(dot);
  });

  renderTimelineSelection();
}

function renderTimelineSelection() {
  for (const dot of el("frame-timeline").querySelectorAll(".timeline-dot")) {
    if (Number(dot.dataset.frame) === currentFrameIndex) {
      dot.setAttribute("aria-current", "true");
    } else {
      dot.removeAttribute("aria-current");
    }
  }
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
    showFrame((currentFrameIndex + 1) % currentTrajectory.frames.length);
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

// Two viewers, two independent sliders, one rule for combining a fitted base with a multiplier.
function arrowScale(base, sliderId) {
  return base * Number(el(sliderId).value);
}

function currentArrowScale() {
  return arrowScale(baseArrowScale, "arrow-scale");
}

// Both numbers, because neither alone is enough: the multiplier says how far from the
// automatic fit you are, and the absolute scale is what makes an arrow length mean something.
function updateArrowScaleReadout() {
  const multiplier = Number(el("arrow-scale").value);
  el("arrow-scale-readout").textContent =
    `${multiplier.toFixed(1)}× (${currentArrowScale().toFixed(1)} Å per a.u.)`;
}

// Takes its viewer, its scale and the array to record handles in, so one arrow convention
// serves both viewers. molview.js in the previous frontend was written this way for the same
// reason -- "every function takes its viewer, so a page may own more than one" -- and the
// PyMOL-faithful geometry below must exist exactly once whatever draws it.
function addDipoleArrows(target, coords, vectors, color, scale, sink) {
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
    const reach = CONE_OVERSHOOT * scale;
    sink.push(
      target.addArrow({
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
        addDipoleArrows(
          viewer,
          frame.coords,
          vectors,
          DIPOLE_COLORS[layer],
          currentArrowScale(),
          multipoleShapes
        );
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

// --- IPD --------------------------------------------------------------------

// The second viewer, and the second timeline. The distinction is the whole point of this
// section: the trajectory timeline changes which molecule is on screen, while the IPD timeline
// holds geometry fixed and steps through the SCF that ran at that geometry.
//
// Induced dipoles are *results*; the MBIS dipoles in the main viewer are inputs. They get
// different colours and different viewers so the two can never be read as the same quantity.

const IPD_ARROW_COLOR = 0xbf00bf; // the induced-dipole colour, kept from the previous frontend
const IPD_INTERVAL_MS = 200; // faster than the trajectory: an SCF is many more steps
const MAX_DOTS = 40; // beyond this the timeline degrades to a range input

function isIpdSectionOpen() {
  return el("ipd-section").open;
}

// The IPD viewer is created on first open, never at startup. A WebGL canvas measures its
// container, and a container inside a closed <details> is 0x0 -- creating it there yields a
// collapsed canvas that stays collapsed after the panel opens.
function initializeIpdViewer() {
  if (ipdViewer || !isLaidOut(el("ipd-viewer"))) {
    return;
  }
  ipdViewer = window.$3Dmol.createViewer(el("ipd-viewer"), { backgroundColor: "white" });

  const frame = selectedFrame();
  if (frame) {
    renderIpdFrame(frame, { resetCamera: true });
  }
}

// The IPD viewer: fixed geometry, induced-dipole convergence. Cameras are deliberately not
// synchronized with the main viewer -- independent orientations are simpler and often useful.
function renderIpdFrame(frame, { resetCamera = false } = {}) {
  // Before the models go, not after: removeAllModels does not touch shapes, so dropping the
  // handles without removing them would leave the previous frame's arrows floating over the
  // new geometry with nothing left able to clear them.
  clearIpdDipoles();
  drawGeometry(ipdViewer, frame, { resetCamera });
}

// --- IPD availability -------------------------------------------------------

function ipdModes(frame) {
  return frame?.ipd?.modes ?? [];
}

function ipdMode(frame, id) {
  return ipdModes(frame).find((mode) => mode.id === id) ?? null;
}

// The mode to show for a frame. A mode the user has actually chosen is kept even when it has
// no result here -- that is precisely the state the Compute button exists for, and silently
// switching back to a stored mode would make the uncomputed one unreachable. Only an unset or
// unrecognised selection falls back, preferring one that has something to show.
function preferredIpdMode(frame) {
  if (ipdMode(frame, currentIpdMode)) {
    return currentIpdMode;
  }
  const modes = ipdModes(frame);
  return modes.find((mode) => mode.stored)?.id ?? modes[0]?.id ?? null;
}

function currentIpdHistory() {
  return selectedFrame()?.ipd?.history?.[currentIpdMode] ?? null;
}

// Manages availability text, the compute controls, the mode selector and panel visibility.
// Synchronous by contract: showFrame calls it on every playback tick.
function renderIpdSection(frame) {
  const message = el("ipd-message");
  const controls = el("ipd-controls");
  const display = el("ipd-display");
  const modes = ipdModes(frame);

  if (!frame?.ipd || !frame.ipd.computable) {
    // State A. Said rather than hidden: an empty panel does not explain itself.
    message.textContent =
      "IPD is unavailable for this frame because the required multipole inputs are missing.";
    message.hidden = false;
    controls.hidden = true;
    display.hidden = true;
    return;
  }

  currentIpdMode = preferredIpdMode(frame);
  fillIpdModeSelect(frame);
  controls.hidden = false;

  const selected = ipdMode(frame, currentIpdMode);
  const stored = Boolean(selected?.stored);

  // Compute is offered per mode, not per frame: one mode can be stored while the other is not.
  el("ipd-compute").disabled = stored;
  el("ipd-compute").textContent = stored ? "Computed" : "Compute IPD";

  if (selected?.problem) {
    // The backend found a stored history that does not describe this geometry. Worth naming --
    // it is data damage, not an absence.
    message.textContent = `Stored ${selected.label} result cannot be used: ${selected.problem}.`;
    message.hidden = false;
    display.hidden = true;
    return;
  }

  if (!stored) {
    // State B.
    message.textContent = "No stored IPD calculation for this frame.";
    message.hidden = false;
    display.hidden = true;
    return;
  }

  // State C.
  message.hidden = true;
  display.hidden = false;
  el("ipd-play").disabled = (selected.iteration_count ?? 0) < 2;
  // Only now does the viewer's container have a size. Opening the panel is not enough: in
  // states A and B it sits inside a hidden div and measures 0x0, so this is the first moment
  // 3Dmol can be given a canvas that will not be born collapsed.
  initializeIpdViewer();
}

function fillIpdModeSelect(frame) {
  const select = el("ipd-mode");
  const modes = ipdModes(frame);
  select.replaceChildren();
  for (const mode of modes) {
    // Every mode is listed whether or not it has a result, so the selector does not change
    // shape as frames are walked and an uncomputed mode can still be chosen and computed.
    const option = new Option(mode.stored ? mode.label : `${mode.label} — not computed`, mode.id);
    select.append(option);
  }
  select.value = currentIpdMode ?? "";
  select.disabled = modes.length === 0;
}

// --- IPD history ------------------------------------------------------------

// Cached on the frame object, which is ours: the whole trajectory already lives in the browser,
// so a mode revisited on a frame already seen costs nothing. The cache is not persistence --
// the server holds the computed result, this only avoids re-fetching it.
async function loadIpdHistory(frame, mode) {
  frame.ipd.history ??= {};
  if (frame.ipd.history[mode]) {
    return frame.ipd.history[mode];
  }
  const uploadId = el("collection-select").value;
  const slug = el("system-select").value;
  const history = await callJson(
    `/api/uploads/${encodeURIComponent(uploadId)}/systems/${encodeURIComponent(slug)}` +
      `/frames/${frame.frame_index}/ipd?mode=${encodeURIComponent(mode)}`
  );
  frame.ipd.history[mode] = history;
  return history;
}

// One scale for the whole history, so arrow length is comparable between iterations and the
// convergence is visible as arrows settling rather than as a rescaling.
function setIpdArrowScale(history) {
  let largest = 0;
  for (const iteration of history.mu_history) {
    for (const vector of iteration) {
      if (vector) {
        largest = Math.max(largest, Math.hypot(...vector));
      }
    }
  }
  ipdBaseArrowScale = largest > 0 ? DEFAULT_ARROW_LEN / largest : 1.0;
  updateIpdArrowScaleReadout();
}

function currentIpdArrowScale() {
  return arrowScale(ipdBaseArrowScale, "ipd-arrow-scale");
}

function updateIpdArrowScaleReadout() {
  const multiplier = Number(el("ipd-arrow-scale").value);
  el("ipd-arrow-scale-readout").textContent =
    `${multiplier.toFixed(1)}× (${currentIpdArrowScale().toFixed(1)} Å per a.u.)`;
}

async function selectIpdMode(mode) {
  stopIpdPlayback();
  clearIpdDipoles();
  currentIpdMode = mode;
  currentIpdIteration = 0;

  const frame = selectedFrame();
  if (!frame || !ipdMode(frame, mode)?.stored) {
    // Nothing to show for this mode. The old mode's timeline would otherwise sit there in the
    // hidden panel and reappear, belonging to a calculation no longer selected.
    el("ipd-timeline").replaceChildren();
    el("ipd-iteration-label").textContent = "";
    renderIpdSection(frame);
    return;
  }

  renderIpdSection(frame);
  const history = await loadIpdHistory(frame, mode);
  setIpdArrowScale(history);
  renderIpdTimeline(history.mu_history.length);
  showIpdIteration(0);
}

// --- IPD iteration timeline -------------------------------------------------

// A history that ran to the SCF cap is 201 entries, which is not a timeline any more -- past
// MAX_DOTS this becomes a range input. Both branches call selectIpdIteration, so the two
// shapes are one behaviour.
function renderIpdTimeline(iterationCount) {
  const timeline = el("ipd-timeline");
  timeline.replaceChildren();

  if (iterationCount > MAX_DOTS) {
    const range = document.createElement("input");
    range.type = "range";
    range.className = "timeline-range";
    range.min = "0";
    range.max = String(iterationCount - 1);
    range.step = "1";
    range.value = "0";
    range.setAttribute("aria-label", "SCF iteration");
    range.addEventListener("input", () => selectIpdIteration(Number(range.value)));
    timeline.append(range);
    return;
  }

  for (let index = 0; index < iterationCount; index += 1) {
    const dot = document.createElement("button");
    dot.type = "button";
    dot.className = "timeline-dot";
    dot.dataset.iteration = String(index);
    dot.title = ipdIterationTooltip(index, iterationCount);
    dot.setAttribute("aria-label", `Select IPD iteration ${index}`);
    dot.addEventListener("click", () => selectIpdIteration(index));
    timeline.append(dot);
  }
}

function renderIpdTimelineSelection() {
  const timeline = el("ipd-timeline");
  const range = timeline.querySelector(".timeline-range");
  if (range) {
    range.value = String(currentIpdIteration);
    return;
  }
  for (const dot of timeline.querySelectorAll(".timeline-dot")) {
    if (Number(dot.dataset.iteration) === currentIpdIteration) {
      dot.setAttribute("aria-current", "true");
    } else {
      dot.removeAttribute("aria-current");
    }
  }
}

// Entry 0 is the pre-SCF direct-field seed, not iteration zero of the solve. A one-entry
// history is a converged result, not an SCF run, and must not be labelled as one.
function ipdIterationTooltip(index, count) {
  if (count === 1) {
    return "Converged induced dipoles";
  }
  return index === 0 ? "Initial (direct-field seed)" : `SCF iteration ${index}`;
}

function ipdIterationLabel(index, history) {
  const count = history.mu_history.length;
  const parts = [ipdIterationTooltip(index, count)];
  if (count > 1) {
    parts.push(`${index} / ${count - 1}`);
  }
  if (typeof history.energy === "number") {
    parts.push(`${history.energy.toFixed(3)} kcal/mol`);
  }
  if (!history.converged) {
    // The SCF stopped at its iteration cap. The last entry is where it got to, not a solution.
    parts.push("not converged");
  }
  return parts.join("  |  ");
}

// --- IPD iteration selection ------------------------------------------------

function selectIpdIteration(index) {
  stopIpdPlayback();
  showIpdIteration(index);
}

// The single entry point for everything that depends on the selected SCF step. It must not
// call showFrame, renderFrame or renderIpdFrame: the geometry has not changed, and rebuilding
// the model would throw away the camera the user set to look at the convergence.
function showIpdIteration(index) {
  const frame = selectedFrame();
  const history = currentIpdHistory();
  if (!frame || !history) {
    return;
  }
  if (index < 0 || index >= history.mu_history.length) {
    return;
  }

  currentIpdIteration = index;
  renderIpdDipoles(frame, history.mu_history[index]);
  renderIpdTimelineSelection();
  el("ipd-iteration-label").textContent = ipdIterationLabel(index, history);
}

// --- IPD arrows -------------------------------------------------------------

function clearIpdDipoles() {
  if (!ipdViewer) {
    return;
  }
  ipdShapes.forEach((shape) => ipdViewer.removeShape(shape));
  ipdShapes = [];
}

// Arrows only. The molecule, its styling and the camera are left alone, which is what lets a
// walk through the SCF be read as convergence rather than as a series of unrelated pictures.
function renderIpdDipoles(frame, vectors) {
  if (!ipdViewer || !frame?.coords || !vectors) {
    return;
  }
  clearIpdDipoles();
  addDipoleArrows(
    ipdViewer,
    frame.coords,
    vectors,
    IPD_ARROW_COLOR,
    currentIpdArrowScale(),
    ipdShapes
  );
  ipdViewer.render();
}

// --- IPD playback -----------------------------------------------------------

function setIpdPlayButton(playing) {
  const button = el("ipd-play");
  button.textContent = playing ? "❚❚" : "▶";
  button.setAttribute("aria-label", playing ? "Pause IPD" : "Play IPD");
}

function stopIpdPlayback() {
  if (ipdPlaybackTimer !== null) {
    clearInterval(ipdPlaybackTimer);
    ipdPlaybackTimer = null;
  }
  setIpdPlayButton(false);
}

function startIpdPlayback() {
  const history = currentIpdHistory();
  if (ipdPlaybackTimer !== null || !history || history.mu_history.length < 2) {
    return;
  }
  setIpdPlayButton(true);
  ipdPlaybackTimer = setInterval(() => {
    // Calls showIpdIteration rather than duplicating the render, so playback and a click on a
    // dot produce exactly the same thing.
    showIpdIteration((currentIpdIteration + 1) % history.mu_history.length);
  }, IPD_INTERVAL_MS);
}

function toggleIpdPlayback() {
  if (ipdPlaybackTimer === null) {
    startIpdPlayback();
  } else {
    stopIpdPlayback();
  }
}

// --- IPD computation --------------------------------------------------------

function ipdUrl(suffix) {
  const uploadId = el("collection-select").value;
  const slug = el("system-select").value;
  return (
    `/api/uploads/${encodeURIComponent(uploadId)}/systems/${encodeURIComponent(slug)}${suffix}`
  );
}

// A computed result replaces the frame's whole availability block, so the cached histories
// hanging off it have to be carried across by hand -- they are still valid, the geometry did
// not change.
function applyIpdMetadata(frame, ipd) {
  const history = frame.ipd?.history;
  frame.ipd = ipd;
  if (history) {
    frame.ipd.history = history;
  }
}

async function computeIpd() {
  const frame = selectedFrame();
  const mode = el("ipd-mode").value;
  if (!frame || !mode) {
    return;
  }
  setStatus(`Computing ${mode} for frame ${frame.frame_index + 1}…`);
  const result = await callJson(ipdUrl(`/frames/${frame.frame_index}/ipd`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });

  applyIpdMetadata(frame, result.ipd);
  // The response already carries the history, so seeding the cache here is what makes the
  // transition from "Compute IPD" to arrows happen without a second request.
  frame.ipd.history ??= {};
  frame.ipd.history[mode] = result.history;

  renderIpdSection(frame);
  await selectIpdMode(mode);
  setStatus(`${mode}: ${result.history.iteration_count} iterations`);
}

// The button that actually gets used: nothing is ever pre-computed, so a 14-frame scan would
// otherwise be 14 presses and 14 collection rewrites.
async function computeIpdForTrajectory() {
  const mode = el("ipd-mode").value;
  if (!currentTrajectory || !mode) {
    return;
  }
  setStatus(`Computing ${mode} for all ${currentTrajectory.frames.length} frames…`);
  const result = await callJson(ipdUrl("/ipd"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });

  for (const entry of result.frames) {
    const frame = currentTrajectory.frames[entry.frame_index];
    if (frame) {
      applyIpdMetadata(frame, entry.ipd);
    }
  }

  renderIpdSection(selectedFrame());
  await selectIpdMode(mode);

  const parts = [`${result.computed.length} computed`];
  if (result.skipped.length) parts.push(`${result.skipped.length} already stored`);
  if (result.failed.length) parts.push(`${result.failed.length} failed`);
  setStatus(`${result.label}: ${parts.join(", ")}`);
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

// --- tabs -------------------------------------------------------------------

// aria-selected is the state; the panel's `hidden` and the CSS both follow it, so there is no
// second copy of "which tab is active" to keep in step.
const TABS = [
  { tab: "tab-trajectory", panel: "panel-trajectory" },
  { tab: "tab-plots", panel: "panel-plots", onShow: () => plots.activate() },
];

function selectTab(id) {
  for (const entry of TABS) {
    const active = entry.tab === id;
    const tab = el(entry.tab);
    tab.setAttribute("aria-selected", String(active));
    // Roving tabindex: the tab bar is one stop in the page's tab order, and the arrow keys
    // move within it.
    tab.tabIndex = active ? 0 : -1;
    el(entry.panel).hidden = !active;
    if (active) {
      entry.onShow?.();
    }
  }
}

function initTabs() {
  TABS.forEach((entry, index) => {
    const tab = el(entry.tab);
    tab.addEventListener("click", () => selectTab(entry.tab));
    tab.addEventListener("keydown", (event) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      if (step === undefined) {
        return;
      }
      event.preventDefault();
      const next = TABS[(index + step + TABS.length) % TABS.length];
      selectTab(next.tab);
      el(next.tab).focus();
    });
  });
}

// --- wiring -----------------------------------------------------------------

function guard(handler) {
  return (event) =>
    Promise.resolve(handler(event)).catch((error) => setStatus(error.message, true));
}

function init() {
  viewer = window.$3Dmol.createViewer(el("viewer"), { backgroundColor: "white" });
  initTabs();

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
  // Trajectory dots wire themselves up in renderTrajectoryTimeline -- each one is bound to the
  // frame it stands for, and every one calls selectFrame.

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

  // The IPD viewer cannot be created until the panel is open and has a size.
  el("ipd-section").addEventListener(
    "toggle",
    guard(async () => {
      if (!isIpdSectionOpen()) {
        stopIpdPlayback();
        return;
      }
      initializeIpdViewer();
      const frame = selectedFrame();
      if (frame && ipdMode(frame, currentIpdMode)?.stored) {
        await selectIpdMode(currentIpdMode);
      }
    })
  );

  el("ipd-mode").addEventListener(
    "change",
    guard((event) => selectIpdMode(event.target.value))
  );
  el("ipd-compute").addEventListener("click", guard(computeIpd));
  el("ipd-compute-all").addEventListener("click", guard(computeIpdForTrajectory));
  el("ipd-play").addEventListener("click", toggleIpdPlayback);
  // Arrows only -- the geometry has not changed, and redrawing it would reset the camera.
  el("ipd-arrow-scale").addEventListener("input", () => {
    updateIpdArrowScaleReadout();
    const history = currentIpdHistory();
    if (history) {
      renderIpdDipoles(selectedFrame(), history.mu_history[currentIpdIteration]);
    }
  });

  guard(loadCollections)();
}

init();
