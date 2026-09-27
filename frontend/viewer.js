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
// 3Dmol's default sodium colour is too dark for the black atomic charge labels.
const SODIUM_COLOR = 0x7dd3fc;
const FRAME_INTERVAL_MS = 500;

// Arrow geometry ported from _arrow_cgo (pymol_dipole.py) and verified to reproduce PyMOL to
// floating-point precision. Changing these changes the scientific reading of the picture.
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

// Fragment identity is deliberately a translucent halo rather than a replacement atom colour:
// element colours still carry chemical meaning, while cyan/magenta remain easy to distinguish
// from one another and from the permanent-dipole palette.
const FRAGMENT_COLORS = {
  A: 0x00a6d6,
  B: 0xd81b60,
};
const FRAGMENT_HALO_RADIUS = 0.48;
const FRAGMENT_HALO_OPACITY = 0.28;

let viewer = null;
let currentTrajectory = null;
let currentFrameIndex = 0;
let playbackTimer = null;
let allSystems = [];

// Which multipole table is on screen, and which atoms it lists. The selected frame stays the
// primary state -- these only decide how that frame is presented.
let currentMultipoleView = "charges";
let currentAtomFilter = "all";
// The IPD comparison has its own filter: changing it must not alter the table on Trajectory.
let currentDipoleComparisonAtomFilter = "all";

// Viewer overlays: what is drawn *on* the molecule, as opposed to tabulated beneath it.
// Dipoles are three independent layers because arrows of different colours coexist readably;
// charge labels are one exclusive choice because three numbers per atom would not.
let showMonomerDipoles = false;
let showDimerDipoles = false;
let showDeltaDipoles = false;
let chargeLabelMode = "none";
// Both fragments are identified on first load; these choices then remain user state while the
// selected frame, system, or collection changes.
let showFragmentA = true;
let showFragmentB = true;

// Handles for what each viewer drew, so permanent, fragment, and induced layers clear only
// their own work.
let fragmentShapes = [];
let multipoleShapes = [];
let chargeLabels = [];
let ipdMultipoleShapes = [];
let ipdChargeLabels = [];

// Angstrom per atomic unit, fitted once per trajectory -- see setArrowScale. The slider is a
// multiplier on top of it, so the automatic fit stays the reference point and "1.0x" always
// means "longest arrow in this trajectory is DEFAULT_ARROW_LEN".
let baseArrowScale = 1.0;

// IPD has its own tab and viewer, but shares the physical frame with Trajectory. Its second
// axis is the SCF history for one frame and damping mode.
let ipdViewer = null;
const trajectoryMeasurement = {
  picks: [],
  shapes: [],
  readoutId: "viewer-measurement",
};
const ipdMeasurement = {
  picks: [],
  shapes: [],
  readoutId: "ipd-viewer-measurement",
};
let ipdShowMonomerDipoles = false;
let ipdShowDimerDipoles = false;
let ipdShowDeltaDipoles = false;
let ipdChargeLabelMode = "none";
let currentIpdMode = null;
let currentIpdIteration = 0;
let ipdPlaybackTimer = null;
let ipdShapes = [];
let ipdBaseArrowScale = 1.0;
let ipdHistoryRequestToken = 0;
let ipdHistoryGenerations = new Map();
let ipdIterationMemory = new Map();
let ipdFailedFrames = new Map();
let ipdComputeInFlight = false;
let deltaMtpCapability = null;
let deltaMtpComputeInFlight = false;

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

// A WebGL canvas measures its container, and a container inside a hidden tab is 0x0. Resizing
// to that collapses the canvas, so every path that measures has to check first.
function isLaidOut(container) {
  return (
    Boolean(container) &&
    !container.closest("[hidden]") &&
    container.clientWidth > 0 &&
    container.clientHeight > 0
  );
}

function resizeViewerIfVisible(target, containerId) {
  const container = el(containerId);
  if (!target || !isLaidOut(container)) return;
  // 3Dmol.resize updates its WebGL drawing buffer to the CSS box and preserves the camera.
  // Do not call zoomTo here: resizing a browser window must not discard the user's view.
  target.resize();
}

function resizeVisibleViewers() {
  resizeViewerIfVisible(viewer, "viewer");
  resizeViewerIfVisible(ipdViewer, "ipd-viewer");
}

function debounce(handler, delay) {
  let timer = null;
  return () => {
    window.clearTimeout(timer);
    timer = window.setTimeout(handler, delay);
  };
}

const resizeVisibleViewersDebounced = debounce(resizeVisibleViewers, 120);

const MEASUREMENT_INSTRUCTION = "Click two atoms to measure their distance.";
const MEASUREMENT_COLOR = 0xd4a017;

function measurementReadout(state, text = MEASUREMENT_INSTRUCTION, complete = false) {
  const node = el(state.readoutId);
  if (!node) return;
  node.textContent = text;
  node.dataset.complete = String(Boolean(complete));
}

function clearMeasurement(target, state, { render = true } = {}) {
  if (target) {
    state.shapes.forEach((shape) => target.removeShape(shape));
  }
  state.picks = [];
  state.shapes = [];
  measurementReadout(state);
  if (target && render) target.render();
}

function point(coords) {
  return { x: coords[0], y: coords[1], z: coords[2] };
}

function atomMeasurementLabel(frame, index) {
  return `${frame.symbols[index]}${index + 1}`;
}

function highlightMeasuredAtom(target, frame, state, index) {
  state.shapes.push(
    target.addSphere({
      center: point(frame.coords[index]),
      radius: 0.28,
      color: MEASUREMENT_COLOR,
      opacity: 0.45,
    })
  );
}

function pickMeasurementAtom(target, frame, state, index) {
  if (!Number.isInteger(index) || !frame?.coords?.[index]) return;
  // After a completed pair, the next atom starts a fresh measurement. With one pending atom,
  // a click completes the pair (including a deliberate same-atom zero-distance measurement).
  if (state.picks.length !== 1) clearMeasurement(target, state, { render: false });
  state.picks.push(index);
  highlightMeasuredAtom(target, frame, state, index);

  if (state.picks.length === 1) {
    measurementReadout(
      state,
      `${atomMeasurementLabel(frame, index)} selected; click a second atom.`
    );
  } else {
    const [first, second] = state.picks;
    const a = frame.coords[first];
    const b = frame.coords[second];
    const distance = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    const text = `${atomMeasurementLabel(frame, first)}–${atomMeasurementLabel(
      frame,
      second
    )}: ${distance.toFixed(3)} Å`;
    state.shapes.push(
      target.addLine({
        start: point(a),
        end: point(b),
        color: MEASUREMENT_COLOR,
        linewidth: 2,
        dashed: true,
      })
    );
    // Keep the molecular view unobstructed; the accessible readout below it carries the
    // complete atom-pair label and distance.
    measurementReadout(state, text, true);
  }
  target.render();
}

function makeAtomsMeasurable(target, model, frame, state) {
  const atoms = model.selectedAtoms({});
  atoms.forEach((atom, index) => {
    atom.__measurementIndex = index;
  });
  model.setClickable({}, true, (atom) => {
    pickMeasurementAtom(target, frame, state, atom.__measurementIndex);
  });
}

// One geometry renderer, two named entry points. renderFrame and renderIpdFrame stay distinct
// -- they are the two halves of the design rule, and each one's caller means something
// different by it -- but the body lives once, because two copies of the bond-stripping,
// styling and measurement behavior would eventually disagree.
function drawGeometry(target, frame, { resetCamera, measurementState }) {
  if (!target || !frame?.xyz) {
    clearMeasurement(target, measurementState, { render: false });
    return;
  }
  clearMeasurement(target, measurementState, { render: false });
  target.removeAllModels();
  const model = target.addModel(frame.xyz, "xyz");
  stripIntermonomerBonds(model, frame.n_atoms_A, frame.n_atoms);
  target.setStyle(
    {},
    { sphere: { scale: SPHERE_SCALE }, stick: { radius: STICK_RADIUS } }
  );
  // Apply the same lighter sodium colour in both viewers so black Δq labels remain legible.
  target.setStyle(
    { elem: "Na" },
    {
      sphere: { scale: SPHERE_SCALE, color: SODIUM_COLOR },
      stick: { radius: STICK_RADIUS, color: SODIUM_COLOR },
    }
  );
  makeAtomsMeasurable(target, model, frame, measurementState);
  // Only on the first frame of a trajectory. Zooming on every step throws the camera away
  // mid-scrub, which is disorienting and hides what actually changed.
  if (resetCamera) {
    target.zoomTo();
  }
  target.render();
}

function clearFragmentHighlights() {
  if (viewer) {
    fragmentShapes.forEach((shape) => viewer.removeShape(shape));
  }
  fragmentShapes = [];
}

function validFragmentGeometry(frame) {
  const nAtoms = frame?.n_atoms;
  const nAtomsA = frame?.n_atoms_A;
  return (
    Number.isInteger(nAtoms) &&
    nAtoms > 0 &&
    Number.isInteger(nAtomsA) &&
    nAtomsA > 0 &&
    nAtomsA < nAtoms &&
    Array.isArray(frame.coords) &&
    frame.coords.length === nAtoms &&
    frame.coords.every(
      (coords) =>
        Array.isArray(coords) &&
        coords.length === 3 &&
        coords.every((value) => typeof value === "number" && Number.isFinite(value))
    )
  );
}

// Halos are viewer shapes rather than model styles. That keeps the element-coloured molecular
// model untouched and gives this layer an independent lifecycle from permanent multipoles and
// click-to-measure shapes.
function renderFragmentHighlights(frame, { render = true } = {}) {
  if (!viewer) return;
  clearFragmentHighlights();

  if (validFragmentGeometry(frame)) {
    frame.coords.forEach((coords, index) => {
      const fragment = index < frame.n_atoms_A ? "A" : "B";
      const visible = fragment === "A" ? showFragmentA : showFragmentB;
      if (!visible) return;
      fragmentShapes.push(
        viewer.addSphere({
          center: point(coords),
          radius: FRAGMENT_HALO_RADIUS,
          color: FRAGMENT_COLORS[fragment],
          opacity: FRAGMENT_HALO_OPACITY,
        })
      );
    });
  }

  // Invalid/legacy data still needs one render after clearing stale shapes from the last frame.
  if (render) viewer.render();
}

// The main viewer: geometry, fragment identity, and MBIS analysis.
function renderFrame(frame, { resetCamera }) {
  // Remove old-coordinate halos before drawGeometry renders the replacement model, then add
  // the selected fragments only after that model has loaded.
  clearFragmentHighlights();
  drawGeometry(viewer, frame, { resetCamera, measurementState: trajectoryMeasurement });
  renderFragmentHighlights(frame);
}

// --- frames -----------------------------------------------------------------

function frameLabel(frame) {
  // separation_label is formatted server-side with its real units -- "0.70 Re" for a ratio,
  // "2.180 A" for a distance -- and is never empty. eq_ratio and contact_distance_ang are
  // both null for frames where they could not be derived, so neither can be formatted here
  // without a guard.
  const parts = [`Frame ${frame.frame_index + 1} / ${currentTrajectory.frames.length}`];
  if (Number.isInteger(frame.n_atoms) && frame.n_atoms > 0) {
    parts.push(`${frame.n_atoms} atoms`);
  }
  if (frame.separation_label) {
    parts.push(frame.separation_label);
  }
  if (typeof frame.contact_distance_ang === "number") {
    parts.push(`closest contact ${frame.contact_distance_ang.toFixed(2)} Å`);
  }
  return parts.join("  |  ");
}

// The single entry point for everything that depends on the selected physical frame. Both tabs
// address this state; the IPD tab adds an SCF iteration without owning a second frame index.
function showFrame(index, { resetCamera = false } = {}) {
  const frame = currentTrajectory?.frames[index];
  if (!frame) {
    return;
  }
  stopIpdPlayback();
  ipdHistoryRequestToken += 1;
  currentFrameIndex = index;
  currentIpdIteration = 0;
  // The IPD canvas may be hidden and therefore not redrawn below, but its old measurement
  // must still clear as soon as the shared physical frame changes.
  clearMeasurement(ipdViewer, ipdMeasurement, { render: false });

  renderFrame(frame, { resetCamera });
  const label = frameLabel(frame);
  el("frame-label").textContent = label;
  el("ipd-frame-label").textContent = label;
  renderTimelineSelection();
  renderMultipoleSection(frame);
  updateOverlayControls(frame);
  renderMultipoleOverlays(frame);
  renderIpdSection(frame);
  // A revisited frame/mode can render synchronously from its frame-local history cache.
  renderAtomicDipoleComparison(frame, currentIpdHistory());

  if (ipdViewer && isIpdTabActive()) {
    renderIpdFrame(frame, { resetCamera });
    // Trajectory playback is deliberately not a hidden separation player for the IPD tab.
    // Manual selection loads once and is guarded against stale frame/mode responses.
    if (playbackTimer === null) {
      guard(() => selectIpdMode(currentIpdMode))();
    }
  }
}

// Every user-driven physical-frame change goes through here. Playback calls showFrame directly
// instead: this path pauses it, which is also what makes the IPD separation timeline manual.
function selectFrame(index) {
  pausePlayback();
  showFrame(index);
}

function setTrajectory(trajectory) {
  pausePlayback();
  stopIpdPlayback();
  ipdHistoryRequestToken += 1;
  clearMeasurement(viewer, trajectoryMeasurement, { render: false });
  clearMeasurement(ipdViewer, ipdMeasurement, { render: false });

  currentTrajectory = trajectory;
  currentFrameIndex = 0;
  currentIpdIteration = 0;
  currentIpdMode = null;
  ipdHistoryGenerations = new Map();
  ipdIterationMemory = new Map();
  ipdFailedFrames = new Map();
  ipdBaseArrowScale = 1.0;
  currentIpdMode = preferredIpdMode(trajectory.frames[0]);

  // The permanent-dipole fit and the selected-mode IPD fit are both trajectory-wide.
  setArrowScale(trajectory);
  setIpdArrowScale(trajectory, currentIpdMode);

  el("play-button").disabled = trajectory.frames.length < 2;
  renderPhysicalTimelines(trajectory);
  showFrame(0, { resetCamera: true });

  // Plots re-slices this exact object, so all tabs share one in-memory trajectory.
  plots.setTrajectory(trajectory);
}

// --- physical-frame timelines ----------------------------------------------

const PHYSICAL_TIMELINES = ["frame-timeline", "ipd-separation-timeline"];

function failedIpdFrames(mode) {
  return ipdFailedFrames.get(mode) ?? new Set();
}

function ipdFrameState(frame, mode) {
  const selected = ipdMode(frame, mode);
  if (failedIpdFrames(mode).has(Number(frame.frame_index))) {
    return { id: "error", label: "recomputation failed; previous result retained if available" };
  }
  if (selected?.problem) {
    return { id: "error", label: `stored result problem: ${selected.problem}` };
  }
  if (selected?.stored) {
    return selected.converged === false
      ? { id: "not-converged", label: "computed; SCF did not converge" }
      : { id: "converged", label: "computed" };
  }
  if (frame?.ipd?.computable) {
    return { id: "missing", label: "not computed" };
  }
  return { id: "unavailable", label: "IPD unavailable" };
}

function renderPhysicalTimeline(timelineId, trajectory) {
  const timeline = el(timelineId);
  timeline.replaceChildren();
  const isIpd = timelineId === "ipd-separation-timeline";

  trajectory.frames.forEach((frame, index) => {
    const dot = document.createElement("button");
    const separation = frame.separation_label || `Frame ${index + 1}`;
    dot.type = "button";
    dot.className = "timeline-dot";
    dot.dataset.frame = String(index);
    dot.title = separation;

    if (isIpd) {
      const state = ipdFrameState(frame, currentIpdMode);
      dot.dataset.state = state.id;
      dot.title = `${separation} — ${state.label}`;
      dot.setAttribute(
        "aria-label",
        `Select separation ${separation}, frame ${index + 1}: ${state.label}`
      );
    } else {
      dot.setAttribute("aria-label", `Select frame ${index + 1}: ${separation}`);
    }

    dot.addEventListener("click", () => selectFrame(index));
    timeline.append(dot);
  });
}

function renderPhysicalTimelines(trajectory) {
  for (const timelineId of PHYSICAL_TIMELINES) {
    renderPhysicalTimeline(timelineId, trajectory);
  }
  renderTimelineSelection();
}

function renderIpdSeparationTimeline() {
  if (currentTrajectory) {
    renderPhysicalTimeline("ipd-separation-timeline", currentTrajectory);
    renderTimelineSelection();
  }
}

function renderTimelineSelection() {
  for (const timelineId of PHYSICAL_TIMELINES) {
    for (const dot of el(timelineId).querySelectorAll(".timeline-dot")) {
      if (Number(dot.dataset.frame) === currentFrameIndex) {
        dot.setAttribute("aria-current", "true");
      } else {
        dot.removeAttribute("aria-current");
      }
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

function completeChargeTotal(values, indices) {
  if (!Array.isArray(values) || indices.length === 0) {
    return null;
  }
  let total = 0;
  for (const index of indices) {
    const value = values[index];
    if (typeof value !== "number" || !Number.isFinite(value)) {
      return null;
    }
    total += value;
  }
  return total;
}

function monomerChargeTotals(frame, data, monomer) {
  const indices = [];
  for (let i = 0; i < frame.n_atoms; i += 1) {
    if (atomMonomer(frame, i) === monomer) {
      indices.push(i);
    }
  }

  const isolated = completeChargeTotal(data.monomer, indices);
  const dimer = completeChargeTotal(data.dimer, indices);
  return {
    isolated,
    dimer,
    delta: isolated === null || dimer === null ? null : dimer - isolated,
  };
}

// Charges and volume ratios: one number per atom, so monomer, dimer and their difference are
// each a single column.
function renderScalarTable(frame, data, heading, { summarizeCharges = false } = {}) {
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

  if (summarizeCharges) {
    const foot = el("multipole-table-foot");
    for (const monomer of ["A", "B"]) {
      if (currentAtomFilter !== "all" && currentAtomFilter !== monomer) {
        continue;
      }
      const totals = monomerChargeTotals(frame, data, monomer);
      appendRow(foot, "td", [
        { text: `Monomer ${monomer} total`, className: "summary-label" },
        monomer,
        { text: num(totals.isolated), className: "numeric group-start" },
        { text: num(totals.dimer), className: "numeric" },
        { text: num(totals.delta, true), className: "numeric delta" },
      ]);
    }
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
  el("multipole-table-head").replaceChildren();
  el("multipole-table-body").replaceChildren();
  el("multipole-table-foot").replaceChildren();
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
    renderScalarTable(
      frame,
      data,
      currentMultipoleView === "charges" ? "q" : "ratio",
      { summarizeCharges: currentMultipoleView === "charges" }
    );
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

// What gets drawn *on* either molecule. Geometry stays separate, the table stays numeric, and
// each viewer/layer owns its handles so permanent overlays never erase induced IPD arrows.

function deltaDipoles(frame) {
  const dipoles = frame.multipoles?.dipoles;
  const monomer = dipoles?.monomer;
  const dimer = dipoles?.dimer;
  if (!monomer || !dimer) {
    return null;
  }
  return monomer.map((mu, i) => [dimer[i][0] - mu[0], dimer[i][1] - mu[1], dimer[i][2] - mu[2]]);
}

// --- final atomic dipole comparison ----------------------------------------

function clearAtomicDipoleComparison() {
  el("atomic-dipole-comparison-table-head").replaceChildren();
  el("atomic-dipole-comparison-table-body").replaceChildren();
  el("atomic-dipole-comparison-section").hidden = true;
}

function alignedDipoleVectors(vectors, nAtoms) {
  return (
    Array.isArray(vectors) &&
    vectors.length === nAtoms &&
    vectors.every(
      (vector) =>
        Array.isArray(vector) &&
        vector.length === 3 &&
        vector.every((value) => typeof value === "number" && Number.isFinite(value))
    )
  );
}

function dipoleComparisonAtomVisible(frame, index) {
  return (
    currentDipoleComparisonAtomFilter === "all" ||
    atomMonomer(frame, index) === currentDipoleComparisonAtomFilter
  );
}

// This table always means the converged solution. It intentionally ignores
// currentIpdIteration, which is only the iteration drawn in the 3D viewer.
function renderAtomicDipoleComparison(frame, history) {
  clearAtomicDipoleComparison();

  const selectedMode = ipdMode(frame, currentIpdMode);
  if (
    !frame ||
    frame !== selectedFrame() ||
    failedIpdFrames(currentIpdMode).has(Number(frame.frame_index)) ||
    !Number.isInteger(frame.n_atoms) ||
    frame.n_atoms < 1 ||
    !Array.isArray(frame.symbols) ||
    frame.symbols.length !== frame.n_atoms ||
    selectedMode?.converged !== true ||
    history?.converged !== true ||
    history.mode !== currentIpdMode ||
    history.n_atoms !== frame.n_atoms ||
    history.n_atoms_A !== frame.n_atoms_A ||
    !Array.isArray(history.mu_history) ||
    history.mu_history.length < 1 ||
    history.iteration_count !== history.mu_history.length
  ) {
    return;
  }

  let mbisDelta;
  try {
    // The same existing MBIS dimer-minus-monomer calculation used by permanent overlays.
    mbisDelta = deltaDipoles(frame);
  } catch (_error) {
    return;
  }
  const finalIpd = history.mu_history[history.mu_history.length - 1];
  if (
    !alignedDipoleVectors(mbisDelta, frame.n_atoms) ||
    !alignedDipoleVectors(finalIpd, frame.n_atoms)
  ) {
    return;
  }

  const head = el("atomic-dipole-comparison-table-head");
  appendRow(head, "th", [
    { text: "Atom", rowSpan: 2 },
    { text: "Monomer", rowSpan: 2 },
    { text: "MBIS Δμ = μ dimer − μ monomer (a.u.)", colSpan: 4, className: "group" },
    { text: "Final IPD induced dipole (a.u.)", colSpan: 4, className: "group" },
  ]);
  appendRow(
    head,
    "th",
    ["x", "y", "z", "|Δμ|", "x", "y", "z", "|μind|"].map((text, index) => ({
      text,
      className: index % 4 === 0 ? "numeric group-start" : "numeric",
    }))
  );

  const body = el("atomic-dipole-comparison-table-body");
  for (let i = 0; i < frame.n_atoms; i += 1) {
    if (!dipoleComparisonAtomVisible(frame, i)) {
      continue;
    }
    appendRow(body, "td", [
      atomLabel(frame, i),
      atomMonomer(frame, i),
      ...vectorCells(mbisDelta[i], true),
      ...vectorCells(finalIpd[i], true),
    ]);
  }
  el("atomic-dipole-comparison-section").hidden = false;
}

function setDipoleComparisonAtomFilter(filter) {
  currentDipoleComparisonAtomFilter = filter;
  setActiveButton(el("dipole-comparison-atom-filters"), "filter", filter);
  renderAtomicDipoleComparison(selectedFrame(), currentIpdHistory());
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
  updatePermanentArrowScaleReadout("arrow-scale", "arrow-scale-readout");
  updatePermanentArrowScaleReadout(
    "ipd-multipole-arrow-scale",
    "ipd-multipole-arrow-scale-readout"
  );
}

// Three independent sliders (permanent dipoles in each viewer, plus induced dipoles), one rule
// for combining a fitted base with a multiplier.
function arrowScale(base, sliderId) {
  return base * Number(el(sliderId).value);
}

function currentArrowScale() {
  return arrowScale(baseArrowScale, "arrow-scale");
}

function currentIpdMultipoleArrowScale() {
  return arrowScale(baseArrowScale, "ipd-multipole-arrow-scale");
}

// Both numbers, because neither alone is enough: the multiplier says how far from the
// automatic fit you are, and the absolute scale is what makes an arrow length mean something.
function updatePermanentArrowScaleReadout(sliderId, readoutId) {
  const multiplier = Number(el(sliderId).value);
  const scale = arrowScale(baseArrowScale, sliderId);
  el(readoutId).textContent = `${multiplier.toFixed(1)}× (${scale.toFixed(1)} Å per a.u.)`;
}

function updateArrowScaleReadout() {
  updatePermanentArrowScaleReadout("arrow-scale", "arrow-scale-readout");
}

// Takes its viewer, scale, and handle array so one arrow convention serves both viewers; the
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

function addChargeLabels(target, frame, mode, sink) {
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
    sink.push(
      target.addLabel(
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

function removePermanentOverlays(target, shapes, labels) {
  shapes.forEach((shape) => target.removeShape(shape));
  labels.forEach((label) => target.removeLabel(label));
}

function addPermanentOverlays(
  target,
  frame,
  { monomer, dimer, delta, chargeMode, scale },
  shapeSink,
  labelSink
) {
  if (!frame?.coords) {
    return;
  }
  for (const [layer, visible] of [
    ["monomer", monomer],
    ["dimer", dimer],
    ["delta", delta],
  ]) {
    const vectors = visible ? dipoleVectors(frame, layer) : null;
    if (vectors) {
      addDipoleArrows(
        target,
        frame.coords,
        vectors,
        DIPOLE_COLORS[layer],
        scale,
        shapeSink
      );
    }
  }
  if (chargeMode !== "none") {
    addChargeLabels(target, frame, chargeMode, labelSink);
  }
}

function clearMultipoleOverlays() {
  removePermanentOverlays(viewer, multipoleShapes, chargeLabels);
  multipoleShapes = [];
  chargeLabels = [];
}

function renderMultipoleOverlays(frame) {
  if (!viewer) {
    return;
  }
  clearMultipoleOverlays();
  addPermanentOverlays(
    viewer,
    frame,
    {
      monomer: showMonomerDipoles,
      dimer: showDimerDipoles,
      delta: showDeltaDipoles,
      chargeMode: chargeLabelMode,
      scale: currentArrowScale(),
    },
    multipoleShapes,
    chargeLabels
  );
  viewer.render();
}

const OVERLAY_CONTROL_IDS = {
  trajectory: {
    monomer: "show-monomer-dipoles",
    dimer: "show-dimer-dipoles",
    delta: "show-delta-dipoles",
    charges: "charge-label-mode",
  },
  ipd: {
    monomer: "ipd-show-monomer-dipoles",
    dimer: "ipd-show-dimer-dipoles",
    delta: "ipd-show-delta-dipoles",
    charges: "ipd-charge-label-mode",
  },
};

// A control for data this frame does not have is disabled rather than hidden, so the reason
// the viewer is empty is visible. Both viewers read the same frame payload.
function updateOverlayControlSet(frame, ids) {
  const dipoles = frame?.multipoles?.dipoles;
  el(ids.monomer).disabled = !dipoles?.monomer;
  el(ids.dimer).disabled = !dipoles?.dimer;
  el(ids.delta).disabled = !(dipoles?.monomer && dipoles?.dimer);

  const charges = frame?.multipoles?.charges;
  const available = {
    none: true,
    monomer: Boolean(charges?.monomer),
    dimer: Boolean(charges?.dimer),
    delta: Boolean(charges?.monomer && charges?.dimer),
  };
  for (const option of el(ids.charges).options) {
    option.disabled = !available[option.value];
  }
}

function updateOverlayControls(frame) {
  updateOverlayControlSet(frame, OVERLAY_CONTROL_IDS.trajectory);
  updateOverlayControlSet(frame, OVERLAY_CONTROL_IDS.ipd);
}

// Toggling an overlay redraws overlays only. Going through showFrame would destroy and rebuild
// the molecule, throwing the camera away for a change that does not touch the geometry.
function setFragmentAVisible(visible) {
  showFragmentA = visible;
  renderFragmentHighlights(selectedFrame());
}

function setFragmentBVisible(visible) {
  showFragmentB = visible;
  renderFragmentHighlights(selectedFrame());
}

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

// The IPD copies are intentionally independent: comparisons often need a permanent layer in
// one viewer while keeping the Trajectory canvas uncluttered.
function setIpdMonomerDipolesVisible(visible) {
  ipdShowMonomerDipoles = visible;
  renderIpdMultipoleOverlays(selectedFrame());
}

function setIpdDimerDipolesVisible(visible) {
  ipdShowDimerDipoles = visible;
  renderIpdMultipoleOverlays(selectedFrame());
}

function setIpdDeltaDipolesVisible(visible) {
  ipdShowDeltaDipoles = visible;
  renderIpdMultipoleOverlays(selectedFrame());
}

function setIpdChargeLabelMode(mode) {
  ipdChargeLabelMode = mode;
  renderIpdMultipoleOverlays(selectedFrame());
}

// --- IPD --------------------------------------------------------------------

// The IPD tab has two named axes. Its separation timeline changes the physical geometry through
// the shared frame state; its SCF timeline holds that geometry fixed and changes only the
// induced-dipole iteration. Permanent MBIS input overlays can coexist here for comparison, but
// remain independently controlled from the Trajectory viewer.

const IPD_ARROW_COLOR = 0xbf00bf; // the induced-dipole colour, kept from the previous frontend
const IPD_INTERVAL_MS = 200; // faster than the trajectory: an SCF is many more steps
const MAX_DOTS = 40; // beyond this the timeline degrades to a range input

function isIpdTabActive() {
  return el("tab-ipd").getAttribute("aria-selected") === "true";
}

// The IPD viewer is created on first tab activation, never at startup. A WebGL canvas measures
// its container, and a container inside a hidden tab is 0x0.
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

function clearIpdMultipoleOverlays() {
  if (!ipdViewer) {
    return;
  }
  removePermanentOverlays(ipdViewer, ipdMultipoleShapes, ipdChargeLabels);
  ipdMultipoleShapes = [];
  ipdChargeLabels = [];
}

function renderIpdMultipoleOverlays(frame) {
  if (!ipdViewer) {
    return;
  }
  clearIpdMultipoleOverlays();
  addPermanentOverlays(
    ipdViewer,
    frame,
    {
      monomer: ipdShowMonomerDipoles,
      dimer: ipdShowDimerDipoles,
      delta: ipdShowDeltaDipoles,
      chargeMode: ipdChargeLabelMode,
      scale: currentIpdMultipoleArrowScale(),
    },
    ipdMultipoleShapes,
    ipdChargeLabels
  );
  ipdViewer.render();
}

// The IPD viewer owns fixed geometry, optional permanent multipoles, and induced-dipole
// convergence as three separate layers. Its camera remains independent from Trajectory.
function renderIpdFrame(frame, { resetCamera = false } = {}) {
  // removeAllModels does not touch shapes or labels, so clear both overlay families before the
  // old geometry goes. Keeping separate handles prevents either family erasing the other.
  clearIpdDipoles();
  clearIpdMultipoleOverlays();
  drawGeometry(ipdViewer, frame, { resetCamera, measurementState: ipdMeasurement });
  renderIpdMultipoleOverlays(frame);
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

// Manages availability text and display state synchronously. The damping selector remains
// trajectory-wide and visible even when the selected separation has no result.
function renderIpdSection(frame) {
  renderDeltaMtpControls(frame);
  const message = el("ipd-message");
  const controls = el("ipd-controls");
  const display = el("ipd-display");
  const modes = ipdModes(frame);

  if (!frame?.ipd || modes.length === 0) {
    clearAtomicDipoleComparison();
    message.textContent = "Select a system to inspect induced point dipoles.";
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
  if (selected?.converged !== true) {
    // A finite last iteration from a capped SCF is not a converged solution to compare.
    clearAtomicDipoleComparison();
  }
  const anyComputable = currentTrajectory?.frames.some((entry) => entry.ipd?.computable);
  const compute = el("ipd-compute");
  const computeAll = el("ipd-compute-all");

  compute.disabled = ipdComputeInFlight || stored || !frame.ipd.computable;
  compute.textContent = stored ? "Computed" : "Compute IPD";
  computeAll.disabled = ipdComputeInFlight || !anyComputable;

  if (!frame.ipd.computable && !stored) {
    message.textContent =
      "IPD is unavailable for this frame because required MBIS/IPD inputs are missing or non-finite.";
    message.hidden = false;
    display.hidden = true;
    return;
  }

  if (selected?.problem) {
    message.textContent = `Stored ${selected.label} result cannot be used: ${selected.problem}.`;
    message.hidden = false;
    display.hidden = true;
    return;
  }

  if (!stored) {
    message.textContent = "No stored IPD calculation for this frame and damping mode.";
    message.hidden = false;
    display.hidden = true;
    return;
  }

  message.hidden = true;
  display.hidden = false;
  el("ipd-play").disabled = ipdComputeInFlight || (selected.iteration_count ?? 0) < 2;
  // The display div and tab both have to be visible before 3Dmol is allowed to measure it.
  if (isIpdTabActive()) {
    initializeIpdViewer();
  }
}

function fillIpdModeSelect(frame) {
  const select = el("ipd-mode");
  const modes = ipdModes(frame);
  select.replaceChildren();
  for (const mode of modes) {
    // Every mode remains reachable; the suffix describes only the selected frame.
    const option = new Option(mode.stored ? mode.label : `${mode.label} — not computed`, mode.id);
    select.append(option);
  }
  select.value = currentIpdMode ?? "";
  select.disabled = ipdComputeInFlight || modes.length === 0;
}

// --- IPD history ------------------------------------------------------------

// Cached on the frame object, which is ours: the whole trajectory already lives in the browser,
// so a mode revisited on a frame already seen costs nothing. The cache is not persistence --
// the server holds the computed result, this only avoids re-fetching it.
function ipdHistoryKey(frame, mode) {
  return `${frame.frame_index}:${mode}`;
}

function invalidateIpdHistory(frame, mode) {
  const key = ipdHistoryKey(frame, mode);
  ipdHistoryGenerations.set(key, (ipdHistoryGenerations.get(key) ?? 0) + 1);
  if (frame.ipd?.history) {
    delete frame.ipd.history[mode];
  }
}

async function loadIpdHistory(frame, mode) {
  frame.ipd.history ??= {};
  if (frame.ipd.history[mode]) {
    return frame.ipd.history[mode];
  }
  const key = ipdHistoryKey(frame, mode);
  const generation = ipdHistoryGenerations.get(key) ?? 0;
  const uploadId = el("collection-select").value;
  const slug = el("system-select").value;
  const history = await callJson(
    `/api/uploads/${encodeURIComponent(uploadId)}/systems/${encodeURIComponent(slug)}` +
      `/frames/${frame.frame_index}/ipd?mode=${encodeURIComponent(mode)}`
  );
  // A request that crossed a successful recomputation may return the old disk value. It can
  // satisfy its stale caller, but it must not repopulate the invalidated cache.
  if ((ipdHistoryGenerations.get(key) ?? 0) === generation) {
    frame.ipd.history ??= {};
    frame.ipd.history[mode] = history;
  }
  return history;
}

// One selected-mode scale spans every SCF iteration at every separation. A frame-by-frame
// auto-fit would make a smaller physical dipole look unchanged as the monomers separate.
function setIpdArrowScale(trajectory, mode) {
  let largest = 0;
  for (const frame of trajectory?.frames ?? []) {
    const maximum = ipdMode(frame, mode)?.max_abs_mu;
    if (typeof maximum === "number" && Number.isFinite(maximum)) {
      largest = Math.max(largest, maximum);
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

function ipdIterationKey(frame, mode) {
  return `${frame.frame_index}:${mode}`;
}

async function selectIpdMode(mode) {
  stopIpdPlayback();
  clearIpdDipoles();
  // Never leave values from the previous frame/mode visible during an asynchronous read.
  clearAtomicDipoleComparison();
  currentIpdMode = mode;
  currentIpdIteration = 0;
  const requestToken = ++ipdHistoryRequestToken;
  const frame = selectedFrame();

  setIpdArrowScale(currentTrajectory, mode);
  renderIpdSeparationTimeline();
  renderIpdSection(frame);

  if (!frame || !ipdMode(frame, mode)?.stored) {
    el("ipd-timeline").replaceChildren();
    el("ipd-iteration-label").textContent = "";
    return;
  }

  let history;
  try {
    history = await loadIpdHistory(frame, mode);
  } catch (error) {
    if (requestToken !== ipdHistoryRequestToken) {
      return;
    }
    clearAtomicDipoleComparison();
    throw error;
  }

  // A slower request for the previous separation or mode may complete after a later click.
  if (
    requestToken !== ipdHistoryRequestToken ||
    frame !== selectedFrame() ||
    mode !== currentIpdMode ||
    !isIpdTabActive()
  ) {
    return;
  }

  renderAtomicDipoleComparison(frame, history);
  renderIpdTimeline(history.mu_history.length);
  const remembered = ipdIterationMemory.get(ipdIterationKey(frame, mode));
  const initial = Number.isInteger(remembered)
    ? Math.min(remembered, history.mu_history.length - 1)
    : history.mu_history.length - 1;
  showIpdIteration(initial);
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
    dot.setAttribute("aria-label", `Select ${ipdIterationTooltip(index, iterationCount)}`);
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
  ipdIterationMemory.set(ipdIterationKey(frame, currentIpdMode), index);
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
  button.setAttribute(
    "aria-label",
    playing ? "Pause SCF convergence" : "Play SCF convergence"
  );
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

function systemUrl(suffix) {
  const uploadId = el("collection-select").value;
  const slug = el("system-select").value;
  return (
    `/api/uploads/${encodeURIComponent(uploadId)}/systems/${encodeURIComponent(slug)}${suffix}`
  );
}

// A refreshed availability block replaces the old one. Cached histories for unaffected modes
// survive, but a successfully recomputed mode must be deleted or the viewer would replay the
// pre-recompute result.
function applyIpdMetadata(frame, ipd, { invalidateMode = null } = {}) {
  const history = frame.ipd?.history;
  if (invalidateMode) {
    invalidateIpdHistory(frame, invalidateMode);
  }
  frame.ipd = ipd;
  if (history && Object.keys(history).length) {
    frame.ipd.history = history;
  }
}

function setIpdComputeState(inFlight, { allFrames = false } = {}) {
  ipdComputeInFlight = inFlight;
  if (inFlight) {
    stopIpdPlayback();
    ipdHistoryRequestToken += 1;
    clearAtomicDipoleComparison();
  }
  renderIpdSection(selectedFrame());
  if (inFlight) {
    el("ipd-compute").textContent = "Computing…";
    if (allFrames) {
      el("ipd-compute-all").textContent = "Computing all frames…";
    }
  } else {
    el("ipd-compute-all").textContent = "Compute all frames";
  }
}

function refreshTrajectoryConsumers(mode) {
  setIpdArrowScale(currentTrajectory, mode);
  renderIpdSeparationTimeline();
  plots.setTrajectory(currentTrajectory);
}

async function computeIpd() {
  const trajectory = currentTrajectory;
  const frame = selectedFrame();
  const mode = el("ipd-mode").value;
  if (!frame || !mode || ipdComputeInFlight) {
    return;
  }

  setIpdComputeState(true);
  setStatus(`Computing ${mode} for frame ${frame.frame_index + 1}…`);
  let result;
  try {
    result = await callJson(systemUrl(`/frames/${frame.frame_index}/ipd`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
  } finally {
    setIpdComputeState(false);
  }

  // A collection/system change while the request was running makes the response obsolete.
  if (trajectory !== currentTrajectory || !result) {
    return;
  }

  applyIpdMetadata(frame, result.ipd, { invalidateMode: mode });
  frame.energies = result.energies;
  // The response already carries this new history, so seed the cache after invalidating it.
  frame.ipd.history ??= {};
  frame.ipd.history[mode] = result.history;
  failedIpdFrames(mode).delete(Number(frame.frame_index));

  refreshTrajectoryConsumers(mode);
  renderIpdSection(frame);
  renderAtomicDipoleComparison(frame, frame.ipd.history?.[mode]);
  if (isIpdTabActive()) {
    await selectIpdMode(mode);
  }
  setStatus(`${result.history.label}: ${result.history.iteration_count} history entries`);
}

// This is deliberately reprocessing, not fill-missing: every computable frame is attempted for
// the selected mode, while each failed replacement leaves its previous stored value untouched.
async function computeIpdForTrajectory() {
  const trajectory = currentTrajectory;
  const mode = el("ipd-mode").value;
  if (!trajectory || !mode || ipdComputeInFlight) {
    return;
  }

  const storedCount = trajectory.frames.filter((frame) => ipdMode(frame, mode)?.stored).length;
  if (
    storedCount > 0 &&
    !window.confirm(
      `Compute all frames will replace ${storedCount} stored ${ipdMode(trajectory.frames[0], mode)?.label ?? mode} result(s). Continue?`
    )
  ) {
    return;
  }

  setIpdComputeState(true, { allFrames: true });
  setStatus(`Computing ${mode} for all ${trajectory.frames.length} frames…`);
  let result;
  try {
    result = await callJson(systemUrl("/ipd"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
  } finally {
    setIpdComputeState(false);
  }

  if (trajectory !== currentTrajectory || !result) {
    return;
  }

  const recomputed = new Set(result.computed.map(Number));
  ipdFailedFrames.set(
    mode,
    new Set(result.failed.map((failure) => Number(failure.frame_index)))
  );

  for (const entry of result.frames) {
    const frame = trajectory.frames.find(
      (candidate) => Number(candidate.frame_index) === Number(entry.frame_index)
    );
    if (!frame) {
      continue;
    }
    applyIpdMetadata(frame, entry.ipd, {
      invalidateMode: recomputed.has(Number(entry.frame_index)) ? mode : null,
    });
    frame.energies = entry.energies;
  }

  refreshTrajectoryConsumers(mode);
  renderIpdSection(selectedFrame());
  renderAtomicDipoleComparison(selectedFrame(), currentIpdHistory());
  if (isIpdTabActive()) {
    await selectIpdMode(mode);
  }

  const parts = [`${result.computed.length} computed`];
  if (result.failed.length) parts.push(`${result.failed.length} failed`);
  const firstFailure = result.failed[0]?.error;
  setStatus(
    `${result.label}: ${parts.join(", ")}${firstFailure ? `. ${firstFailure}` : ""}`,
    result.failed.length > 0
  );
}

// --- Delta-MTP computation --------------------------------------------------

function renderDeltaMtpControls(frame) {
  const controls = el("delta-mtp-controls");
  const message = el("delta-mtp-message");
  const compute = el("delta-mtp-compute");
  const computeAll = el("delta-mtp-compute-all");

  if (!frame?.delta_mtp) {
    controls.hidden = true;
    message.hidden = true;
    return;
  }

  controls.hidden = false;
  const stored = Boolean(frame.delta_mtp.stored);
  const anyComputable = currentTrajectory?.frames.some(
    (entry) => entry.delta_mtp?.computable
  );
  const serverAvailable = deltaMtpCapability?.available === true;
  compute.disabled =
    deltaMtpComputeInFlight || stored || !frame.delta_mtp.computable || !serverAvailable;
  compute.textContent = stored ? "Computed" : "Compute Delta-MTP";
  computeAll.disabled =
    deltaMtpComputeInFlight || !anyComputable || !serverAvailable;

  if (deltaMtpCapability === null) {
    message.textContent = "Checking whether this server can compute Delta-MTP…";
    message.hidden = false;
  } else if (!serverAvailable) {
    message.textContent = deltaMtpCapability.reason || "Delta-MTP is unavailable on this server.";
    message.hidden = false;
  } else if (frame.delta_mtp.problem) {
    message.textContent = frame.delta_mtp.problem;
    message.hidden = false;
  } else if (!frame.delta_mtp.computable && !stored) {
    message.textContent =
      "Delta-MTP is unavailable for this frame because one or more of its nine A/B/dimer multipole inputs are missing, non-finite, or incorrectly shaped.";
    message.hidden = false;
  } else {
    message.hidden = true;
  }
}

function setDeltaMtpComputeState(inFlight, { allFrames = false } = {}) {
  deltaMtpComputeInFlight = inFlight;
  renderDeltaMtpControls(selectedFrame());
  if (inFlight) {
    el("delta-mtp-compute").textContent = "Computing…";
    if (allFrames) {
      el("delta-mtp-compute-all").textContent = "Computing all frames…";
    }
  } else {
    el("delta-mtp-compute-all").textContent = "Compute Delta-MTP for all frames";
  }
}

function applyDeltaMtpResult(frame, entry) {
  frame.delta_mtp = entry.delta_mtp;
  frame.energies = entry.energies;
}

function refreshDeltaMtpConsumers(trajectory, energyCatalog) {
  trajectory.energy_catalog = energyCatalog;
  renderDeltaMtpControls(selectedFrame());
  plots.setTrajectory(trajectory);
}

async function computeDeltaMtp() {
  const trajectory = currentTrajectory;
  const frame = selectedFrame();
  if (!frame || deltaMtpComputeInFlight || frame.delta_mtp?.stored) {
    return;
  }

  setDeltaMtpComputeState(true);
  setStatus(`Computing Delta-MTP for frame ${frame.frame_index + 1}…`);
  let result;
  try {
    result = await callJson(systemUrl(`/frames/${frame.frame_index}/delta-mtp`), {
      method: "POST",
    });
  } finally {
    setDeltaMtpComputeState(false);
  }

  if (trajectory !== currentTrajectory || !result) {
    return;
  }
  applyDeltaMtpResult(frame, result);
  refreshDeltaMtpConsumers(trajectory, result.energy_catalog);
  setStatus(`Delta-MTP computed for frame ${frame.frame_index + 1}.`);
}

async function computeDeltaMtpForTrajectory() {
  const trajectory = currentTrajectory;
  if (!trajectory || deltaMtpComputeInFlight) {
    return;
  }

  const storedCount = trajectory.frames.filter(
    (frame) => frame.delta_mtp?.stored
  ).length;
  if (
    storedCount > 0 &&
    !window.confirm(
      `Compute Delta-MTP for all frames will replace ${storedCount} stored result(s). Continue?`
    )
  ) {
    return;
  }

  setDeltaMtpComputeState(true, { allFrames: true });
  setStatus(`Computing Delta-MTP for all ${trajectory.frames.length} frames…`);
  let result;
  try {
    result = await callJson(systemUrl("/delta-mtp"), { method: "POST" });
  } finally {
    setDeltaMtpComputeState(false);
  }

  if (trajectory !== currentTrajectory || !result) {
    return;
  }
  for (const entry of result.frames) {
    const frame = trajectory.frames.find(
      (candidate) => Number(candidate.frame_index) === Number(entry.frame_index)
    );
    if (frame) {
      applyDeltaMtpResult(frame, entry);
    }
  }
  refreshDeltaMtpConsumers(trajectory, result.energy_catalog);

  const parts = [`${result.computed.length} computed`];
  if (result.failed.length) parts.push(`${result.failed.length} failed`);
  const firstFailure = result.failed[0]?.error;
  setStatus(
    `Delta-MTP: ${parts.join(", ")}${firstFailure ? `. ${firstFailure}` : ""}`,
    result.failed.length > 0
  );
}

async function loadDeltaMtpCapability() {
  try {
    deltaMtpCapability = await callJson("/api/delta-mtp/capability");
  } catch (error) {
    deltaMtpCapability = { available: false, reason: error.message };
  }
  renderDeltaMtpControls(selectedFrame());
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

function systemSelectOptions(systems) {
  return systems.map((system) => ({
    label: `${system.system_id} (${system.frame_count} frames)`,
    value: system.slug,
  }));
}

function eligibleSystems() {
  const motif = el("binding-motif-select").value;
  return motif
    ? allSystems.filter((system) => system.binding_motif === motif)
    : allSystems;
}

async function applyMotifFilter() {
  const systemSelect = el("system-select");
  const previousSlug = systemSelect.value;
  const eligible = eligibleSystems();
  fillSelect(systemSelect, systemSelectOptions(eligible), { placeholder: "No systems" });
  if (eligible.some((system) => system.slug === previousSlug)) {
    systemSelect.value = previousSlug;
  }
  // Filtering does not reload an unchanged system. If it became ineligible, the select's
  // first option is now current and must replace the displayed trajectory.
  if (systemSelect.value && systemSelect.value !== previousSlug) {
    await loadTrajectory();
  }
}

async function loadSystems() {
  const uploadId = el("collection-select").value;
  if (!uploadId) {
    return;
  }
  const listing = await callJson(`/api/uploads/${encodeURIComponent(uploadId)}/systems`);
  allSystems = listing.systems;

  const motifField = el("binding-motif-field");
  const motifSelect = el("binding-motif-select");
  const motifs = listing.binding_motifs ?? [];
  motifField.hidden = motifs.length === 0;
  if (motifs.length) {
    fillSelect(
      motifSelect,
      [
        { label: "All binding motifs", value: "" },
        ...motifs.map((motif) => ({
          label: `${motif.name} (${motif.system_count} systems)`,
          value: motif.name,
        })),
      ],
      { placeholder: "All binding motifs" }
    );
    motifSelect.value = "";
  } else {
    motifSelect.replaceChildren();
    motifSelect.disabled = true;
  }

  // A collection change starts at All and at its first system, matching the pre-filter
  // workflow. Selection preservation applies while filtering within this listing.
  fillSelect(el("system-select"), systemSelectOptions(allSystems), {
    placeholder: "No systems",
  });
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

function activateTrajectoryTab() {
  // A resize that happened while this tab was hidden was deliberately skipped. Measure only
  // after selectTab has exposed the panel, preserving the existing camera.
  requestAnimationFrame(() => resizeViewerIfVisible(viewer, "viewer"));
}

async function activateIpdTab() {
  // If Trajectory was playing when the user changed tabs, do not turn it into an implicit IPD
  // separation animation. Separation changes in this tab are manual.
  pausePlayback();
  const frame = selectedFrame();
  renderIpdSection(frame);
  initializeIpdViewer();
  if (ipdViewer && frame) {
    renderIpdFrame(frame);
    requestAnimationFrame(() => resizeViewerIfVisible(ipdViewer, "ipd-viewer"));
  }
  if (frame && ipdMode(frame, currentIpdMode)?.stored) {
    await selectIpdMode(currentIpdMode);
  }
}

function deactivateIpdTab() {
  stopIpdPlayback();
  ipdHistoryRequestToken += 1;
}

// aria-selected is the state; the panel's `hidden` and the CSS both follow it, matching the
// existing Plots-tab lifecycle rather than inventing a second navigation mechanism.
const TABS = [
  { tab: "tab-trajectory", panel: "panel-trajectory", onShow: activateTrajectoryTab },
  {
    tab: "tab-ipd",
    panel: "panel-ipd",
    onShow: activateIpdTab,
    onHide: deactivateIpdTab,
  },
  { tab: "tab-plots", panel: "panel-plots", onShow: () => plots.activate() },
];

function selectTab(id) {
  let activated = null;
  for (const entry of TABS) {
    const active = entry.tab === id;
    const tab = el(entry.tab);
    const wasActive = tab.getAttribute("aria-selected") === "true";
    tab.setAttribute("aria-selected", String(active));
    // Roving tabindex: the tab bar is one stop in the page's tab order, and the arrow keys
    // move within it.
    tab.tabIndex = active ? 0 : -1;
    el(entry.panel).hidden = !active;
    if (!active && wasActive) {
      entry.onHide?.();
    }
    if (active && !wasActive) {
      activated = entry;
    }
  }
  // Measure a tab only after every sibling has received its final hidden state. This is the
  // same activation boundary used by Plotly and the lazy IPD 3Dmol canvas.
  Promise.resolve(activated?.onShow?.()).catch((error) => setStatus(error.message, true));
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
  window.addEventListener("resize", resizeVisibleViewersDebounced);

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
  el("binding-motif-select").addEventListener("change", guard(applyMotifFilter));
  el("system-select").addEventListener("change", guard(loadTrajectory));
  el("play-button").addEventListener("click", togglePlayback);
  // Both physical-frame timelines wire their own buttons to the shared selectFrame path.

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
  el("dipole-comparison-atom-filters").addEventListener("click", (event) => {
    const { filter } = event.target.dataset;
    if (filter) {
      setDipoleComparisonAtomFilter(filter);
    }
  });

  el("show-fragment-a").addEventListener("change", (event) => {
    setFragmentAVisible(event.target.checked);
  });
  el("show-fragment-b").addEventListener("change", (event) => {
    setFragmentBVisible(event.target.checked);
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

  el("ipd-show-monomer-dipoles").addEventListener("change", (event) => {
    setIpdMonomerDipolesVisible(event.target.checked);
  });
  el("ipd-show-dimer-dipoles").addEventListener("change", (event) => {
    setIpdDimerDipolesVisible(event.target.checked);
  });
  el("ipd-show-delta-dipoles").addEventListener("change", (event) => {
    setIpdDeltaDipolesVisible(event.target.checked);
  });
  el("ipd-charge-label-mode").addEventListener("change", (event) => {
    setIpdChargeLabelMode(event.target.value);
  });
  // Permanent overlays only: induced arrows and the SCF iteration remain untouched.
  el("ipd-multipole-arrow-scale").addEventListener("input", () => {
    updatePermanentArrowScaleReadout(
      "ipd-multipole-arrow-scale",
      "ipd-multipole-arrow-scale-readout"
    );
    renderIpdMultipoleOverlays(selectedFrame());
  });

  el("ipd-mode").addEventListener(
    "change",
    guard((event) => selectIpdMode(event.target.value))
  );
  el("ipd-compute").addEventListener("click", guard(computeIpd));
  el("ipd-compute-all").addEventListener("click", guard(computeIpdForTrajectory));
  el("delta-mtp-compute").addEventListener("click", guard(computeDeltaMtp));
  el("delta-mtp-compute-all").addEventListener(
    "click",
    guard(computeDeltaMtpForTrajectory)
  );
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
  guard(loadDeltaMtpCapability)();
}

init();
