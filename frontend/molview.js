// The 3Dmol drawing primitives shared by every tab that renders a dimer.
//
// Extracted from main.js when the Trajectory tab arrived. The arrow geometry here is ported
// from _arrow_cgo (pymol_dipole.py) and global_scale (pymol_dipole_movie.py) and was verified
// to reproduce PyMOL to floating-point precision -- see README.md "Arrow convention". Two
// copies of that would eventually disagree, and a disagreement is a scientifically wrong
// picture rather than a cosmetic bug, which is why this is shared where fillSelect is not.
//
// Imports nothing. Every function takes its viewer, so a page may own more than one.

// Changing these changes the scientific reading of the picture, so they stay named and
// grouped.
export const DEFAULT_ARROW_LEN = 2.0; // Angstrom, longest arrow in the whole history
export const MIN_MU = 1e-6; // below this an arrow is skipped entirely
export const CONE_OVERSHOOT = 1.3; // total arrow length is 1.3 * scale * |mu|
export const CYL_RADIUS = 0.05;
export const CONE_RADIUS = 0.1;
export const ARROW_COLOR = 0xbf00bf; // INDUCED_MODEL_COLOR (0.75, 0.0, 0.75)
export const SPHERE_SCALE = 0.15;
export const STICK_RADIUS = 0.55 * SPHERE_SCALE; // 0.0825

// A WebGL canvas measures its container, and a container inside a hidden tab panel is 0x0.
// Resizing to that collapses the canvas, and it stays collapsed after the panel is shown
// again -- so every path that measures has to check first.
export function isLaidOut(container) {
  return (
    Boolean(container) &&
    !container.closest("[hidden]") &&
    container.clientWidth > 0 &&
    container.clientHeight > 0
  );
}

// XYZ carries no bond table, so 3Dmol guesses bonds by distance: it bonds two atoms whenever
// they sit closer than r1 + r2 + 0.25 A, which for Na-O is 2.52 A and so invents an Na+---O
// stick at every separation up to 1.10 Re. The two monomers of a dimer are non-bonded by
// construction, so any bond crossing the A/B boundary is an artefact of that guess.
//
// Bonds are assigned once when the model is parsed and re-read from atom.bonds on every
// setStyle, so deleting them here -- before the first setStyle -- is all it takes; there is no
// drawn geometry to fix up afterwards.
//
// Returns false when the viewer and the data disagree on atom count, leaving the bonds alone;
// the caller decides how to report that, since this module owns no status line.
export function stripIntermonomerBonds(model, nAtomsA, nAtoms) {
  if (!nAtomsA) {
    // No declared monomer split: there is no boundary to enforce, and inventing one would be
    // worse than leaving 3Dmol's guess alone.
    return true;
  }

  // An empty selection returns the model's atoms, by reference, in the same index order that
  // atom.bonds refers to. The XYZ parser does not set atom.index, so position *is* the index
  // -- which only holds while the selection is everything.
  const atoms = model.selectedAtoms({});
  if (atoms.length !== nAtoms) {
    return false;
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
  return true;
}

// Spheres and thin sticks, the one nuclei style in the app.
export function applyMoleculeStyle(viewer) {
  viewer.setStyle(
    {},
    {
      sphere: { scale: SPHERE_SCALE, colorscheme: "greenCarbon" },
      stick: { radius: STICK_RADIUS, colorscheme: "greenCarbon" },
    },
  );
}

// Reproduces _arrow_cgo (pymol_dipole.py:158-193): the arrow *begins* at the atom centre
// rather than being centred on it, and the cone is added on top of the full shaft, so the tip
// reaches CONE_OVERSHOOT * scale * |mu|.
//
// `mu` is one dipole per atom, in the same order as `coords`.
export function addDipoleArrows(viewer, coords, mu, scale) {
  for (let i = 0; i < mu.length; i += 1) {
    const vector = mu[i];
    const magnitude = Math.hypot(vector[0], vector[1], vector[2]);
    if (magnitude < MIN_MU) {
      // A zero-length shaft plus a degenerate cone renders as a stray speck that flickers
      // frame to frame.
      continue;
    }
    const [x, y, z] = coords[i];
    const reach = CONE_OVERSHOOT * scale;
    viewer.addArrow({
      start: { x, y, z },
      end: {
        x: x + reach * vector[0],
        y: y + reach * vector[1],
        z: z + reach * vector[2],
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
