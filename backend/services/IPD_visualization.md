# GOAL: Add an IPD-specific analysis workflow to existing trajectory tab

IPD should be treated as a secondary analysis of the currently selected trajectory frame, but it should use its own dedicated 3D viewer.

The final conceptual hierarchy should be:

Collection
    ↓
System / trajectory
    ↓
Selected physical frame
    ├── Main trajectory viewer
    │      ├── geometry
    │      ├── monomer MBIS dipoles
    │      ├── dimer MBIS dipoles
    │      ├── Δ MBIS dipoles
    │      └── charge labels
    │
    ├── Multipole table
    │
    └── IPD analysis
           ↓
       separate IPD viewer
           ↓
       selected IPD iteration
           ↓
       induced dipole arrows

The two viewers must always correspond to the same physical trajectory frame.

The main viewer answers:

*What does this physical frame look like, and how do the isolated-monomer and dimer MBIS multipoles differ?*

The IPD viewer answers:

At this fixed geometry, how does the classical induced-dipole solution evolve through the SCF iterations?

These should remain distinct in both the UI and JavaScript.

## Phase 1

### Preserve the Existing Trajectory Architecture

The existing responsibilities should remain: 

showFrame() = orchestrates everything associated with the selected physical frame

renderFrame() = renders geometry in MAIN trajectory viewer

IPD should *extend* this structure rather than replace it.

## Add a Dedicated IPD viewer

Add:

`let ipdViewer = null;`

Keep the current:

viewer

as the main trajectory / multipole viewer.

Do not render IPD arrows in the main viewer.

This avoids mixing:

monomer MBIS dipoles
dimer MBIS dipoles
Δ MBIS dipoles
charge labels
IPD induced dipoles

in one crowded visualization.

The two viewers should have separate responsibilities:

viewer
    → physical trajectory / MBIS analysis

ipdViewer
    → IPD induced-dipole convergence

## Put the IPD viewer inside a Collapsible IPD section

The UI should conceptually look like: 

TRAJECTORY

0.70 Re       0.80 Re       0.90 Re       1.00 Re
   ○────────────○──────────────●──────────────○

Selected frame:
0.90 Re | closest contact 2.18 Å


┌──────────────────────────────────────────────┐
│              MAIN 3D VIEWER                  │
│                                              │
│   geometry + MBIS multipole overlays         │
└──────────────────────────────────────────────┘

▼ Atomic Multipoles
   ...


▼ IPD

   Radius mode: [ MBIS ▼ ]

   Initial      1      2      3      4      Converged
      ●─────────○──────○──────○──────○──────────○

   ┌──────────────────────────────────────────┐
   │            IPD 3D VIEWER                 │
   │                                          │
   │      fixed selected-frame geometry       │
   │      induced dipoles vary by SCF         │
   └──────────────────────────────────────────┘

   Iteration 0 / 5

   [ ▶ Play ]


## Lazy-Initialize the IPD viewer

```
function initializeIpdViewer() {
    if (ipdViewer) {
        return;
    }

    ipdViewer = window.$3Dmol.createViewer(
        el("ipd-viewer"),
        { backgroundColor: "white" }
    );

    const frame = selectedFrame();

    if (frame) {
        renderIpdFrame(frame, {
            resetCamera: true,
        });
    }
}
    
```

## Keep Main and IPD Viewer Cameras Independent Initially

Do not synchronize camera rotation between the two viewers in the first implementation.

Independent cameras are simpler and may actually be useful.

Do not add automatic camera synchronization unless it becomes a demonstrated usability need later.

A future optional feature could be:

[ Match main-view orientation ]

but that is not part of the initial implementation.

## Reuse Existing Geometry Logic

Create an IPD-specific geometry renderer similar to the existing renderFrame().

```
function renderIpdFrame(
    frame,
    { resetCamera = false } = {}
) {
    if (!ipdViewer || !frame.xyz) {
        return;
    }

    ipdViewer.removeAllModels();

    const model =
        ipdViewer.addModel(frame.xyz, "xyz");

    stripIntermonomerBonds(
        model,
        frame.n_atoms_A,
        frame.n_atoms
    );

    ipdViewer.setStyle(
        {},
        {
            sphere: { scale: SPHERE_SCALE },
            stick: { radius: STICK_RADIUS },
        }
    );

    if (resetCamera) {
        ipdViewer.zoomTo();
    }

    ipdViewer.render();
}
    
```

## Use Explicit Dots for Trajectory Timeline

Move toward ecplixit clicable dots for physical trajector

Move toward explicit clickable dots for physical trajectory frames.

For example:

Trajectory

0.70       0.80       0.90       1.00       1.10
 ○──────────○──────────●──────────○──────────○

Each dot has exactly one meaning:

Select this physical geometry.

Every dot should call:

selectFrame(index);

Do not make trajectory dots directly open IPD.

The IPD panel automatically updates to reflect whichever frame is selected.


## Keep `selectFrame()` as manual selection path

Add or preserve

```
function selectFrame(index) {
    pausePlayback();

    const slider = el("frame-slider");

    if (slider) {
        slider.value = index;
    }

    showFrame(index);
}
```

## Use Explicit Dots for IPD iterations

Do not use a slider, use explicit clickable dots. Each dot should call `selectIpdIteration(index)`

IPD convergence

Initial       1       2       3       4       Converged
   ●──────────○───────○───────○───────○──────────○


## Treat the Two Timelines at Different Levels

The two timelines must have different meanings.

Trajectory timeline

Changes physical geometry and therefore updates:

1. main viewer geometry
2. IPD viewer geometry
3. multipole table
4. MBIS viewer overlays
5. closest-contact metadata
6. energies later
7. IPD availability/modes

IPD iteration timeline

Keeps geometry fixed and updates only:

1. induced-dipole arrows
2. selected iteration indicator
3. iteration-specific metadata

This distinction is central.

## Add Minimal IPD State

Add:

```
let currentIpdMode = null;
let currentIpdIteration = 0;
let ipdPlaybackTimer = null;
let ipdShapes = [];
```

Do not add a generalized state system.

## Keep IPD shapes separate from Main Viewer overlays

The main viewer may already have:

multipoleShapes
chargeLabels

for permanent MBIS information.

The IPD viewer should have:

ipdShapes

only.

Conceptually:

MAIN VIEWER
    geometry model
    multipoleShapes
    chargeLabels

IPD VIEWER
    geometry model
    ipdShapes

## Add Lightweight IPD metadata to Normal Trajectory JSON

The normal trajectory response should tell the browser what IPD functionality exists for each frame.

Example:

```
{
  "frame_index": 3,
  "ipd": {
    "available": true,
    "computable": true,
    "modes": [
      {
        "id": "mbis",
        "label": "MBIS radius",
        "iteration_count": 8
      },
      {
        "id": "vdw",
        "label": "vdW radius",
        "iteration_count": 6
      }
    ]
  }
}
```

If IPD has not yet been computed:

```
"ipd": {
  "available": false,
  "computable": true,
  "modes": []
}
```

14. Do NOT put every IPD history in main trajectory JSON.

The main trajectory JSON should remain fast enough to load immediately.

IPD history can scale as:

frames
× radius modes
× iterations
× atoms
× 3 dipole components

Use the normal trajectory response for availability, computability, modes, and iteration counts.

Lazy-load detailed history only when required.

15. Add IPD History Endpoint

Conceptually use

`GET /api/uploads/<upload_id>/systems/<system_slug>/frames/<frame_index>/ipd?mode=<mode>`

The response should contain one selected frame and radius mode.

Example:

```
{
  "mode": "mbis",
  "energy": -4.52,
  "mu_history": [
    [
      [0.01, 0.02, 0.03],
      [0.02, 0.01, 0.04]
    ],
    [
      [0.02, 0.03, 0.04],
      [0.03, 0.02, 0.05]
    ]
  ]
}
```
Each iteration should already be: (n_atoms, 3) in the exact atom order of geometry.

Do any A/B concatenation in Python. 

16. Let `radius_thole` remain source of truth.
Responsibilities should remain:

dataframe_schema.py
    → uploaded input vocabulary

radius_thole.py
    → IPD result vocabulary and computation

IPD serialization/service
    → translate stored IPD results into frontend-friendly JSON


17. Cache Loaded IPD History in Browser

After loading a frame/mode once, cache it locally.

Conceptually:

```
async function loadIpdHistory(frame, mode) {
    frame.ipd.history ??= {};

    if (frame.ipd.history[mode]) {
        return frame.ipd.history[mode];
    }

    const history =
        await callJson(buildIpdHistoryUrl(
            frame,
            mode
        ));

    frame.ipd.history[mode] = history;

    return history;
}
```

so first inspection performs an HTTP request, later inspection uses cached history.





