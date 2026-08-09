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


## Cache Loaded IPD History in Browser

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

## IPD Panel has Three States

State A — Not computable

▼ IPD

IPD is unavailable for this frame because the
required multipole inputs are missing.

State B — Computable but not stored

▼ IPD

No stored IPD calculation for this frame.

Damping mode: [ MBIS ▼ ]

[ Compute IPD ]

State C — Stored history available

▼ IPD

Damping mode: [ MBIS ▼ ]

Initial      1      2      3      4      Converged
   ●─────────○──────○──────○──────○──────────○

┌─────────────────────────────────────────┐
│             IPD VIEWER                  │
│                                         │
│ induced dipoles for selected iteration  │
└─────────────────────────────────────────┘

Iteration 0 / 5

Energy: -4.52 kcal/mol

[ ▶ Play ]

Damping Mode will be Thole Damping with universal thole damping value 0.39 or Thole Damping with combination rule with scaled TS-vdw radii (inter only)


## Implement renderIPDSection(frame) 

This function manages

1. availability text
2. compute controls
3. mode selector
4. iteration timeline
5. IPD viewer visibility

## Update Both Viewers wehn Physical Frame changes

Extend `showFrame()` to

```
function showFrame(
    index,
    { resetCamera = false } = {}
) {
    const frame =
        currentTrajectory?.frames[index];

    if (!frame) {
        return;
    }

    stopIpdPlayback();

    currentFrameIndex = index;
    currentIpdIteration = 0;

    // Main viewer
    renderFrame(frame, { resetCamera });

    el("frame-label").textContent =
        frameLabel(frame);

    renderTimelineSelection();
    renderMultipoleSection(frame);
    renderMultipoleOverlays(frame);

    // IPD
    renderIpdSection(frame);

    if (ipdViewer && isIpdSectionOpen()) {
        renderIpdFrame(frame, {
            resetCamera,
        });
    }
}
```

## Reset IPD State on Physical Frame Change

If the user switches:

0.90 Re → 1.00 Re

then reset:

currentIpdIteration = 0;

Stop IPD playback.

Clear old IPD arrow shapes.

Update the IPD panel for the newly selected frame.

## Build IPD Dot Timeline with 

Use real buttons for each iteration

```
function renderIpdTimeline(iterationCount) {
    const timeline = el("ipd-timeline");

    timeline.replaceChildren();

    for (
        let index = 0;
        index < iterationCount;
        index += 1
    ) {
        const button =
            document.createElement("button");

        button.type = "button";
        button.className = "ipd-dot";

        button.setAttribute(
            "aria-label",
            `Select IPD iteration ${index}`
        );

        button.addEventListener(
            "click",
            () => selectIpdIteration(index)
        );

        timeline.append(button);
    }
}
```

## Handle Long IPD Histories Gracefully

For small histories:

Initial   1   2   3   4   Converged

is fine.

For longer histories, do not label every dot.

For example:

Initial                                Converged
   ●──○──○──○──○──○──○──○──○──○──○────○

Iteration 6 / 12

The dots remain individually clickable.

## Implement selectIPDIteration()

Manual IPD selection should go through:

```
function selectIpdIteration(index) {
    stopIpdPlayback();
    showIpdIteration(index);
}
```

## Implement `showIpdIteration()`

This should be the single entry point for IPD-iteration-dependent UI.

Conceptually:

```
function showIpdIteration(index) {
    const frame = selectedFrame();
    const history = currentIpdHistory();

    if (!frame || !history) {
        return;
    }

    if (
        index < 0 ||
        index >= history.mu_history.length
    ) {
        return;
    }

    currentIpdIteration = index;

    renderIpdDipoles(
        frame,
        history.mu_history[index]
    );

    renderIpdTimelineSelection();

    el("ipd-iteration-label").textContent =
        `Iteration ${index + 1} / ${history.mu_history.length}`;
}

```
This function must not call:

showFrame()
renderFrame()
renderIpdFrame()

because geometry has not changed.

# Render IPD Dipoles Only in `ipdViewer`

Track:

let ipdShapes = [];

Clear them with ipdViewer.removeShape(...).

Then render the selected iteration's arrows into ipdViewer.

Do not recreate the molecule model at each iteration.

Only replace arrows.

This preserves molecular geometry, rotation, zoom, and camera orientation while inspecting convergence.

## Damping Mode Selector

If multiple IPD modes exist:

Damping mode:
[ Thole (0.39 inter + intra) ▼ ]

Changing mode should:

1. stop IPD playback
2. clear current IPD arrows
3. set currentIpdMode
4. lazy-load that mode's history
5. rebuild iteration dots
6. select iteration 0

Conceptually:

```
async function selectIpdMode(mode) {
    stopIpdPlayback();
    clearIpdDipoles();

    currentIpdMode = mode;
    currentIpdIteration = 0;

    const frame = selectedFrame();

    const history =
        await loadIpdHistory(frame, mode);

    renderIpdTimeline(
        history.mu_history.length
    );

    showIpdIteration(0);
}
```

## Add IPD Playback Only after Dot Selection Works

Manual dot switching should work first.

Then add:

startIpdPlayback()
stopIpdPlayback()
toggleIpdPlayback()

Playback should call:

showIpdIteration(next)

rather than duplicating iteration-rendering logic.

## Compute-IPD flow
If:

frame.ipd.available === false

but:

frame.ipd.computable === true

show:

[ Compute IPD ]

The frontend request should identify only the stored system/frame and requested mode.

Conceptually:

POST /api/uploads/<upload_id>/systems/<system_slug>/frames/<frame_index>/ipd

with:

{
  "mode": "Thole (0.39 inter + intra)"
}

Do not send the whole molecule or dataframe row back if the backend already owns it.

## Reuse the Existing IPD Physics Backend

The flow should be:

selected stored frame
        ↓
backend IPD endpoint
        ↓
existing IPD service
        ↓
radius_thole
        ↓
results

Do not duplicate radius_thole logic in frontend or generic upload modules.

## Persist Newly Computed IPD results

A computed history must survive refresh.

Conceptually:

processed backend dataframe / parquet
        ↓
radius_thole computation
        ↓
write IPD result columns
        ↓
persist updated backend representation
        ↓
regenerate / refresh frame IPD metadata

The frontend cache is not persistence.

## After Computation, Open New History immediately

Conceptually:

const result =
    await computeIpd(frame, mode);

frame.ipd.available = true;
frame.ipd.modes = result.modes;

await selectIpdMode(result.mode);

The user should transition directly from:

Compute IPD

to:

Initial ○ ○ ○ Converged
+ IPD viewer arrows

without a page reload.

## Keep IPD View Controls Specific to IPD

Main viewer controls can remain focused on MBIS overlays:

MBIS dipoles
☐ Monomer
☐ Dimer
☐ Δ

Charge labels
○ None
○ Monomer
○ Dimer
○ Δ

IPD viewer controls should focus on:

Damping mode
Iteration
IPD dipole scale
Playback

Do not merge them into one giant viewer control group.

## Recommended Javascript Structure

Evolve the existing file to roughly:

// --- viewer -----------------------------------------------------------------

viewer
ipdViewer

stripIntermonomerBonds()
renderFrame()
renderIpdFrame()
initializeIpdViewer()


// --- frames -----------------------------------------------------------------

currentTrajectory
currentFrameIndex

frameLabel()
selectedFrame()
selectFrame()
showFrame()
setTrajectory()


// --- trajectory timeline ----------------------------------------------------

renderTrajectoryTimeline()
renderTimelineSelection()


// --- multipoles -------------------------------------------------------------

renderMultipoleSection()
...


// --- viewer overlays --------------------------------------------------------

renderMultipoleOverlays()
...


// --- trajectory playback ----------------------------------------------------

playbackTimer
startPlayback()
pausePlayback()
togglePlayback()


// --- IPD --------------------------------------------------------------------

currentIpdMode
currentIpdIteration
ipdPlaybackTimer
ipdShapes

renderIpdSection()

selectIpdMode()
loadIpdHistory()
currentIpdHistory()

renderIpdTimeline()
renderIpdTimelineSelection()

selectIpdIteration()
showIpdIteration()

clearIpdDipoles()
renderIpdDipoles()

startIpdPlayback()
stopIpdPlayback()
toggleIpdPlayback()

computeIpd()


// --- loading ----------------------------------------------------------------

...


// --- wiring -----------------------------------------------------------------

...

Do not split into many frontend files unless this file becomes genuinely difficult to navigate.

# Recommended Implementation Order

1. Add explicit trajectory dots and verify each manual selector reaches selectFrame(index).
2. Add IPD metadata to serialized frames.
5. Add the IPD history endpoint.
3. Add the collapsible IPD section with not-computable / computable / available states.
4. Lazy-initialize ipdViewer when the IPD section opens.
5. Add radius/mode selection and lazy loading.
6. Build the IPD iteration dot timeline.
7. Implement selectIpdIteration() and showIpdIteration().
8. Integrate the existing induced-dipole arrow rendering into ipdViewer.
9. Synchronize physical frame changes so both viewers always use the same geometry.
10. Add Compute IPD for eligible frames lacking stored history.
11. Persist results.
12. Add IPD playback last.

# Final Design Rule

Maintain these four distinct functions:

selectFrame(index)
    → user chooses physical geometry

showFrame(index)
    → synchronize everything dependent on physical geometry

selectIpdIteration(index)
    → user chooses one SCF step

showIpdIteration(index)
    → update only IPD-iteration-dependent visualization

And maintain two distinct viewer responsibilities:

MAIN VIEWER
    → geometry + MBIS analysis

IPD VIEWER
    → fixed geometry + induced-dipole convergence



---

# As-built — backend

Everything above is the specification. This section is what was actually built, added after the
backend half shipped so the two can be compared. The frontend half is **not** built.

Nothing above this line was edited — where the spec and the code disagree, see
*Alignment and divergence* at the end.

## Module layout

```
thole_damping ← ipd_results ← system_serialization ← ipd_service ← app
capability    ← ipd_service
trajectory_service ← ipd_service
```

Acyclic, and the shape is forced by one constraint: `system_serialization` imports
`ipd_results` to attach IPD metadata to every frame it writes, so `ipd_results` must not reach
back for the collection writer. That is why it touches no disk at all.

| Module | Owns | Public surface |
|---|---|---|
| `capability.py` | Whether *this server* can compute | `capability(refresh=False)`, `require_capability()`, `missing_ipd_kwargs(fn)`, `REQUIRED_IPD_KWARGS` |
| `ipd_results.py` | Mode vocabulary; stored results → JSON. No disk, no Flask, no science | `MODES`, `mode_parameterization(id)`, `mode_label(id)`, `frame_ipd_metadata(df, row_index, *, n_atoms, n_atoms_A, computable)`, `frame_ipd_history(df, row_index, mode_id)` |
| `ipd_service.py` | Disk, compute, persistence, locking | `get_history(upload_id, slug, frame_index, mode_id)`, `compute(upload_id, slug, frame_index, mode_id)`, `compute_system(upload_id, slug, mode_id)`, `NotFound` |

Two existing modules changed:

* `system_serialization._frame_payload` now takes `(df, frame, *, ipd_computable)` — the whole
  dataframe, not just the row, because IPD availability is read through `thole_damping`'s
  label-based accessors. It attaches `payload["ipd"]`, or `None` for a frame with no geometry.
* `dataframe_schema.SCHEMA_VERSION` is **3**. Bumping it is the mechanism that makes an
  already-stored collection reprocess instead of being handed back untouched; without the bump
  no existing collection would ever grow an `ipd` block.

`radius_thole.py` was not touched. It is reference material.

## The two damping modes

`ipd_results.MODES` is the only place a mode id, its label and its `thole_damping` arguments
meet. One `**parameterization` splat serves naming, reading, writing and computing, so a mode
cannot be read under one parameterization and written under another.

| id | label | inter | intra |
|---|---|---|---|
| `thole_0.39` | Thole 0.39 (inter + intra) | `fixed_0.39` | `fixed_0.39` |
| `ts_mbis_inter` | Thole TS-vdW radii (inter only) | `ts_mbis_radii` | `fixed_0.39` |

`thole_damping` supports two further combinations (`ts_mbis_radii` on intra, and on both). Each
is one entry in this dict; they are omitted because the spec asks for these two.

## HTTP surface

```
GET  /api/ipd/capability
GET  /api/uploads/<upload_id>/systems/<slug>/frames/<int:frame_index>/ipd?mode=<id>
POST /api/uploads/<upload_id>/systems/<slug>/frames/<int:frame_index>/ipd   {"mode": "<id>"}
POST /api/uploads/<upload_id>/systems/<slug>/ipd                            {"mode": "<id>"}
```

`/trajectory` is unchanged, but every frame in it now carries an `ipd` block.

Failures come back through `errors.IpdError`, which already existed and was reused rather than
redefined. It carries `code`, `details`, `retryable` and `user_fixable` alongside the `error`
key every other failure in this app uses, so one client-side handler reads them all.

| Status | When |
|---|---|
| 400 | `missing_ipd_mode`, `unknown_ipd_mode`, `ipd_inputs_missing` |
| 404 | `ipd_not_computed`, `no_geometry`, or no such frame/system/upload |
| 409 | `ipd_history_mismatch` — stored history does not fit the geometry |
| 500 | `ipd_computation_failed` — the SCF raised; the original text is kept |
| 503 | `apnet_pt_missing` / `_broken` / `_incompatible` |

## Payload shapes

`frame.ipd`, verbatim from a stored collection:

```json
{
  "computable": true,
  "available": true,
  "modes": [
    {"id": "thole_0.39", "label": "Thole 0.39 (inter + intra)",
     "stored": true, "iteration_count": 26, "energy": -28.60868263244629,
     "converged": true, "problem": null},
    {"id": "ts_mbis_inter", "label": "Thole TS-vdW radii (inter only)",
     "stored": true, "iteration_count": 23, "energy": -5.333324432373047,
     "converged": true, "problem": null}
  ]
}
```

Every mode is listed whether or not it has a result, so the dropdown does not change shape as
frames are walked and a mode that has not been run can still be offered for compute.
`available` is any mode stored; `computable` is the `ipd_computable` feature.

`iteration_count` is the number of **history entries**, which is the number of dots on the
timeline — entry 0 is the pre-SCF seed the UI labels "Initial". `thole_damping` counts the same
array as `iterations = len - 1`; the name is chosen to not be that.

`problem` is non-null only when a stored history does not describe this geometry.

The history endpoint:

```json
{"mode": "thole_0.39", "label": "Thole 0.39 (inter + intra)",
 "energy": -28.608683, "iteration_count": 26, "converged": true,
 "n_atoms": 4, "n_atoms_A": 3,
 "mu_history": [[[x,y,z], ...4 atoms...], ...26 iterations...]}
```

`POST` on a frame returns `{"frame_index", "ipd", "history"}` — the refreshed availability block
*and* the new history, so the UI goes straight from Compute to arrows without a second request.
`POST` on a system returns `{"mode", "label", "computed", "skipped", "failed", "frames"}`, with
per-frame `ipd` blocks and no histories.

## Invariants

Each of these, if broken, produces output that looks right and is wrong.

* **`mu_history[i]` is `(n_atoms, 3)` in geometry atom order** — monomer A then B, joined by
  `np.concatenate(..., axis=1)` in `ipd_results`. Atom `i` is in A iff `i < n_atoms_A`. Joined
  in Python because atom order is a backend invariant; reconstructing it in the browser would be
  a second place for that rule to live.
* **`_history_problem` runs before the join.** It checks rank, matching iteration counts,
  3-vectors, and both atom counts against the geometry. Metadata reports the failure in
  `problem` and marks the mode unstored; `frame_ipd_history` refuses with a 409. A history with
  the wrong atom count would otherwise concatenate into an array of the right rank and the wrong
  meaning — real numbers on the wrong nuclei, with nothing downstream able to notice.
* **Only `thole_damping.ipd_column_names` is read.** Uploaded pickles carry histories under
  `radius_thole`'s older names (`IPD (MBIS) 0.39 all`, `IPD (radius thole ts_mbis inter only)`,
  …) and those are ignored on purpose. Every history the app shows is one it computed and filed
  under a name that states its own parameterization, so a displayed result can always be traced
  to the choices that made it.
* **`capability()` and `frame.ipd.computable` are different questions.** The first is about the
  server, the second about the data. They fail independently and deserve different sentences.
* **Writes take a per-collection lock.** `_write_atomic` makes each file replacement atomic, but
  a compute is a read-modify-write spanning `processed.pkl` and every system JSON with an SCF in
  the middle. Two concurrent computes without the lock would both start from the same dataframe
  and the second would erase the first — silently, and unrecoverably.
* **`frame_index` ≠ `row_index`.** A frame index is a position in the animation (sorted by
  separation); a row index is a position in the dataframe (build order). The mapping was decided
  at upload and is read from the stored system JSON, and the dataframe *label* is obtained as
  `df.index[position]` rather than assumed equal to the position.

Persistence re-runs `save_collection` rather than patching one frame's JSON, so there is exactly
one code path producing a stored collection. `add_derived_columns` is deliberately not re-run —
`processed.pkl` already carries its output.

## Measured

Numbers behind the design choices, and the values that show it is correct.

* One SCF-capped history is **279 KB** of JSON; the whole Na-benzene trajectory payload is
  **62 KB**. Inlining all seven histories would have been ~2 MB against 62 KB — this is why
  availability rides in the trajectory payload and history does not.
* `na_water_dimers.pkl` frame 0: `thole_0.39` → **−28.608683** (26 entries), `ts_mbis_inter` →
  **−5.333324** (23 entries). Both match the `radius_thole` parity run bit for bit, which is
  what confirms each mode id maps to the parameterization it claims.
* `131_Na-benzene.pkl`: 5 of 7 frames hit the 200-iteration cap and report `converged: false`;
  the other two converge at 32 and 28. Whole-trajectory compute takes ~2 s.
* Four concurrent POSTs against one collection (2 frames × 2 modes) all landed.

---

# Alignment and divergence

## Followed as specified

§14's lightweight metadata block. §15's endpoint URL verbatim, and its instruction that each
iteration arrive as `(n_atoms, 3)` in geometry atom order with the A/B concatenation done in
Python. §16's three-way responsibility split. The three panel states. The POST body carrying
only `{"mode": ...}` and never the molecule. The recommended implementation order.

## Differed, and why

| Spec says | Built | Why |
|---|---|---|
| §16 `radius_thole` is the result vocabulary | `thole_damping` | The spec predates that module. `radius_thole.py` is reference only and was not touched. |
| §14 mode ids `"mbis"` / `"vdw"` | `thole_0.39` / `ts_mbis_inter` | The line closing State C names the two physical choices — universal 0.39, and the TS-vdW combination rule inter-only. §14's example reads as written before the modes settled. |
| §14 lists only stored modes | Lists every mode with a `stored` flag | A mode that vanishes when it has no result cannot be offered for compute, and the dropdown would change shape between frames. Same reasoning as explaining a disabled tab instead of hiding it. |
| — | `POST …/systems/<slug>/ipd` | Trajectory-level compute. Per-frame alone means one press and one full collection rewrite per frame; this amortizes both. Skips frames already computed, records per-frame failures, keeps going. |
| — | `capability.py`, the per-collection lock, the history/geometry shape check | From CLAUDE.md's target layout and *Invariants that must survive*, not from this spec. |
| History body: `mode`, `energy`, `mu_history` | Plus `label`, `iteration_count`, `converged`, `n_atoms`, `n_atoms_A` | The viewer needs the A/B split to colour or filter arrows, and `converged` is what distinguishes a real result from one that ran into the iteration cap. |

## Where the implementation plan was wrong

Recorded because these are the kind of thing that otherwise gets re-litigated:

* Estimated ~150 KB per capped history. Actual: **279 KB** — JSON floats are considerably
  wordier than the underlying array.
* Planned to rename `system_serialization._jsonable` → `jsonable` for reuse. Not needed:
  `ipd_results` needed a *vectorized* converter for 201×13×3 arrays, not a per-element
  recursion, so it has its own `_jsonable_array`.
* Planned a new `IpdError` in `ipd_service`. `errors.py` already had exactly it — `code`,
  `status`, `details`, `retryable`, `user_fixable`, `to_dict()` — so it was reused.

Also corrected against an earlier claim: `131_Na-benzene` row 0 does **not** hit the iteration
cap. It converges at 32. Rows 1–5 are the ones at the cap.

## Not built

The entire frontend half. `viewer.js` has no `ipdViewer`, no `renderIpdSection`, no iteration
timeline, and the trajectory still uses the frame slider rather than clickable dots. The
*Recommended Implementation Order* above still stands for that work, minus items 2, 5 and 11 —
the metadata, the history endpoint and persistence — which are the backend work now done.

Registration-time IPD computation is deliberately **off**, contradicting CLAUDE.md's earlier
"opportunistic at registration" decision. Every frame starts in State B and compute is an
explicit action, per this spec's three-state panel. The trajectory-level POST is what makes that
practical.

---

# As-built — frontend

Added after the frontend half shipped, alongside the backend section above. As there, nothing in
the specification body was edited; where the two disagree, see *Alignment and divergence —
frontend* at the end.

## What changed, file by file

| File | Change |
|---|---|
| `frontend/viewer.js` | The arrow/geometry parameterization below; a `// --- trajectory timeline ---` section; a `// --- IPD ---` section of ~330 lines (availability, history + cache, iteration timeline, arrows, playback, compute); `showFrame` extended to reset IPD state and drive the second viewer |
| `frontend/index.html` | `#frame-slider` replaced by `#frame-timeline`; new `<details id="ipd-section">` holding `#ipd-message`, `#ipd-controls` (`#ipd-mode`, `#ipd-compute`, `#ipd-compute-all`) and `#ipd-display` (`#ipd-timeline`, `#ipd-viewer`, `#ipd-play`, `#ipd-iteration-label`, `#ipd-arrow-scale`) |
| `frontend/viewer.css` | `.timeline` / `.timeline-dot` / `.timeline-range`, ported from the previous frontend's `styles.css`; `.ipd*` panel rules; a height for `#ipd-viewer` |

**Three existing functions changed signature, and that is the whole of the edit to previously
working behaviour:**

```
addDipoleArrows(coords, vectors, color)
  -> addDipoleArrows(target, coords, vectors, color, scale, sink)

renderFrame(frame, {resetCamera})
  -> thin wrapper over new drawGeometry(target, frame, {resetCamera})

currentArrowScale()
  -> built on new arrowScale(base, sliderId)
```

plus `isLaidOut` lifted from `molview.js`. Nothing in the multipole table or the MBIS overlay
code was touched. This was done first, as a no-visible-change step, so that anything that broke
afterwards could not be blamed on it.

It is also a restoration rather than an invention: `molview.js` states at line 9 that *"every
function takes its viewer, so a page may own more than one"*, and its `addDipoleArrows` already
took `(viewer, coords, mu, scale)`. The single-viewer version in `viewer.js` had regressed that,
and a second viewer is exactly the case the original signature existed for.

## The four functions

The spec's closing design rule, as implemented:

| Function | Does |
|---|---|
| `selectFrame(index)` | user picks a geometry — pauses playback, calls `showFrame` |
| `showFrame(index)` | everything that depends on the physical frame, both viewers |
| `selectIpdIteration(index)` | user picks an SCF step — stops IPD playback, calls `showIpdIteration` |
| `showIpdIteration(index)` | **arrows only** |

`showIpdIteration` calls no model-rebuilding function: no `removeAllModels`, no `zoomTo`, no
`renderIpdFrame`. That is what lets a walk through the SCF read as convergence — the camera,
zoom and rotation the user set to look at it survive every step.

`renderFrame` and `renderIpdFrame` are kept as distinct names because their callers mean
different things, but both delegate to `drawGeometry`. Two copies of the bond-stripping and
styling would eventually disagree, and a disagreement there is a wrong picture rather than a
cosmetic bug.

## State

```js
let ipdViewer = null;
let currentIpdMode = null;
let currentIpdIteration = 0;
let ipdPlaybackTimer = null;
let ipdShapes = [];
let ipdBaseArrowScale = 1.0;
```

`ipdShapes` stays separate from `multipoleShapes` even though the IPD viewer carries only one
layer. It costs nothing and it is the discipline that stops the two layers clearing each other's
work if a second layer is ever added.

Induced-dipole arrows are `0xbf00bf`, the induced colour from the previous frontend, and
deliberately none of `DIPOLE_COLORS` — those are *permanent* MBIS quantities, and reusing one
would say "induced" about an input.

## Panel states

`renderIpdSection(frame)` reads `frame.ipd` and nothing else:

| Condition | Shown |
|---|---|
| `computable: false` | "IPD is unavailable for this frame because the required multipole inputs are missing." |
| selected mode has `problem` | what the backend found wrong with the stored history |
| `stored: false` for the selected mode | "No stored IPD calculation for this frame." + Compute |
| `stored: true` | timeline, viewer, iteration label, energy, play |

Evaluated **per mode per frame**, not per frame: one mode can be stored while the other is not,
and the panel offers Compute for whichever is missing.

## Constraints that shaped the code

Each of these, if ignored, produces something that works in a demo and fails in use.

* **`renderIpdSection` must stay synchronous.** `showFrame` runs on every 500 ms playback tick,
  so anything there that fetched would issue one request per frame per lap. Histories load only
  on an explicit action — opening the panel, choosing a mode, computing — or on a hand-driven
  frame change, gated on `playbackTimer === null`. Measured: **0 requests** across ~6 frames of
  playback.
* **Lazy viewer creation needs two triggers, not one.** A WebGL canvas measures its container,
  and in states A and B `#ipd-viewer` sits inside a hidden div at 0x0. Initialising only on the
  `<details>` toggle produced a viewer that was never created at all; `initializeIpdViewer` is
  therefore also called at the point State C makes the panel visible. This was a real bug found
  in testing, not a hypothetical.
* **A compute replaces `frame.ipd` wholesale**, which would drop the cached histories hanging
  off it. `applyIpdMetadata` carries the `history` sub-object across — the geometry did not
  change, so those entries are still valid.
* **A mode the user chose is kept even when it has no result here.** An earlier version fell
  back to whichever mode was stored, which silently reversed the selection and made Compute
  unreachable for the other one. `preferredIpdMode` now falls back only for an unset or
  unrecognised selection.
* **The history cache is not persistence.** It lives on the frame object for the session; the
  server holds the computed result.

## Measured

* `na_water_dimers` frame 0: 26 iteration dots, **−28.609 kcal/mol** — the same value the
  `radius_thole` parity run produced, now travelling the whole stack to the label.
  `ts_mbis_inter` on the same frame: 23 dots, **−5.333 kcal/mol**.
* `131_Na-benzene` frame 1: 201 entries, range input rather than dots, labelled **not
  converged**.
* Revisiting a mode already loaded costs **0 requests**.
* Verified in headless Chromium over CDP: 24 of 25 checks from a clean state, zero console
  errors. (The one failure was the harness asserting State B against results left on disk by a
  previous run.)

One caution for anyone verifying arrows by screenshot: at 1x scale on Na-benzene the arrows are
nearly invisible, because the default camera looks straight down the C6 axis and ~80% of each
induced dipole points into the screen, on a shaft two pixels wide. They are drawn correctly —
they scale up cleanly with the slider and are plainly visible once rotated. Absence in a
face-on screenshot is not evidence of absence.

---

# Alignment and divergence — frontend

## Followed as specified

The two-viewer split with no camera synchronisation. The four-function design rule. Lazy
`initializeIpdViewer`. The three panel states. Clickable trajectory dots, every one calling
`selectFrame(index)`. The per-frame history cache, in the shape the spec sketches. `ipdShapes`
kept separate. `showIpdIteration` as the single iteration entry point. Playback added last,
calling `showIpdIteration` rather than duplicating the render. Single file, per the spec's
instruction not to split without cause.

## Differed, and why

| Spec | Built | Why |
|---|---|---|
| "Do not use a slider, use explicit clickable dots" for IPD iterations | dots up to 40, range input beyond | 5 of 7 `131_Na-benzene` frames are 201 entries. The previous frontend had already met this wall and shipped `MAX_DOTS = 40` with the same degradation; both branches call `selectIpdIteration`, so it stays one behaviour |
| `renderIpdFrame` as its own geometry renderer | thin wrapper over shared `drawGeometry` | the design rule wants two names with two responsibilities, not two copies of one body |
| per-frame Compute only | plus **Compute for all frames** | nothing is ever pre-computed, so a 14-frame scan would otherwise be 14 presses and 14 collection rewrites |
| — | Stage 0 parameterization | the single-viewer signatures could not serve a second viewer at all |

## Worth knowing

The iteration label reads **kcal/mol**, taken from the previous frontend where the same quantity
was named `energy_kcalmol`. The backend field is a bare `energy`. Renaming it to carry its unit
is a reasonable follow-up and is not done here.

## Viewer overlay layout

The overlay controls sit **beside** the main canvas rather than beneath it. `.viewer-panel` is
a two-column flex row: `.viewer-main` (canvas, trajectory dots, playback) on the left, and
`.overlays.overlays-side` on the right, divided by a `border-left`.

Three things about it are worth knowing:

* **No JavaScript.** 3Dmol's `GLViewer` registers a `ResizeObserver` **on its container**, plus
  a window resize listener and an IntersectionObserver, so a CSS-driven width change resizes the
  canvas and re-renders unaided. Verified rather than assumed: the canvas measures within 2px of
  its container at every viewport tested.
* **`main`'s `max-width` went 60rem → 78rem.** The point was to use the room that was empty to
  the right, not to shrink the canvas into it. Measured at a 1600px viewport the canvas is
  **942px**, slightly wider than the 928px it had as a full-width block, and it still gains a
  256px sidebar.
* **`.viewer-main { min-width: 0 }` is load-bearing.** A flex item defaults to
  `min-width: auto` and refuses to shrink below its content, which would push the sidebar out of
  the panel instead of sharing the row.

`.overlays-side` is a modifier on the existing `.overlays`; the IPD panel keeps the plain
stacked version, because it holds a single slider and a column there would be mostly empty.
Below 60rem a media query — the stylesheet's first — returns both to the stacked arrangement.
