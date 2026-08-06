// Tab bar: which of the two full-page sections is showing.
//
// Deliberately imports nothing. Anything that needs to react to a tab change listens for
// the `app:tabchange` event on `document` instead -- which is how main.js gets to decide
// when its 3Dmol viewer resizes, without this module knowing the viewer exists.

const TABS = [
  { name: "visualize", tab: "tabVisualize", page: "pageVisualize" },
  { name: "trajectory", tab: "tabTrajectory", page: "pageTrajectory" },
  { name: "compute", tab: "tabCompute", page: "pageCompute" },
  { name: "plot", tab: "tabPlot", page: "pagePlot" },
];

const entries = TABS.map((entry) => ({
  ...entry,
  tabEl: document.getElementById(entry.tab),
  pageEl: document.getElementById(entry.page),
}));

let current = "visualize";

export function currentTab() {
  return current;
}

// `focusTab` moves focus onto the newly selected tab. Required whenever the app switches
// tabs on the user's behalf: the control they just activated is about to be hidden, and
// leaving focus on a removed element strands keyboard and screen-reader users.
export function showTab(name, { focusTab = false } = {}) {
  const target = entries.find((entry) => entry.name === name);
  if (!target) {
    return;
  }

  current = name;
  for (const entry of entries) {
    const selected = entry.name === name;
    entry.pageEl.hidden = !selected;
    entry.tabEl.setAttribute("aria-selected", String(selected));
    // Roving tabindex: only the selected tab is in the sequential tab order, so Tab
    // moves past the whole bar rather than through every tab in it.
    entry.tabEl.tabIndex = selected ? 0 : -1;
  }

  document.dispatchEvent(
    new CustomEvent("app:tabchange", { detail: { name } }),
  );

  if (focusTab) {
    target.tabEl.focus();
  }
}

// Manual activation: arrows move focus only, Enter/Space commits. Arrowing straight into
// a page swap would fire a catalog reload and a viewer resize for a tab the user was only
// passing over.
function focusTabAt(index) {
  const wrapped = (index + entries.length) % entries.length;
  entries[wrapped].tabEl.focus();
}

function indexOfFocused() {
  return entries.findIndex((entry) => entry.tabEl === document.activeElement);
}

for (const entry of entries) {
  entry.tabEl.addEventListener("click", () => showTab(entry.name));

  entry.tabEl.addEventListener("keydown", (event) => {
    const index = indexOfFocused();
    if (index === -1) {
      return;
    }

    switch (event.key) {
      case "ArrowRight":
      case "ArrowDown":
        event.preventDefault();
        focusTabAt(index + 1);
        break;
      case "ArrowLeft":
      case "ArrowUp":
        event.preventDefault();
        focusTabAt(index - 1);
        break;
      case "Home":
        event.preventDefault();
        focusTabAt(0);
        break;
      case "End":
        event.preventDefault();
        focusTabAt(entries.length - 1);
        break;
      case " ":
      case "Spacebar":
        // Space would scroll the page; Enter needs no help, since a <button> already
        // activates on it and fires the click handler above.
        event.preventDefault();
        showTab(entry.name);
        break;
      default:
        break;
    }
  });
}

// The app always starts on Visualize, and never restores a remembered tab. main.js builds
// the 3Dmol viewer at module scope, and a WebGL canvas created inside a `display: none`
// container measures 0x0 -- so booting on Compute would produce an invisible viewer.
showTab("visualize");
