import { loadProject, saveProject, computeNetwork, computeNetworkDot } from "./api";
import { initTabs, type TabName } from "./tabs";
import { renderActivitiesTable, renderResourcesTable, renderMilestonesTable, renderProjectForm } from "./tables";
import { renderGraph, type LayoutOptions } from "./graph";
import type { ProjectData, GraphData } from "./types";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let currentFile = "";
let project: ProjectData = {
  start: new Date().toISOString().slice(0, 10),
  activities: [],
  resources: [],
  milestones: [],
};
let currentTab: TabName = "activities";
let graphData: GraphData | null = null;
let dotSvg: string | null = null;

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------

const fileInput = document.getElementById("file-path") as HTMLInputElement;
const btnLoad = document.getElementById("btn-load")!;
const btnSave = document.getElementById("btn-save")!;
const btnCompute = document.getElementById("btn-compute")!;
const tableContainer = document.getElementById("table-container")!;
const graphContainer = document.getElementById("graph-container")!;
const statusMsg = document.getElementById("status-msg")!;

// ---------------------------------------------------------------------------
// Status message helper
// ---------------------------------------------------------------------------

function showStatus(msg: string, isError = false): void {
  statusMsg.textContent = msg;
  statusMsg.className = isError ? "status-msg error" : "status-msg success";
  setTimeout(() => {
    statusMsg.textContent = "";
    statusMsg.className = "status-msg";
  }, 4000);
}

// ---------------------------------------------------------------------------
// Render the active table
// ---------------------------------------------------------------------------

function renderCurrentTable(): void {
  switch (currentTab) {
    case "activities":
      renderActivitiesTable(tableContainer, project.activities, () => {});
      break;
    case "resources":
      renderResourcesTable(tableContainer, project.resources, () => {});
      break;
    case "milestones":
      renderMilestonesTable(tableContainer, project.milestones, () => {});
      break;
    case "project":
      renderProjectForm(tableContainer, project, () => {});
      break;
  }
}

// ---------------------------------------------------------------------------
// Event handlers
// ---------------------------------------------------------------------------

btnLoad.addEventListener("click", async () => {
  const filePath = fileInput.value.trim();
  if (!filePath) {
    showStatus("Enter a file path first.", true);
    return;
  }
  try {
    const result = await loadProject(filePath);
    project = result.project;
    currentFile = result.file;
    fileInput.value = currentFile;
    renderCurrentTable();
    graphData = null;
    graphContainer.innerHTML = '<p class="graph-empty">Click "Compute Network" to generate the diagram.</p>';
    showStatus("Project loaded successfully.");
  } catch (e: any) {
    showStatus(`Load failed: ${e.message}`, true);
  }
});

btnSave.addEventListener("click", async () => {
  const filePath = fileInput.value.trim();
  if (!filePath) {
    showStatus("Enter a file path first.", true);
    return;
  }
  try {
    await saveProject(filePath, project);
    currentFile = filePath;
    showStatus("Project saved successfully.");
  } catch (e: any) {
    showStatus(`Save failed: ${e.message}`, true);
  }
});

btnCompute.addEventListener("click", async () => {
  if (!project.activities.length) {
    showStatus("No activities to compute.", true);
    return;
  }
  try {
    const renderer = optRenderer.value;
    if (renderer === "graphviz") {
      dotSvg = await computeNetworkDot(project.activities, themeSelect.value, optRankdir.value);
      graphData = null;
      renderDotSvg();
    } else {
      graphData = await computeNetwork(project.activities);
      dotSvg = null;
      renderGraph(graphContainer, graphData, getLayoutOptions());
    }
    showStatus("Network computed successfully.");
  } catch (e: any) {
    showStatus(`Compute failed: ${e.message}`, true);
  }
});

// ---------------------------------------------------------------------------
// Layout options
// ---------------------------------------------------------------------------

const optRenderer = document.getElementById("opt-renderer") as HTMLSelectElement;
const optRanker = document.getElementById("opt-ranker") as HTMLSelectElement;
const optRankdir = document.getElementById("opt-rankdir") as HTMLSelectElement;
const optAlign = document.getElementById("opt-align") as HTMLSelectElement;

function getLayoutOptions(): LayoutOptions {
  return {
    ranker: optRanker.value,
    rankdir: optRankdir.value,
    align: optAlign.value,
  };
}

function renderDotSvg(): void {
  if (dotSvg) {
    graphContainer.innerHTML = `<div class="dot-svg-container">${dotSvg}</div>`;
  }
}

function redrawGraph(): void {
  if (optRenderer.value === "graphviz") {
    // Graphviz layout is server-side; need to recompute
    if (dotSvg) renderDotSvg();
  } else if (graphData) {
    renderGraph(graphContainer, graphData, getLayoutOptions());
  }
}

// When renderer changes, redraw with available data or prompt recompute
optRenderer.addEventListener("change", async () => {
  const renderer = optRenderer.value;
  // Show/hide d3-dagre-only options
  const dagreOnly = document.querySelectorAll<HTMLElement>(".dagre-only");
  dagreOnly.forEach((el) => (el.style.display = renderer === "d3-dagre" ? "" : "none"));

  if (renderer === "graphviz") {
    if (dotSvg) {
      renderDotSvg();
    } else if (project.activities.length) {
      try {
        dotSvg = await computeNetworkDot(project.activities, themeSelect.value, optRankdir.value);
        renderDotSvg();
      } catch (e: any) {
        showStatus(`Compute failed: ${e.message}`, true);
      }
    }
  } else {
    if (graphData) {
      renderGraph(graphContainer, graphData, getLayoutOptions());
    } else if (project.activities.length) {
      try {
        graphData = await computeNetwork(project.activities);
        renderGraph(graphContainer, graphData, getLayoutOptions());
      } catch (e: any) {
        showStatus(`Compute failed: ${e.message}`, true);
      }
    }
  }
});

[optRanker, optRankdir, optAlign].forEach((sel) => {
  sel.addEventListener("change", async () => {
    if (optRenderer.value === "graphviz") {
      if (sel === optRankdir && project.activities.length) {
        try {
          dotSvg = await computeNetworkDot(project.activities, themeSelect.value, optRankdir.value);
          renderDotSvg();
        } catch (e: any) {
          showStatus(`Compute failed: ${e.message}`, true);
        }
      }
    } else {
      redrawGraph();
    }
  });
});

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------

initTabs((tab) => {
  currentTab = tab;
  renderCurrentTable();
});

// ---------------------------------------------------------------------------
// Collapse/Expand top section
// ---------------------------------------------------------------------------

const btnCollapse = document.getElementById("btn-collapse")!;
const topSection = document.getElementById("top-section")!;

btnCollapse.addEventListener("click", () => {
  topSection.classList.toggle("collapsed");
  btnCollapse.innerHTML = topSection.classList.contains("collapsed") ? "&#x25BC;" : "&#x25B2;";
});

// ---------------------------------------------------------------------------
// Initial render
// ---------------------------------------------------------------------------

// Theme selector
const themeSelect = document.getElementById("theme-select") as HTMLSelectElement;
const savedTheme = localStorage.getItem("aoa-theme") || "material-light";
document.documentElement.setAttribute("data-theme", savedTheme);
themeSelect.value = savedTheme;
themeSelect.addEventListener("change", async () => {
  const theme = themeSelect.value;
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("aoa-theme", theme);
  if (optRenderer.value === "graphviz" && project.activities.length) {
    try {
      dotSvg = await computeNetworkDot(project.activities, theme, optRankdir.value);
      renderDotSvg();
    } catch (e: any) {
      showStatus(`Compute failed: ${e.message}`, true);
    }
  } else {
    redrawGraph();
  }
});

renderCurrentTable();
graphContainer.innerHTML = '<p class="graph-empty">Load a YAML project and click "Compute Network" to see the diagram.</p>';
