import { loadProject, saveProject, computeNetwork, computeNetworkDot, computeNetworkGantt } from "./api";
import { initTabs, type TabName } from "./tabs";
import { renderActivitiesTable, renderResourcesTable, renderMilestonesTable, renderProjectForm } from "./tables";
import { renderGraph, type LayoutOptions } from "./graph";
import { renderGantt, type GanttData } from "./gantt";
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
let ganttData: GanttData | null = null;

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
const btnExportSvg = document.getElementById("btn-export-svg")!;
const btnExportPng = document.getElementById("btn-export-png")!;

// Whitelist of CSS properties to inline so exported SVGs render standalone.
const SVG_STYLE_PROPS = [
  "fill",
  "fill-opacity",
  "stroke",
  "stroke-width",
  "stroke-opacity",
  "stroke-dasharray",
  "stroke-linecap",
  "stroke-linejoin",
  "opacity",
  "font-family",
  "font-size",
  "font-weight",
  "font-style",
  "text-anchor",
  "dominant-baseline",
  "visibility",
  "display",
];

function inlineStyles(source: SVGSVGElement, target: SVGSVGElement): void {
  const srcNodes = source.querySelectorAll<SVGElement>("*");
  const dstNodes = target.querySelectorAll<SVGElement>("*");
  // Also handle the root svg itself
  const inlineFor = (src: Element, dst: Element) => {
    const cs = window.getComputedStyle(src);
    let styleStr = "";
    for (const prop of SVG_STYLE_PROPS) {
      const val = cs.getPropertyValue(prop);
      if (val && val !== "none" && val !== "normal") {
        styleStr += `${prop}:${val};`;
      }
    }
    if (styleStr) (dst as SVGElement).setAttribute("style", styleStr);
  };
  inlineFor(source, target);
  srcNodes.forEach((src, i) => inlineFor(src, dstNodes[i]));
}

function buildStandaloneSvg(): { clone: SVGSVGElement; width: number; height: number } | null {
  const svg = graphContainer.querySelector("svg");
  if (!svg) return null;
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
  inlineStyles(svg, clone);

  const bbox = svg.getBoundingClientRect();
  if (!clone.getAttribute("viewBox")) {
    clone.setAttribute("viewBox", `0 0 ${bbox.width} ${bbox.height}`);
  }
  const width = Math.round(bbox.width);
  const height = Math.round(bbox.height);
  clone.setAttribute("width", String(width));
  clone.setAttribute("height", String(height));

  // Add a background rect matching the current theme so PNGs aren't transparent.
  const bg = window.getComputedStyle(graphContainer).backgroundColor;
  if (bg && bg !== "rgba(0, 0, 0, 0)" && bg !== "transparent") {
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("width", "100%");
    rect.setAttribute("height", "100%");
    rect.setAttribute("fill", bg);
    clone.insertBefore(rect, clone.firstChild);
  }
  return { clone, width, height };
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

btnExportSvg.addEventListener("click", () => {
  const built = buildStandaloneSvg();
  if (!built) {
    showStatus("Nothing to export — compute the diagram first.", true);
    return;
  }
  const source = new XMLSerializer().serializeToString(built.clone);
  const blob = new Blob(['<?xml version="1.0" encoding="UTF-8"?>\n', source], {
    type: "image/svg+xml;charset=utf-8",
  });
  triggerDownload(blob, `aoa-diagram-${optRenderer.value}-${Date.now()}.svg`);
  showStatus("SVG exported.");
});

btnExportPng.addEventListener("click", () => {
  const built = buildStandaloneSvg();
  if (!built) {
    showStatus("Nothing to export — compute the diagram first.", true);
    return;
  }
  const source = new XMLSerializer().serializeToString(built.clone);
  const svgBlob = new Blob([source], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(svgBlob);
  const img = new Image();
  const scale = 2; // hi-DPI
  img.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = built.width * scale;
    canvas.height = built.height * scale;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0);
    URL.revokeObjectURL(url);
    canvas.toBlob((blob) => {
      if (!blob) {
        showStatus("PNG conversion failed.", true);
        return;
      }
      triggerDownload(blob, `aoa-diagram-${optRenderer.value}-${Date.now()}.png`);
      showStatus("PNG exported.");
    }, "image/png");
  };
  img.onerror = () => {
    URL.revokeObjectURL(url);
    showStatus("PNG export failed to load SVG.", true);
  };
  img.src = url;
});

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
      ganttData = null;
      renderDotSvg();
    } else if (renderer === "gantt") {
      ganttData = await computeNetworkGantt(project.activities, project.milestones, project.start);
      graphData = null;
      dotSvg = null;
      renderGantt(graphContainer, ganttData);
    } else {
      graphData = await computeNetwork(project.activities);
      dotSvg = null;
      ganttData = null;
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
  // Show/hide renderer-specific options
  document.querySelectorAll<HTMLElement>(".renderer-opt").forEach((el) => {
    const supported = (el.dataset.renderer ?? "").split(/\s+/);
    el.style.display = supported.includes(renderer) ? "" : "none";
  });

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
  } else if (renderer === "gantt") {
    if (ganttData) {
      renderGantt(graphContainer, ganttData);
    } else if (project.activities.length) {
      try {
        ganttData = await computeNetworkGantt(project.activities, project.milestones, project.start);
        renderGantt(graphContainer, ganttData);
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
// Initialize renderer-specific option visibility
optRenderer.dispatchEvent(new Event("change"));
