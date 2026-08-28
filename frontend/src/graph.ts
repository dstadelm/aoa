/**
 * Renders an AoA network diagram using d3 + dagre.
 */

import * as d3 from "d3";
import dagre from "@dagrejs/dagre";
import type { GraphData, GraphEdge } from "./types";

const NODE_RADIUS = 20;
const NODE_SIZE = NODE_RADIUS * 2;
const ARROW_ID = "arrowhead";
const LABEL_PADDING = 8;
const LINE_HEIGHT_EM = 1.15;

export interface LayoutOptions {
  ranker: string;
  rankdir: string;
  align: string;
}

function getThemeColors() {
  const style = getComputedStyle(document.documentElement);
  return {
    critical: style.getPropertyValue("--graph-critical").trim() || "#d32f2f",
    red: style.getPropertyValue("--graph-red").trim() || "#e53935",
    orange: style.getPropertyValue("--graph-orange").trim() || "#fb8c00",
    green: style.getPropertyValue("--graph-green").trim() || "#43a047",
    edge: style.getPropertyValue("--graph-edge").trim() || "#757575",
    arrow: style.getPropertyValue("--graph-arrow").trim() || "#555",
    nodeStroke: style.getPropertyValue("--node-stroke").trim() || "#333",
    nodeFill: style.getPropertyValue("--surface").trim() || "#fff",
    text: style.getPropertyValue("--text").trim() || "#222",
    textMuted: style.getPropertyValue("--text-muted").trim() || "#666",
  };
}

function edgeColor(e: GraphEdge, colors: ReturnType<typeof getThemeColors>): string {
  if (e.critical) return colors.critical;
  switch (e.color) {
    case "red": return colors.red;
    case "orange": return colors.orange;
    case "green": return colors.green;
    default: return colors.edge;
  }
}

export function renderGraph(container: HTMLElement, data: GraphData, options?: LayoutOptions): void {
  container.innerHTML = "";

  if (!data.nodes.length) {
    container.innerHTML = '<p class="graph-empty">No network to display. Load a project and click "Compute Network".</p>';
    return;
  }

  const colors = getThemeColors();

  // Create the target SVG upfront so we can measure text against the actual CSS
  // (font-family / size inherited by `.edge-label`).
  const svg = d3
    .select(container)
    .append("svg")
    .attr("width", "100%")
    .attr("height", "100%")
    .attr("class", "network-svg");

  // Hidden measurement group. Removed after the pre-layout pass.
  const measureGroup = svg.append("g").attr("visibility", "hidden");
  const measureText = measureGroup.append("text").attr("class", "edge-label").node() as SVGTextElement;
  const fontSizePx = parseFloat(window.getComputedStyle(measureText).fontSize) || 10;
  const lineHeightPx = fontSizePx * LINE_HEIGHT_EM;

  interface LabelInfo {
    lines: string[];
    width: number;
    height: number;
  }
  const labelInfo = new Map<number, LabelInfo>();

  data.edges.forEach((e, i) => {
    if (e.isDummy || !e.label) return;
    const lines = e.label.split("\n");
    let maxW = 0;
    for (const ln of lines) {
      measureText.textContent = ln;
      const w = measureText.getBBox().width;
      if (w > maxW) maxW = w;
    }
    labelInfo.set(i, {
      lines,
      width: maxW + LABEL_PADDING,
      height: lines.length * lineHeightPx + LABEL_PADDING,
    });
  });
  measureGroup.remove();

  // Build dagre graph
  const g = new dagre.graphlib.Graph();
  const graphOpts: Record<string, any> = {
    rankdir: options?.rankdir || "TB",
    marginx: 30,
    marginy: 30,
    ranksep: 60,
    nodesep: 40,
    ranker: options?.ranker || "network-simplex",
  };
  if (options?.align) graphOpts.align = options.align;
  g.setGraph(graphOpts);
  g.setDefaultEdgeLabel(() => ({}));

  data.nodes.forEach((n) => {
    g.setNode(String(n.id), { label: String(n.id), width: NODE_SIZE, height: NODE_SIZE });
  });

  data.edges.forEach((e, i) => {
    const info = labelInfo.get(i);
    const edgeAttrs: Record<string, any> = { index: i };
    if (info) {
      edgeAttrs.width = info.width;
      edgeAttrs.height = info.height;
      edgeAttrs.labelpos = "c";
    }
    g.setEdge(String(e.source), String(e.target), edgeAttrs);
  });

  dagre.layout(g);

  const graphMeta = g.graph();
  const svgWidth = (graphMeta.width ?? 600) + 60;
  const svgHeight = (graphMeta.height ?? 400) + 60;

  // Arrow marker
  svg
    .append("defs")
    .append("marker")
    .attr("id", ARROW_ID)
    .attr("viewBox", "0 0 10 10")
    .attr("refX", 10)
    .attr("refY", 5)
    .attr("markerUnits", "userSpaceOnUse")
    .attr("markerWidth", 12)
    .attr("markerHeight", 12)
    .attr("orient", "auto-start-reverse")
    .append("path")
    .attr("d", "M 0 0 L 10 5 L 0 10 z")
    .attr("fill", colors.arrow);

  const root = svg.append("g");

  // Zoom & pan
  const zoom = d3.zoom<SVGSVGElement, unknown>()
    .scaleExtent([0.1, 5])
    .on("zoom", (event) => {
      root.attr("transform", event.transform);
    });

  svg.call(zoom);

  // Set initial transform to fit the graph
  const containerRect = container.getBoundingClientRect();
  const scale = Math.min(
    containerRect.width / svgWidth,
    containerRect.height / svgHeight,
    1.5
  );
  const tx = (containerRect.width - svgWidth * scale) / 2;
  const ty = 20;
  svg.call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));

  // Edges
  const edgeGroup = root.append("g").attr("class", "edges");

  g.edges().forEach((edgeKey) => {
    const edgeData = g.edge(edgeKey);
    const idx = edgeData.index as number;
    const e = data.edges[idx];
    const points: { x: number; y: number }[] = edgeData.points;

    const line = d3
      .line<{ x: number; y: number }>()
      .x((d) => d.x)
      .y((d) => d.y)
      .curve(d3.curveBasis);

    const path = edgeGroup
      .append("path")
      .attr("d", line(points)!)
      .attr("fill", "none")
      .attr("stroke", edgeColor(e, colors))
      .attr("stroke-width", e.critical ? 4 : 2)
      .attr("marker-end", `url(#${ARROW_ID})`);

    if (e.isDummy) {
      path.attr("stroke-dasharray", "6,4");
    }

    // Tooltip
    const title = e.isDummy
      ? `Dummy [${e.activityId}]`
      : `[${e.activityId}] ${e.label}\nES: ${e.earliestStart}  D: ${e.planned_effort}  EF: ${e.earliestFinish}\nLS: ${e.latestStart}  TF: ${e.totalFloat}  LF: ${e.latestFinish}\nFF: ${e.freeFloat}`;
    path.append("title").text(title);

    // Edge label (multi-line via explicit \n; dagre reserved space for it).
    // Draw above the arrow (like the previous behavior) rather than on top of it.
    const info = labelInfo.get(idx);
    if (info && !e.isDummy) {
      const cx = edgeData.x as number;
      const cy = edgeData.y as number;
      // Shift the whole text block up so its bottom sits just above the arrow center.
      const totalHeight = info.lines.length * lineHeightPx;
      const startY = cy - totalHeight / 2 - 4;
      const label = edgeGroup
        .append("text")
        .attr("x", cx)
        .attr("y", startY)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .attr("class", "edge-label");
      info.lines.forEach((ln, li) => {
        label
          .append("tspan")
          .attr("x", cx)
          .attr("dy", li === 0 ? "0" : `${LINE_HEIGHT_EM}em`)
          .text(ln);
      });
    }
  });

  // Nodes
  const nodeGroup = root.append("g").attr("class", "nodes");

  g.nodes().forEach((nodeId) => {
    const nodeData = g.node(nodeId);
    const graphNode = data.nodes.find((n) => n.id === parseInt(nodeId));

    const ng = nodeGroup
      .append("g")
      .attr("transform", `translate(${nodeData.x}, ${nodeData.y})`);

    ng.append("circle")
      .attr("r", NODE_RADIUS)
      .attr("class", "node-rect");

    if (graphNode) {
      ng.append("title").text(
        `Node ${nodeId}\nES: ${graphNode.earliestStart}\nLS: ${graphNode.latestStart}\nDepth: ${graphNode.maxDepth}`
      );
    }
  });
}
