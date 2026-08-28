import type { ProjectData, GraphData, ActivityData } from "./types";
import type { GanttData } from "./gantt";

const API_BASE = "/api";

export async function loadProject(filePath: string): Promise<{ file: string; project: ProjectData }> {
  const res = await fetch(`${API_BASE}/project?file=${encodeURIComponent(filePath)}`);
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

export async function saveProject(filePath: string, project: ProjectData): Promise<void> {
  const res = await fetch(`${API_BASE}/project`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file: filePath, project }),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error);
}

export async function computeNetwork(activities: ActivityData[], resources: any[] = []): Promise<GraphData> {
  const res = await fetch(`${API_BASE}/network`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ activities, resources }),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

export async function computeNetworkGantt(
  activities: ActivityData[],
  milestones: any[],
  start: string,
  resources: any[] = [],
): Promise<GanttData> {
  const res = await fetch(`${API_BASE}/network/gantt`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ activities, milestones, start, resources }),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

export async function computeNetworkDot(activities: ActivityData[], theme?: string, rankdir?: string, resources: any[] = []): Promise<string> {
  const res = await fetch(`${API_BASE}/network/dot`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ activities, theme, rankdir, resources }),
  });
  if (!res.ok) {
    const data = await res.json();
    throw new Error(data.error || "Failed to render DOT");
  }
  return await res.text();
}
