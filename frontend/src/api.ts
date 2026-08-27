import type { ProjectData, GraphData, ActivityData } from "./types";

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

export async function computeNetwork(activities: ActivityData[]): Promise<GraphData> {
  const res = await fetch(`${API_BASE}/network`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ activities }),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

export async function computeNetworkDot(activities: ActivityData[], theme?: string, rankdir?: string): Promise<string> {
  const res = await fetch(`${API_BASE}/network/dot`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ activities, theme, rankdir }),
  });
  if (!res.ok) {
    const data = await res.json();
    throw new Error(data.error || "Failed to render DOT");
  }
  return await res.text();
}
