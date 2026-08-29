/**
 * Renders a Gantt chart using Frappe Gantt.
 */

import Gantt from "frappe-gantt";
import "frappe-gantt-css";

export interface GanttTask {
  id: string;
  label: string;
  start: string;
  end: string;
  duration: number;
  critical: boolean;
  color?: "critical" | "red" | "orange" | "green" | "";
  predecessors: string[];
}

export interface GanttMilestone {
  id: string;
  label: string;
  date: string;
}

export interface GanttData {
  start: string;
  tasks: GanttTask[];
  milestones: GanttMilestone[];
}

export type GanttViewMode = "Day" | "Week" | "Month" | "Year";

export function renderGantt(container: HTMLElement, data: GanttData, viewMode: GanttViewMode = "Day"): void {
  container.innerHTML = "";
  if (!data.tasks.length && !data.milestones.length) {
    container.innerHTML = '<p class="graph-empty">No activities to display.</p>';
    return;
  }

  const wrapper = document.createElement("div");
  wrapper.className = "frappe-gantt-wrapper";
  container.appendChild(wrapper);

  // Map tasks to Frappe format
  const frappeTasks = data.tasks.map((t) => {
    const cls = t.critical
      ? "bar-critical"
      : t.color && t.color !== "critical"
        ? `bar-color-${t.color}`
        : "";
    return {
      id: t.id,
      name: `[${t.id}] ${t.label}`,
      start: t.start,
      end: t.end,
      progress: 0,
      dependencies: t.predecessors.join(","),
      custom_class: cls,
    };
  });

  // Milestones: zero-duration bars, styled differently
  const frappeMilestones = data.milestones.map((m) => ({
    id: m.id,
    name: `[${m.id}] ${m.label}`,
    start: m.date,
    end: m.date,
    progress: 0,
    dependencies: "",
    custom_class: "bar-milestone",
  }));

  const allTasks = [...frappeTasks, ...frappeMilestones];

  new Gantt(wrapper, allTasks, {
    view_mode: viewMode,
    date_format: "YYYY-MM-DD",
    bar_height: 20,
    bar_corner_radius: 3,
    arrow_curve: 8,
    padding: 18,
    infinite_padding: false,
    popup_on: "hover",
  });
}
