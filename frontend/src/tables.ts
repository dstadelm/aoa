/**
 * Editable HTML tables for resources, milestones, and activities.
 */

import type { ActivityData, ResourceData, MilestoneData, ProjectData } from "./types";
import { STATES } from "./types";

// ---------------------------------------------------------------------------
// Generic helpers
// ---------------------------------------------------------------------------

function el<K extends keyof HTMLElementTagNameMap>(tag: K, attrs?: Record<string, string>, text?: string): HTMLElementTagNameMap[K] {
  const elem = document.createElement(tag);
  if (attrs) Object.entries(attrs).forEach(([k, v]) => elem.setAttribute(k, v));
  if (text !== undefined) elem.textContent = text;
  return elem;
}

function editableCell(value: string, onChange: (val: string) => void): HTMLTableCellElement {
  const td = el("td");
  td.contentEditable = "true";
  td.textContent = value;
  td.addEventListener("blur", () => onChange(td.textContent?.trim() ?? ""));
  return td;
}

function selectCell(value: string, options: string[], onChange: (val: string) => void): HTMLTableCellElement {
  const td = el("td");
  const select = el("select");
  options.forEach((opt) => {
    const option = el("option", { value: opt }, opt);
    if (opt === value) option.selected = true;
    select.appendChild(option);
  });
  select.addEventListener("change", () => onChange(select.value));
  td.appendChild(select);
  return td;
}

function deleteCell(onClick: () => void): HTMLTableCellElement {
  const td = el("td");
  const btn = el("button", { class: "delete-btn", title: "Delete row" }, "\u00D7");
  btn.addEventListener("click", onClick);
  td.appendChild(btn);
  return td;
}

// ---------------------------------------------------------------------------
// Project form (start date, etc.)
// ---------------------------------------------------------------------------

export function renderProjectForm(
  container: HTMLElement,
  project: ProjectData,
  onUpdate: () => void,
): void {
  container.innerHTML = "";
  const form = el("div", { class: "project-form" });

  const label = el("label", {}, "Start Date: ");
  const input = el("input", { type: "date", value: project.start || "" });
  input.addEventListener("change", () => {
    project.start = input.value;
    onUpdate();
  });
  label.appendChild(input);
  form.appendChild(label);

  container.appendChild(form);
}

// ---------------------------------------------------------------------------
// Activities table
// ---------------------------------------------------------------------------

export function renderActivitiesTable(
  container: HTMLElement,
  activities: ActivityData[],
  onUpdate: () => void,
): void {
  container.innerHTML = "";
  const table = el("table", { class: "data-table" });
  const thead = el("thead");
  const headerRow = el("tr");
  ["ID", "Activity", "Predecessors", "Planned Effort", "Owner", "Resource", "State", ""].forEach((h) =>
    headerRow.appendChild(el("th", {}, h))
  );
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = el("tbody");
  activities.forEach((a, idx) => {
    const tr = el("tr");
    tr.appendChild(editableCell(String(a.id), (v) => { a.id = parseInt(v) || 0; onUpdate(); }));
    tr.appendChild(editableCell(a.activity, (v) => { a.activity = v; onUpdate(); }));
    tr.appendChild(editableCell(a.predecessors.join(", "), (v) => {
      a.predecessors = v ? v.split(",").map((s) => parseInt(s.trim())).filter((n) => !isNaN(n)) : [];
      onUpdate();
    }));
    tr.appendChild(editableCell(String(a.planned_effort), (v) => { a.planned_effort = parseFloat(v) || 0; onUpdate(); }));
    tr.appendChild(editableCell(a.owner, (v) => { a.owner = v; onUpdate(); }));
    tr.appendChild(editableCell(a.resource, (v) => { a.resource = v; onUpdate(); }));
    tr.appendChild(selectCell(a.state, STATES, (v) => { a.state = v; onUpdate(); }));
    tr.appendChild(deleteCell(() => { activities.splice(idx, 1); renderActivitiesTable(container, activities, onUpdate); onUpdate(); }));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  container.appendChild(table);

  const addBtn = el("button", { class: "add-btn" }, "+ Add Activity");
  addBtn.addEventListener("click", () => {
    const maxId = activities.reduce((m, a) => Math.max(m, a.id), 0);
    activities.push({ id: maxId + 1, activity: "", predecessors: [], planned_effort: 0, owner: "", resource: "", state: "OPEN" });
    renderActivitiesTable(container, activities, onUpdate);
    onUpdate();
  });
  container.appendChild(addBtn);
}

// ---------------------------------------------------------------------------
// Resources table
// ---------------------------------------------------------------------------

export function renderResourcesTable(
  container: HTMLElement,
  resources: ResourceData[],
  onUpdate: () => void,
): void {
  container.innerHTML = "";
  const table = el("table", { class: "data-table" });
  const thead = el("thead");
  const headerRow = el("tr");
  ["ID", "Name", "Workload", "Weekdays", "Holidays", ""].forEach((h) =>
    headerRow.appendChild(el("th", {}, h))
  );
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = el("tbody");
  resources.forEach((r, idx) => {
    const tr = el("tr");
    tr.appendChild(editableCell(r.id, (v) => { r.id = v; onUpdate(); }));
    tr.appendChild(editableCell(r.name, (v) => { r.name = v; onUpdate(); }));
    tr.appendChild(editableCell(r.workload, (v) => { r.workload = v; onUpdate(); }));
    tr.appendChild(editableCell(r.weekdays, (v) => { r.weekdays = v; onUpdate(); }));
    tr.appendChild(editableCell(r.holidays.join(", "), (v) => {
      r.holidays = v ? v.split(",").map((s) => s.trim()) : [];
      onUpdate();
    }));
    tr.appendChild(deleteCell(() => { resources.splice(idx, 1); renderResourcesTable(container, resources, onUpdate); onUpdate(); }));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  container.appendChild(table);

  const addBtn = el("button", { class: "add-btn" }, "+ Add Resource");
  addBtn.addEventListener("click", () => {
    resources.push({ id: "", name: "", workload: "", weekdays: "1111100", holidays: [] });
    renderResourcesTable(container, resources, onUpdate);
    onUpdate();
  });
  container.appendChild(addBtn);
}

// ---------------------------------------------------------------------------
// Milestones table
// ---------------------------------------------------------------------------

export function renderMilestonesTable(
  container: HTMLElement,
  milestones: MilestoneData[],
  onUpdate: () => void,
): void {
  container.innerHTML = "";
  const table = el("table", { class: "data-table" });
  const thead = el("thead");
  const headerRow = el("tr");
  ["ID", "Description", "Owner", "Due Date", "State", ""].forEach((h) =>
    headerRow.appendChild(el("th", {}, h))
  );
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = el("tbody");
  milestones.forEach((m, idx) => {
    const tr = el("tr");
    tr.appendChild(editableCell(m.id, (v) => { m.id = v; onUpdate(); }));
    tr.appendChild(editableCell(m.description, (v) => { m.description = v; onUpdate(); }));
    tr.appendChild(editableCell(m.owner, (v) => { m.owner = v; onUpdate(); }));
    tr.appendChild(editableCell(m.due_date, (v) => { m.due_date = v; onUpdate(); }));
    tr.appendChild(selectCell(m.state, STATES, (v) => { m.state = v; onUpdate(); }));
    tr.appendChild(deleteCell(() => { milestones.splice(idx, 1); renderMilestonesTable(container, milestones, onUpdate); onUpdate(); }));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  container.appendChild(table);

  const addBtn = el("button", { class: "add-btn" }, "+ Add Milestone");
  addBtn.addEventListener("click", () => {
    milestones.push({ id: "", description: "", owner: "", due_date: new Date().toISOString().slice(0, 10), state: "OPEN" });
    renderMilestonesTable(container, milestones, onUpdate);
    onUpdate();
  });
  container.appendChild(addBtn);
}
