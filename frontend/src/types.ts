// ---- Project Data Types ----

export interface ActivityData {
  id: number;
  activity: string;
  predecessors: number[];
  planned_effort: number;
  owner: string;
  resource: string;
  state: string;
}

export interface ResourceData {
  id: string;
  name: string;
  workload: string;
  weekdays: string;
  holidays: string[];
}

export interface MilestoneData {
  id: string;
  description: string;
  owner: string;
  due_date: string;
  state: string;
}

export interface ProjectData {
  start: string;
  activities: ActivityData[];
  resources: ResourceData[];
  milestones: MilestoneData[];
}

// ---- Network Graph Types ----

export interface GraphNode {
  id: number;
  maxDepth: number;
  earliestStart: number;
  latestStart: number;
}

export interface GraphEdge {
  source: number;
  target: number;
  activityId: number;
  label: string;
  isDummy: boolean;
  critical: boolean;
  color: string;
  planned_effort: number;
  earliestStart: number;
  earliestFinish: number;
  latestStart: number;
  latestFinish: number;
  totalFloat: number;
  freeFloat: number;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export const STATES = ["OPEN", "IN_PROGRESS", "DONE", "ON_HOLD", "CANCELLED"];
