declare module "frappe-gantt" {
  export interface FrappeTask {
    id: string;
    name: string;
    start: string;
    end: string;
    progress?: number;
    dependencies?: string;
    custom_class?: string;
  }

  export interface FrappeOptions {
    view_mode?: "Quarter Day" | "Half Day" | "Day" | "Week" | "Month" | "Year";
    date_format?: string;
    bar_height?: number;
    bar_corner_radius?: number;
    arrow_curve?: number;
    padding?: number;
    infinite_padding?: boolean;
    popup_on?: "hover" | "click";
    language?: string;
    header_height?: number;
    column_width?: number;
    step?: number;
    view_modes?: string[];
    custom_popup_html?: ((task: FrappeTask) => string) | null;
    on_click?: (task: FrappeTask) => void;
    on_date_change?: (task: FrappeTask, start: Date, end: Date) => void;
    on_progress_change?: (task: FrappeTask, progress: number) => void;
    on_view_change?: (mode: string) => void;
  }

  export default class Gantt {
    constructor(wrapper: HTMLElement | string, tasks: FrappeTask[], options?: FrappeOptions);
    change_view_mode(mode: string): void;
    refresh(tasks: FrappeTask[]): void;
  }
}

declare module "frappe-gantt-css";
