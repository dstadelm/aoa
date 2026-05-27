/**
 * Tab switching logic for the top section.
 */

export type TabName = "resources" | "milestones" | "activities";

export function initTabs(onSwitch: (tab: TabName) => void): void {
  const buttons = document.querySelectorAll<HTMLButtonElement>(".tab-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      onSwitch(btn.dataset.tab as TabName);
    });
  });
}
