import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";

export default defineConfig({
  resolve: {
    alias: {
      "frappe-gantt-css": fileURLToPath(
        new URL("./node_modules/frappe-gantt/dist/frappe-gantt.css", import.meta.url),
      ),
    },
  },
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: "http://localhost:5000",
        changeOrigin: true,
      },
    },
  },
});
