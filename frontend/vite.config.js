import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy /api -> the FastAPI backend so the browser never deals with CORS and the
// frontend code stays environment-agnostic (it just calls "/api/ask").
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
