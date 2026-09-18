import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    open: false,
    // Proxy frontend API calls to FastAPI so the browser only uses port 3000.
    proxy: {
      "/chat": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/speech-to-text": "http://localhost:8000",
      "/text-to-speech": "http://localhost:8000",
      "/translate": "http://localhost:8000",
      "/speech-to-speech": "http://localhost:8000",
    },
  },
});
