import { defineConfig } from "vite";

export default defineConfig({
  clearScreen: false,
  build: {
    rollupOptions: {
      input: ["index.html", "vision-debug.html"],
    },
  },
  server: {
    host: "127.0.0.1",
    port: 1420,
    strictPort: true,
  },
});

