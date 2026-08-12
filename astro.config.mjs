// @ts-check
import { defineConfig } from "astro/config";

export default defineConfig({
  vite: {
    build: {
      chunkSizeWarningLimit: 1000,
    },
  },
});

