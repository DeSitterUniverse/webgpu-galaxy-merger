import { defineConfig } from "vitest/config";

// The longer reference suite remains separate from normal development tests.
export default defineConfig({
  test: {
    include: ["tests/galaxy/**/*.galaxy.test.ts"],
    testTimeout: 120_000,
    hookTimeout: 120_000,
  },
});

