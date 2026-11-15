import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// A separate file rather than a `test` block in vite.config.ts: vitest prefers
// this one when both exist, so the build config the user already has stays
// untouched and no `test` key has to be added to vite's UserConfig type.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.{ts,tsx}"],
    setupFiles: ["./src/test/setup.ts"],
    restoreMocks: true,
  },
});
