import { randomBytes } from "node:crypto";
import { fileURLToPath } from "node:url";
import { defineConfig, devices } from "@playwright/test";

const root = fileURLToPath(new URL("..", import.meta.url));
const database = process.env.DATABASE_URL;
if (!database) throw new Error("Set DATABASE_URL to the disposable riskforge_e2e database.");
const parsed = new URL(database);
if (parsed.pathname !== "/riskforge_e2e" || !["localhost", "127.0.0.1", "[::1]"].includes(parsed.hostname)) {
  throw new Error("E2E requires riskforge_e2e on a loopback host.");
}
process.env.RISKFORGE_REVIEWER_API_KEY ||= randomBytes(32).toString("hex");
process.env.RISKFORGE_REVIEWER_ID ||= "e2e-automation";
process.env.PYTHONPATH = root;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  forbidOnly: !!process.env.CI,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:5174",
    channel: "chromium",
    trace: "off",
    screenshot: "off",
    video: "off",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: "python -m uvicorn app.main:app --host 127.0.0.1 --port 8010",
      cwd: root,
      url: "http://127.0.0.1:8010/v1/audit/decisions",
      reuseExistingServer: false,
      timeout: 60_000,
    },
    {
      command: "npm run dev -- --port 5174 --strictPort",
      url: "http://127.0.0.1:5174",
      env: { RISKFORGE_API_TARGET: "http://127.0.0.1:8010" },
      reuseExistingServer: false,
    },
  ],
});
