import { defineConfig } from "@playwright/test";
import os from "node:os";
import path from "node:path";
export default defineConfig({
  testDir: "./e2e",
  timeout: 60000,
  expect: { timeout: 10000 },
  workers: 1,
  fullyParallel: false,
  reporter: "list",
  outputDir:
    process.env.LITAGENT_QA_OUTPUT ||
    path.join(os.tmpdir(), "litagent-ui-qa", "playwright"),
  use: {
    channel: process.env.PLAYWRIGHT_CHANNEL || "chromium",
    headless: true,
    viewport: { width: 1536, height: 1024 },
    screenshot: "only-on-failure",
  },
});
