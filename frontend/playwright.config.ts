import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  use: {
    baseURL: 'http://127.0.0.1:8765',
    channel: process.env.E2E_BROWSER_CHANNEL || undefined,
    trace: 'retain-on-failure',
  },
  webServer: {
    command: 'python -m tests.e2e_server',
    cwd: '..',
    url: 'http://127.0.0.1:8765/',
    reuseExistingServer: false,
    timeout: 60000,
  },
});
