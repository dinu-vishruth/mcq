import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.route('https://fonts.googleapis.com/**', route => route.abort());
  await page.route('https://fonts.gstatic.com/**', route => route.abort());
  await page.goto('/', { waitUntil: 'domcontentloaded' });
  await page.getByRole('button', { name: 'Try Demo' }).click();
  await expect(page).toHaveURL(/dashboard/);
});

test('saved resource opens without reupload; Skip and Submit lead to durable private results', async ({ page, browser }) => {
  await page.goto('/journey');
  await page.getByRole('button', { name: 'Make Quiz', exact: true }).first().click();
  await expect(page).toHaveURL(/practice\?doc=\d+/);
  const resourceId = new URL(page.url()).searchParams.get('doc');
  await expect(page.getByRole('button', { name: 'Generate & Start' })).toBeEnabled();
  const generation = page.waitForRequest(req => req.url().includes('/api/practice/generate'));
  await page.getByRole('button', { name: 'Generate & Start' }).click();
  expect((await generation).postDataJSON().document_id).toBe(Number(resourceId));
  await expect(page).toHaveURL(/mcq_test/);
  const quizData = JSON.parse(await page.locator('#root').getAttribute('data-bootstrap') || '{}');
  expect(JSON.stringify(quizData)).not.toContain('answer_text');
  await page.getByRole('button', { name: /Mitochondria/ }).click();
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  await page.getByRole('button', { name: 'Skip', exact: true }).click();
  await expect(page.getByRole('status')).toHaveText('1 answered · 1 unanswered');
  await page.getByRole('button', { name: 'Submit Answers' }).click();
  await expect(page).toHaveURL(/results\/[a-z0-9]+/);
  await expect(page.getByText('1 / 2 correct')).toBeVisible();
  const explanations = page.getByText('Explanation', { exact: true }).locator('..');
  await expect(explanations).toHaveCount(2);
  for (const explanation of await explanations.all()) {
    await expect(explanation).toContainText('Mitochondria produce ATP through cellular respiration.');
  }
  const resultURL = page.url();
  await page.reload();
  await expect(page.getByText('1 / 2 correct')).toBeVisible();
  await page.goto('/mcq_test');
  await expect(page).toHaveURL(resultURL);
  const stranger = await browser.newContext();
  const privatePage = await stranger.newPage();
  await privatePage.goto('/');
  await privatePage.getByRole('button', { name: 'Try Demo' }).click();
  const denied = await privatePage.goto(resultURL);
  expect(denied?.status()).toBe(404);
  await stranger.close();
});

test('one-off upload reaches settings and failed generation preserves material for retry', async ({ page }) => {
  await page.goto('/upload');
  await page.locator('input[type=file]').setInputFiles({
    name: 'biology.txt', mimeType: 'text/plain',
    buffer: Buffer.from('Mitochondria produce ATP through cellular respiration. '.repeat(20)),
  });
  await page.getByRole('button', { name: 'Upload & Continue' }).click();
  await expect(page).toHaveURL(/knowledge\?upload=/);
  const settingsURL = page.url();
  await expect(page.getByRole('button', { name: 'Generate & Start' })).toBeEnabled();
  await page.route('**/api/practice/generate', route => route.fulfill({
    status: 502, contentType: 'application/json', body: JSON.stringify({ error: 'AI temporarily unavailable' }),
  }));
  await page.getByRole('button', { name: 'Generate & Start' }).click();
  await expect(page.getByText('AI temporarily unavailable')).toBeVisible();
  await expect(page).toHaveURL(settingsURL);
  await page.unroute('**/api/practice/generate');
  await page.getByRole('button', { name: 'Generate & Start' }).click();
  await expect(page).toHaveURL(/mcq_test/);
});
