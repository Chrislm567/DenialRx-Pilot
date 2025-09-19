import { test, expect } from '@playwright/test';

test.describe('Phase Runner dashboard', () => {
  test('navigates overview, claims and scorecards', async ({ page }) => {
    await page.goto('/');

    await expect(page.getByRole('heading', { name: 'Pulse' })).toBeVisible();

    await page.getByRole('link', { name: 'Claims Explorer' }).click();
    await expect(page.getByRole('heading', { name: 'Claims Explorer' })).toBeVisible();
    await expect(page.getByRole('table')).toBeVisible();

    await page.getByRole('link', { name: 'Payer Scorecards' }).click();
    await expect(page.getByRole('heading', { name: 'Payer Scorecards' })).toBeVisible();
  });

  test('filters claims table', async ({ page }) => {
    await page.goto('/claims-explorer');

    const table = page.getByRole('table');
    await expect(table).toBeVisible();

    await page.locator('select[name="status"]').selectOption('Denied');
    await expect(page.locator('select[name="status"]')).toHaveValue('Denied');
    await expect(page.getByTestId('count')).toHaveText('2');

    const rows = table.getByRole('row');
    await expect(rows).toHaveCount(3); // header + 2 denied rows

    const deniedCells = table.getByRole('cell', { name: 'Denied' });
    await expect(deniedCells).toHaveCount(2);
  });
});
