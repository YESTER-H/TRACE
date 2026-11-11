const { chromium } = require('playwright');
const [url, out] = process.argv.slice(2);
(async () => {
  const browser = await chromium.launch({ channel: 'chromium', args: ['--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=gl-egl'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForTimeout(8000);
  await page.getByText('Interact', { exact: true }).click();
  await page.waitForTimeout(800);
  await page.screenshot({ path: `${out}/interact.png`, timeout: 120000 });
  console.log(await page.locator('body').innerText());
  await browser.close();
})();
