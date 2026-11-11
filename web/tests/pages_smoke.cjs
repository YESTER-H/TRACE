// Load the landing page under its GitHub Pages prefix and check the embedded demo runs.
const { chromium } = require('playwright');
const [url, out] = process.argv.slice(2);
(async () => {
  const browser = await chromium.launch({ channel: 'chromium', args: ['--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=gl-egl'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const bad = [];
  page.on('response', (r) => { if (r.status() >= 400) bad.push(`${r.status()} ${r.url()}`); });
  page.on('pageerror', (e) => bad.push(`pageerror ${e.message}`));
  await page.goto(url, { waitUntil: 'load' });
  await page.locator('#demo').scrollIntoViewIfNeeded();
  await page.waitForTimeout(15000);
  await page.screenshot({ path: `${out}/landing.png`, timeout: 120000, fullPage: false });
  const frame = page.frameLocator('iframe');
  console.log('IFRAME BUTTONS', JSON.stringify(await frame.locator('button').allInnerTexts()));
  console.log('BAD', JSON.stringify(bad));
  await browser.close();
})();
