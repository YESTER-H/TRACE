// Press "Shove" repeatedly; after each, screenshot right away and once the robot had time to get up.
const { chromium } = require('playwright');
const [url, out, rounds = '4'] = process.argv.slice(2);
(async () => {
  const browser = await chromium.launch({ channel: 'chromium', args: ['--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=gl-egl'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  const logs = [];
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') logs.push(`[${m.type()}] ${m.text()}`); });
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForTimeout(10000);
  console.log('BUTTONS', JSON.stringify(await page.$$eval('button', (bs) => bs.map((b) => b.innerText.trim()).filter(Boolean))));
  const shove = page.getByRole('button', { name: /shove/i }).first();
  for (let i = 0; i < Number(rounds); i++) {
    await shove.click();
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `${out}/shove${i}_a.png`, timeout: 120000 });
    await page.waitForTimeout(16000);
    await page.screenshot({ path: `${out}/shove${i}_b.png`, timeout: 120000 });
  }
  console.log(logs.slice(0, 40).join('\n'));
  await browser.close();
})();
