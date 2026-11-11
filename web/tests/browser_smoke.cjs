// Headless browser smoke test of the built demo: load, list controls, fire the lie-down events,
// and take screenshots. usage: NODE_PATH=<mjswan template>/node_modules node browser_smoke.cjs <url> <out_dir>
const { chromium } = require('playwright');
const [url, out] = process.argv.slice(2);
const shot = (page, name) => page.screenshot({ path: `${out}/${name}.png`, timeout: 120000 });
(async () => {
  const browser = await chromium.launch({
    channel: 'chromium',
    args: ['--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=gl-egl', '--enable-features=Vulkan'],
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  const logs = [];
  page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForTimeout(15000);
  console.log(logs.slice(0, 60).join('\n'));
  const gl = await page.evaluate(() => {
    const c = document.createElement('canvas').getContext('webgl2');
    const ext = c && c.getExtension('WEBGL_debug_renderer_info');
    return c ? c.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : c.RENDERER) : 'no webgl2';
  });
  console.log('GL', gl);
  const buttons = await page.$$eval('button', (bs) => bs.map((b) => b.innerText.trim()).filter(Boolean));
  console.log('BUTTONS', JSON.stringify(buttons));
  await shot(page, '00_loaded_15s');
  for (const [pattern, name] of [[/face up/i, '01_face_up'], [/face down/i, '02_face_down']]) {
    const b = page.getByRole('button', { name: pattern });
    if (await b.count()) {
      await b.first().click();
      await page.waitForTimeout(1500); await shot(page, `${name}_0s`);
      await page.waitForTimeout(10000); await shot(page, `${name}_10s`);
    } else console.log('no button', pattern);
  }
  console.log(logs.slice(60, 120).join('\n'));
  await browser.close();
})();
