const { chromium } = require(process.env.PLAYWRIGHT || 'playwright');
(async () => {
  const b = await chromium.launch({ args: ['--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 400, height: 400 } });
  await p.goto('file://' + require('path').resolve(__dirname, '..', 'fluoddity-ca.html') + '?manual');
  await p.waitForTimeout(800);
  await p.evaluate(() => __fl.world(0.1));
  const names = await p.evaluate(() => __fl.presets.map(c => c.name));
  const [K, S] = process.argv.slice(2).map(Number);
  for (const name of names.filter((_, i) => i % S === K)) {
    await p.evaluate(n => __fl.load(n), name);
    let worst = {max: 0}, rows = [];
    for (let k = 0; k < 6; k++) {
      await p.evaluate(() => __fl.step(50));
      const j = await p.evaluate(() => __fl.jumps());
      rows.push(j); if (j.max > worst.max) worst = j;
    }
    const f = x => x.toFixed(2);
    console.log([name, f(worst.max), f(Math.max(...rows.map(r=>r.p999))), f(Math.max(...rows.map(r=>r.p99))), f(Math.max(...rows.map(r=>r.p50)))].join('\t'));
  }
  await b.close();
})();
