const { chromium } = require(process.env.PLAYWRIGHT || 'playwright');
const fs = require('fs');
(async () => {
  const b = await chromium.launch({ args: ['--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 500, height: 500 } });
  const logs = []; p.on('pageerror', e => logs.push('ERR '+e.message)); p.on('console', m => { if (m.type()==='error') logs.push(m.text()); });
  await p.goto('file://' + require('path').resolve(__dirname, '..', 'fluoddity-ca-next.html') + '?manual');
  await p.waitForTimeout(800);
  await p.evaluate(() => __fl.world(0.1));
  const [name, steps, ...spreads] = process.argv.slice(2);
  for (const s of spreads) {
    await p.evaluate(([n, s]) => { __fl.load(n); __fl.look.spread = +s; }, [name, s]);
    await p.evaluate(st => __fl.step(+st), steps);
    const m = await p.evaluate(() => __fl.mass());
    const url = await p.evaluate(() => __fl.snap());
    fs.writeFileSync(`spread-${name}-${s}.png`, Buffer.from(url.split(',')[1], 'base64'));
    console.log(name, 'spread', s, 'mass', m[0].toFixed(2), 'occupied', (100*m[1]).toFixed(0)+'%');
  }
  console.log(logs.filter(l => !/CERT/.test(l)).join('\n'));
  await b.close();
})();
