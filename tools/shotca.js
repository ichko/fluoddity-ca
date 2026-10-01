const { chromium } = require(process.env.PLAYWRIGHT || 'playwright');
const fs = require('fs');
(async () => {
  const b = await chromium.launch({ args: ['--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 700, height: 600 } });
  const logs = []; p.on('console', m => logs.push(m.type()+': '+m.text())); p.on('pageerror', e => logs.push('ERR '+e.message));
  await p.goto('file://' + require('path').resolve(__dirname, '..', 'fluoddity-ca.html') + '?manual');
  await p.waitForTimeout(1000);
  const [ws, steps, ...names] = process.argv.slice(2);
  await p.evaluate(w => { __fl.world(+w); __fl.look.speed = 5; }, ws);
  for (const name of names) {
    const t0 = Date.now();
    await p.evaluate(n => __fl.load(n), name);
    await p.evaluate(s => { __fl.step(+s); }, steps);
    const url = await p.evaluate(() => __fl.snap());
    fs.writeFileSync(`ca-${name.replace(/\W/g,'')}.png`, Buffer.from(url.split(',')[1], 'base64'));
    console.log(name, 'mass', JSON.stringify(await p.evaluate(() => __fl.mass())), 'frame', await p.evaluate(() => __fl.frame), (Date.now()-t0)+'ms');
  }
  console.log(logs.filter(l=>!/CERT/.test(l)).slice(-8).join('\n'));
  await b.close();
})();
