// node snap.js WS STEPS NWORKERS id1 id2 ...   (ids from archive.jsonl, or file:path.json for an array of presets)
const L = require('./lib'); const fs = require('fs');
const [ws, steps, nw, ...ids] = process.argv.slice(2);
let items;
if(ids[0].startsWith('file:')) items = JSON.parse(fs.readFileSync(ids[0].slice(5))).map((o, i) => ({ key: o.name || ('p' + i), o }));
else { const A = fs.readFileSync(__dirname + '/archive.jsonl', 'utf8').trim().split('\n').map(l => JSON.parse(l)); items = ids.map(id => ({ key: id, o: A.find(e => e.id === +id).o })); }
(async () => {
  const q = items.slice();
  await Promise.all(Array.from({length: +nw}, async () => {
    const w = new L.Worker(0); await w.start(); await w.p.evaluate(ws => { __fl.world(+ws); __fl.look.speed = 1; }, ws);
    while(q.length){ const it = q.shift(); const t = Date.now();
      const png = await w.snap(it.o, +steps, 400);
      fs.writeFileSync(`${__dirname}/snaps/${it.key}-${ws}-${steps}.png`, png);
      const m = await w.p.evaluate(() => __fl.mass());
      console.log(it.key, 'mass', m[0].toFixed(1), 'occ', m[1].toFixed(2), ((Date.now()-t)/1000).toFixed(0) + 's'); }
    if(w.errs.length) console.log('ERRS', w.errs); await w.stop();
  }));
})();
