// steady-state novelty search over CA presets. Resumable: archive.jsonl.
const L = require('./lib'); const fs = require('fs');
const P = eval(fs.readFileSync('../presets.js','utf8') + ';PRESETS');
const WS = 0.05, T = [140, 210, 220];
const NW = +(process.env.NW || 2), BUDGET_H = +(process.env.HOURS || 4);
const ARCH = __dirname + '/archive.jsonl', LOG = __dirname + '/log.txt';
const log = s => { const l = new Date().toISOString().slice(11,19) + ' ' + s; fs.appendFileSync(LOG, l + '\n'); console.log(l); };
const rnd = (a, b) => a + Math.random()*(b - a), ri = n => Math.floor(Math.random()*n), gauss = () => { let u = 0, v = 0; while(!u) u = Math.random(); v = Math.random(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); };
const pick = a => a[ri(a.length)];
const R = { axial_force:[-1.2,1.2], lateral_force:[-3,3], sensor_gain:[-50,70], mutation_scale:[-0.5,1], drag:[-0.95,0.95], strafe_power:[0,1.2],
  sensor_angle:[-1,1], global_force_mult:[0,8], sensor_distance:[-1.2,4], trail_persistence:[0.4,0.999], trail_diffusion:[0,1], orientation_mix:[0,1], hue_sensitivity:[-1,1] };
const FROZEN = new Set([10, 11, 32, 33, 60, 61]);   // feed the mutation hash: touching them reshuffles the brain
const cl = (v, [a, b]) => Math.max(a, Math.min(b, v));
const KEEP = ['axial_force','lateral_force','sensor_gain','mutation_scale','drag','strafe_power','sensor_angle','global_force_mult','sensor_distance','trail_persistence','trail_diffusion','hazard_rate','disable_symmetry','absolute_orientation','orientation_mix','boundary_conditions','initial_conditions','num_cohorts','rule_seed','ink_weight','hue_sensitivity','color_by_cohort','watercolor_mode','rule','spread','pressure','rest','brain'];
const clean = o => { const r = {}; for(const k of KEEP) r[k] = o[k]; return r; };
function randKnobs(o){ o.spread = +rnd(1, 3.5).toFixed(2); o.pressure = +(Math.random() < 0.5 ? rnd(0, 0.3) : rnd(0, 1.5)).toFixed(3);
  o.rest = +rnd(0.5, 4).toFixed(2); o.brain = ri(o.num_cohorts); return o; }

function mutate(par, archive){
  const o = JSON.parse(JSON.stringify(par.o)); const ops = [];
  const nops = Math.random() < 0.3 ? 2 : 1;
  for(let n = 0; n < nops; n++){
    const r = Math.random();
    if(r < 0.5){
      const ks = Object.keys(R), m = 1 + ri(3);
      for(let i = 0; i < m; i++){ const k = pick(ks), rg = R[k], span = rg[1] - rg[0];
        if(k === 'trail_persistence'){ const e = Math.max(1e-3, 1 - o[k])*Math.exp(0.25*gauss()); o[k] = +cl(1 - e, rg).toFixed(4); }
        else o[k] = +cl(o[k]*(1 + 0.12*gauss()) + 0.02*span*gauss(), rg).toFixed(4);
        ops.push(k); }
    } else if(r < 0.7){
      if(o.rule.every(x => x === 0)){ if(Math.random() < 0.3){ o.rule_seed = Math.random(); ops.push('seed'); } else { o.mutation_scale = +cl(o.mutation_scale*(1+0.15*gauss()) + 0.02*gauss(), R.mutation_scale).toFixed(4); ops.push('mutation_scale'); } continue; }
      const amp = rnd(0.05, 0.2);
      for(let k = 0; k < 80; k++){ if(FROZEN.has(k) || Math.random() > 0.25) continue;
        const isFreq = (Math.floor(k/4) % 2) === 0;
        o.rule[k] = isFreq ? o.rule[k]*(1 + 0.1*gauss()) : o.rule[k] + amp*gauss(); o.rule[k] = +o.rule[k].toFixed(5); }
      ops.push('rule');
    } else if(r < 0.87){
      const k = pick(['spread','pressure','rest','brain']);
      if(Math.random() < 0.5){ // small nudge
        if(k === 'spread') o.spread = +cl(o.spread + 0.25*gauss(), [1, 4]).toFixed(2);
        if(k === 'pressure') o.pressure = +cl(o.pressure*(1 + 0.2*gauss()) + 0.05*gauss(), [0, 2]).toFixed(3);
        if(k === 'rest') o.rest = +cl(o.rest*(1 + 0.2*gauss()), [0.3, 6]).toFixed(2);
        if(k === 'brain') o.brain = (o.brain + pick([-1, 1]) + o.num_cohorts) % o.num_cohorts;
      } else { const t = randKnobs({num_cohorts: o.num_cohorts}); o[k] = t[k]; }
      ops.push('k:' + k);
    } else if(r < 0.94 && archive.length > 5){
      const other = pick(archive).o;
      for(const k of ['rule','rule_seed','num_cohorts','brain','mutation_scale']) o[k] = JSON.parse(JSON.stringify(other[k]));
      o.brain = Math.min(o.brain, o.num_cohorts - 1); ops.push('cross');
    } else {
      const k = pick(['boundary_conditions','initial_conditions','absolute_orientation','disable_symmetry','seed']);
      if(k === 'seed') o.rule_seed = Math.random();
      else if(k === 'disable_symmetry') o.disable_symmetry = !o.disable_symmetry;
      else if(k === 'absolute_orientation') o.absolute_orientation = ri(3);
      else o[k] = ri(3);
      ops.push(k);
    }
  }
  return { o, ops };
}

const archive = [];   // {id, parent, origin, depth, ops, o, valid, why, f, q, raw}
let nextId = 0;
if(fs.existsSync(ARCH)) for(const l of fs.readFileSync(ARCH, 'utf8').split('\n')) if(l.trim()){ const e = JSON.parse(l); archive.push(e); nextId = Math.max(nextId, e.id + 1); }
const valid = () => archive.filter(e => e.valid && e.q >= 0.3);

// resumable seed phase: seed presets not yet in the archive, up to SEEDS total
const SEEDS = +(process.env.SEEDS || 50);
const seeded = new Set(archive.filter(e => e.depth === 0).map(e => e.origin));
const seedQueue = (() => { const core = P.filter(p => p.group === 'Core' && !seeded.has(p.name)), adv = P.filter(p => p.group !== 'Core' && !seeded.has(p.name)).sort(() => Math.random() - 0.5);
  return core.concat(adv).slice(0, Math.max(0, SEEDS - seeded.size)).map(p => ({ o: randKnobs(clean(p)), parent: null, origin: p.name, depth: 0, ops: ['seed'] })); })();
function nextCandidate(){
  if(seedQueue.length && (Math.random() < 0.5 || valid().length < 8)) return seedQueue.shift();
  const V = valid(); if(!V.length) return null;
  // tournament on novelty x quality
  let best = null, bs = -1;
  for(let t = 0; t < 5; t++){ const e = pick(V); const s = L.novelty(e.f, V, 12, e.id)*(0.4 + 0.6*e.q); if(s > bs){ bs = s; best = e; } }
  const { o, ops } = mutate(best, V);
  return { o, parent: best.id, origin: best.origin, depth: best.depth + 1, ops };
}

const t0 = Date.now(); let busy = 0, done = 0;
async function runWorker(i){
  let w = new L.Worker(i); await w.start(); await w.p.evaluate(ws => __fl.world(ws), WS);
  while((Date.now() - t0)/3.6e6 < BUDGET_H && !fs.existsSync(__dirname + '/STOP')){
    const c = nextCandidate(); if(!c){ await new Promise(r => setTimeout(r, 2000)); continue; }
    const id = nextId++; const ts = Date.now(); let F;
    try { F = L.features(await w.evalPreset(c.o, T)); }
    catch(e){ log(`w${i} #${id} ${c.origin}/d${c.depth} [${c.ops.join(',')}] error ${e.message}; restarting worker`);
      if(/timeout/.test(e.message)){ const x = { id, parent: c.parent, origin: c.origin, depth: c.depth, ops: c.ops, o: c.o, valid: false, why: 'timeout', f: null, q: 0, raw: {}, t: Date.now() }; archive.push(x); fs.appendFileSync(ARCH, JSON.stringify(x) + '\n'); } await w.stop(); w = new L.Worker(i); await w.start(); await w.p.evaluate(ws => __fl.world(ws), WS); continue; }
    const e = { id, parent: c.parent, origin: c.origin, depth: c.depth, ops: c.ops, o: c.o, valid: F.valid, why: F.why, f: F.f, q: F.q, raw: F.raw, t: Date.now() };
    archive.push(e); fs.appendFileSync(ARCH, JSON.stringify(e) + '\n'); done++;
    const V = valid(); const nov = e.valid ? L.novelty(e.f, V, 12, e.id) : 0;
    log(`w${i} #${id} ${c.origin}/d${c.depth} [${c.ops.join(',')}] ${e.valid ? 'ok' : 'X ' + e.why} q=${e.q} nov=${nov.toFixed(3)} live=${(F.raw.live||0).toFixed(3)} cv=${(F.raw.cv||0).toFixed(2)} occ=${(F.raw.occ||0).toFixed(2)} ${((Date.now()-ts)/1000).toFixed(0)}s`);
    if(done % 30 === 0){ const vv = valid(); const ns = vv.map(x => L.novelty(x.f, vv, 12, x.id)); log(`== ${archive.length} evaluated, ${vv.length} valid(q>=.3), mean nov ${(ns.reduce((a,b)=>a+b,0)/ns.length).toFixed(3)}, max depth ${Math.max(...archive.map(x=>x.depth))}`); }
  }
  await w.stop();
}
log(`start: archive ${archive.length}, seeds ${seedQueue.length}, workers ${NW}, budget ${BUDGET_H}h`);
Promise.all(Array.from({length: NW}, (_, i) => runWorker(i))).then(() => log('done ' + archive.length));
