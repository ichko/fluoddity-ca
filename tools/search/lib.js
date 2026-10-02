const { chromium } = require(process.env.PLAYWRIGHT || 'playwright');
const fs = require('fs');
const PAGE = 'file://' + __dirname + '/page.html?manual';
const G = 64;
const ARGS = ['--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist'];

class Worker {
  constructor(id){ this.id = id; }
  async start(){
    this.b = await chromium.launch({ args: ARGS });
    this.p = await this.b.newPage({ viewport: { width: 400, height: 400 } });
    this.errs = []; this.p.on('pageerror', e => this.errs.push(e.message));
    await this.p.goto(PAGE); await this.p.waitForTimeout(500);
    await this.p.evaluate(() => { __fl.world(0.1); __fl.look.speed = 1; });
  }
  async stop(){ try { await this.b.close(); } catch(e){} }
  async evalPreset(o, T = [200, 290, 300]){
    const run = async () => {
      await this.p.evaluate(o => __fl.loadObj(o), o);
      const rbs = []; let s = 0;
      for(const t of T){ await this.p.evaluate(n => __fl.step(n), t - s); s = t; rbs.push(await this.p.evaluate(g => __fl.readback(g), G)); }
      const N = await this.p.evaluate(() => 60000);
      return rbs;
    };
    let to; const tm = new Promise((_, rej) => { to = setTimeout(() => rej(new Error('timeout')), 450000); });
    try { return await Promise.race([run(), tm]); } finally { clearTimeout(to); }
  }
  async snap(o, steps, size = 400){
    await this.p.setViewportSize({ width: size, height: size });
    await this.p.evaluate(() => __fl.frameTick && 0);
    await this.p.evaluate(o => __fl.loadObj(o), o);
    await this.p.evaluate(n => __fl.step(n), steps);
    const url = await this.p.evaluate(() => { __fl.look.speed = 1; return __fl.snap(); });
    return Buffer.from(url.split(',')[1], 'base64');
  }
}

// ---------- features
function fft1(re, im){ // in place radix-2
  const n = re.length;
  for(let i = 1, j = 0; i < n; i++){ let bit = n >> 1; for(; j & bit; bit >>= 1) j ^= bit; j ^= bit;
    if(i < j){ [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
  for(let len = 2; len <= n; len <<= 1){ const a = -2*Math.PI/len, wr = Math.cos(a), wi = Math.sin(a);
    for(let i = 0; i < n; i += len){ let cr = 1, ci = 0;
      for(let j = 0; j < len/2; j++){ const ur = re[i+j], ui = im[i+j];
        const vr = re[i+j+len/2]*cr - im[i+j+len/2]*ci, vi = re[i+j+len/2]*ci + im[i+j+len/2]*cr;
        re[i+j] = ur+vr; im[i+j] = ui+vi; re[i+j+len/2] = ur-vr; im[i+j+len/2] = ui-vi;
        const t = cr*wr - ci*wi; ci = cr*wi + ci*wr; cr = t; } } }
}
function spectrum(a){
  const n = G, mean = a.reduce((s, v) => s + v, 0)/a.length;
  const re = Array.from({length:n}, (_, j) => Array.from({length:n}, (_, i) => a[j*n+i] - mean)), im = re.map(r => r.map(() => 0));
  for(let j = 0; j < n; j++) fft1(re[j], im[j]);
  for(let i = 0; i < n; i++){ const cr = re.map(r => r[i]), ci = im.map(r => r[i]); fft1(cr, ci); for(let j = 0; j < n; j++){ re[j][i] = cr[j]; im[j][i] = ci[j]; } }
  const P = new Float64Array(n/2+1);
  for(let j = 0; j < n; j++) for(let i = 0; i < n; i++){
    const kx = i <= n/2 ? i : i - n, ky = j <= n/2 ? j : j - n, k = Math.round(Math.hypot(kx, ky));
    if(k >= 1 && k <= n/2) P[k] += re[j][i]**2 + im[j][i]**2; }
  let s = 0, sk = 0, pk = 1, pv = -1;
  for(let k = 1; k <= n/2; k++){ s += P[k]; sk += P[k]*Math.log2(k); if(P[k] > pv){ pv = P[k]; pk = k; } }
  return { centroid: s > 0 ? sk/s : 0, peak: Math.log2(pk) };
}
function coherence(a){
  const n = G, v = a.map(x => Math.sqrt(Math.max(x, 0)));
  const at = (i, j) => v[((j+n)%n)*n + ((i+n)%n)];
  const gx = new Float64Array(n*n), gy = new Float64Array(n*n);
  for(let j = 0; j < n; j++) for(let i = 0; i < n; i++){ gx[j*n+i] = (at(i+1,j) - at(i-1,j))/2; gy[j*n+i] = (at(i,j+1) - at(i,j-1))/2; }
  let num = 0, den = 0; const r = 2;
  for(let j = 0; j < n; j++) for(let i = 0; i < n; i++){
    let xx = 0, xy = 0, yy = 0;
    for(let dj = -r; dj <= r; dj++) for(let di = -r; di <= r; di++){ const k = ((j+dj+n)%n)*n + ((i+di+n)%n); xx += gx[k]*gx[k]; xy += gx[k]*gy[k]; yy += gy[k]*gy[k]; }
    const tr = xx + yy; if(tr <= 1e-12) continue;
    const c = Math.sqrt((xx-yy)**2 + 4*xy*xy)/tr; num += c*tr; den += tr; }
  return den > 0 ? num/den : 0;
}
function blobs(a){
  const n = G, seen = new Uint8Array(n*n); let cnt = 0;
  for(let s = 0; s < n*n; s++){ if(seen[s] || a[s] <= 1.0) continue;
    let size = 0; const st = [s]; seen[s] = 1;
    while(st.length){ const k = st.pop(); size++; const i = k % n, j = (k - i)/n;
      for(const [di, dj] of [[1,0],[-1,0],[0,1],[0,-1]]){ const kk = ((j+dj+n)%n)*n + ((i+di+n)%n); if(!seen[kk] && a[kk] > 1.0){ seen[kk] = 1; st.push(kk); } } }
    if(size >= 2) cnt++; }
  return cnt;
}
const moved = (a, b) => { let d = 0, s = 0; for(let i = 0; i < a.length; i++){ d += Math.abs(a[i] - b[i]); s += a[i]; } return 0.5*d/Math.max(s, 1e-9); };
const cvOf = a => { const m = a.reduce((s, v) => s + v, 0)/a.length; return Math.sqrt(a.reduce((s, v) => s + (v-m)**2, 0)/a.length)/Math.max(m, 1e-9); };

const FEAT_NAMES = ['occ','cv','scale','peak','coh','live','drift','hueSpread','hueX','hueY','blobs','speed','top'];
const FEAT_W     = [ 1.2,  1.2,  1.2,   0.6,   1.2,  1.0,   0.6,   0.8,        0.5,   0.5,   1.0,    0.6,   0.6];
function features(rbs){
  const [a, b, c] = rbs, N = c.N;
  const raw = { mass: c.tot/N, nan: a.nan + b.nan + c.nan, occ: c.occ, top: c.top, maxm: c.maxm, speed: c.speed,
    hueR: c.hueR, hueA: c.hueA };
  if(raw.nan > 0 || !isFinite(c.tot)) return { valid: false, why: 'nan', raw };
  raw.cv = cvOf(c.blk); raw.live = moved(b.blk, c.blk); raw.drift = moved(a.blk, c.blk);
  const sp = spectrum(c.blk); raw.centroid = sp.centroid; raw.peak = sp.peak;
  raw.coh = coherence(c.blk); raw.blobs = blobs(c.blk);
  const cl = x => Math.max(0, Math.min(1, x));
  const f = [ raw.occ, cl((Math.log10(raw.cv) + 1)/2), cl(raw.centroid/5), cl(raw.peak/5), cl(raw.coh),
    cl(Math.sqrt(raw.live)/0.8), cl(Math.sqrt(raw.drift)), cl(1 - raw.hueR), raw.hueR*Math.cos(raw.hueA)*0.5 + 0.5, raw.hueR*Math.sin(raw.hueA)*0.5 + 0.5,
    cl(Math.log2(1 + raw.blobs)/9), cl((Math.log10(raw.speed + 0.01) + 2)/3), cl(raw.top*2) ];
  let why = null;
  if(Math.abs(raw.mass - 1) > 0.02) why = 'mass';
  else if(raw.top > 0.5) why = 'collapsed';
  else if(raw.occ < 0.02) why = 'collapsed';
  else if(raw.cv < 0.2) why = 'uniform';
  else if(raw.live < 0.003 && raw.drift < 0.02) why = 'static';
  // quality: penalise near-uniform noise and very static or frantic states
  let q = 1;
  q *= cl((raw.cv - 0.2)/0.4);
  q *= cl(raw.live/0.01 + raw.drift/0.05);
  if(raw.live > 0.4) q *= cl(1 - (raw.live - 0.4)/0.4);
  if(raw.top > 0.3) q *= cl(1 - (raw.top - 0.3)/0.2);
  return { valid: !why, why, raw, f, q: +q.toFixed(3) };
}
function dist(a, b){ let s = 0; for(let i = 0; i < a.length; i++) s += FEAT_W[i]*(a[i]-b[i])**2; return Math.sqrt(s); }
function novelty(f, pool, k = 12, selfId){
  const ds = []; for(const e of pool){ if(e.id === selfId) continue; ds.push(dist(f, e.f)); }
  ds.sort((x, y) => x - y); const kk = Math.min(k, ds.length); if(!kk) return 1;
  return ds.slice(0, kk).reduce((s, v) => s + v, 0)/kk;
}
module.exports = { Worker, features, novelty, dist, FEAT_NAMES, G };
