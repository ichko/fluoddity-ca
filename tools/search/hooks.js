__fl.loadObj = o => {
  loadPreset(o);
  look.spread = o.spread; look.pressure = o.pressure; look.rest = o.rest;
  look.brain = Math.max(0, Math.min(S.num_cohorts - 1, Math.round(o.brain)));
  rulesDirty = true; frame = 0;
};
__fl.readback = (G) => {
  const w = W, CN = w.CN, buf = new Float32Array(CN*CN*4);
  gl.bindFramebuffer(gl.FRAMEBUFFER, w.stateF[w.cur]); gl.readBuffer(gl.COLOR_ATTACHMENT0);
  gl.readPixels(0, 0, CN, CN, gl.RGBA, gl.FLOAT, buf);
  const blk = new Float64Array(G*G), hc = new Float64Array(G*G), hs = new Float64Array(G*G);
  let tot = 0, nan = 0, occ = 0, sp = 0, hx = 0, hy = 0, mx = 0;
  const mean = w.N/(CN*CN); const ms = new Float32Array(CN*CN);
  for(let j = 0; j < CN; j++){ const bj = Math.floor(j*G/CN);
    for(let i = 0; i < CN; i++){ const k = (j*CN + i)*4, m = buf[k];
      if(!isFinite(m) || !isFinite(buf[k+1]) || !isFinite(buf[k+3])){ nan++; continue; }
      ms[j*CN+i] = m; if(m <= 0) continue;
      tot += m; if(m > 0.05*mean) occ++; if(m > mx) mx = m;
      sp += Math.hypot(buf[k+1], buf[k+2]);
      const hue = S.hue_sensitivity*buf[k+3]/m, a = 2*Math.PI*hue;
      const c = m*Math.cos(a), s = m*Math.sin(a); hx += c; hy += s;
      const b = bj*G + Math.floor(i*G/CN); blk[b] += m; hc[b] += c; hs[b] += s;
    } }
  ms.sort(); let top = 0; const nt = Math.ceil(CN*CN*0.005); for(let i = 0; i < nt; i++) top += ms[ms.length-1-i];
  return { tot, nan, occ: occ/(CN*CN), top: top/Math.max(tot,1e-9), maxm: mx/mean,
           speed: sp/Math.max(tot,1e-9)*CN/2, hueR: Math.hypot(hx, hy)/Math.max(tot,1e-9), hueA: Math.atan2(hy, hx),
           blk: Array.from(blk, v => +(v/mean).toFixed(4)), hc: Array.from(hc, v => +(v/mean).toFixed(3)), hs: Array.from(hs, v => +(v/mean).toFixed(3)) };
};
