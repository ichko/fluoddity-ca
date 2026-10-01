const { chromium } = require(process.env.PLAYWRIGHT || 'playwright');
(async () => {
  const b = await chromium.launch({ args: ['--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist'] });
  for (const [page, extra] of [['fluoddity.html',''], ['fluoddity-ca.html',''], ['fluoddity-ca-next.html','1'], ['fluoddity-ca-next.html','2']]) {
    const p = await b.newPage({ viewport: { width: 300, height: 300 } });
    await p.goto('file://' + require('path').resolve(__dirname, '..') + '/' + page + '?manual');
    await p.waitForTimeout(500);
    const ms = await p.evaluate(extra => {
      __fl.world(1.0); __fl.load('_Default'); if (extra) __fl.look.spread = +extra;
      __fl.step(20);
      const gl = document.getElementById('c').getContext('webgl2'); const px = new Uint8Array(4);
      gl.readPixels(0,0,1,1,gl.RGBA,gl.UNSIGNED_BYTE,px);
      const t = performance.now(); __fl.step(40); gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.readPixels(0,0,1,1,gl.RGBA,gl.UNSIGNED_BYTE,px);
      return (performance.now() - t)/40;
    }, extra);
    console.log(page + (extra ? ' spread ' + extra : ''), ms.toFixed(1), 'ms/step');
    await p.close();
  }
  await b.close();
})();
