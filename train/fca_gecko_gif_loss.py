"""Growth GIF with a live loss plot, in colour: target | grown, the loss curves underneath
(mass, and colour when the checkpoint has it), the trained window shaded. Also a strip PNG.

  python3 fca_gecko_gif_loss.py TAG [T=500] [OUT_STEM]
writes OUT_STEM.gif and OUT_STEM_strip.png (default fca_gecko_<TAG>_grow_loss) next to this file,
and prints the losses at 96, 160, 300 and 500.
"""
import sys, os, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fluoddity_ca_gecko as f
from fca_gecko_ablate import build
from PIL import Image, ImageDraw

tag = sys.argv[1]
T = int(sys.argv[2]) if len(sys.argv) > 2 else 500
stem = sys.argv[3] if len(sys.argv) > 3 else f"fca_gecko_{tag}_grow_loss"
torch.set_num_threads(int(os.environ.get("THREADS", "1")))
m, a = build(tag)
n = a.grid
tgt = f.load_target(a); total = float(tgt.sum())
col = f.has_col(a)
if col:
    trgb = f.load_target_rgb(a); vmax = float(trgb.max())
    show = lambda S: f.to_rgb_img(f.view(a, m.render(S)), vmax)
    timg = f.to_rgb_img(f.view(a, trgb), vmax)
else:
    vmax = float(tgt.max()) * 1.2
    show = lambda S: f.to_img(f.view(a, S[0].reshape(n, n)), vmax)
    timg = f.to_img(f.view(a, tgt), vmax)
c0 = (n // getattr(a, "zoom", 1) - 176) // 2   # --zoom: shown at the target's resolution (z x z averages), as scored
crop = lambda img: img[c0:c0 + 176, c0:c0 + 176]
EVERY = 2
strip_steps = [0, 16, 32, 48, 64, 96, 160, 300, 500]
imgs, lm, lc, strip = {}, [], [], {}
with torch.no_grad():
    S, Tr = m.init_state(total)
    for s in range(T + 1):
        if s > 0: S, Tr = m.step(S, Tr)
        _, l1, l2 = f.frame_loss(m, S, tgt, trgb if col else None, a)
        lm.append(float(l1)); lc.append(float(l2))
        if s % EVERY == 0: imgs[s] = crop(show(S))
        if s in strip_steps: strip[s] = crop(show(S))
Z = 352; PH = 160; W = Z * 2 + 8
tim = Image.fromarray(crop(timg)).resize((Z, Z), Image.NEAREST)
ymax = 1.0
w0, w1 = a.t0, a.t1 + a.hold
def plot(d, upto):
    x0, y0, w, h = 50, Z + 32, W - 70, PH - 50
    d.rectangle([x0, y0, x0 + w, y0 + h], outline=(60, 64, 70))
    for yv in (0.25, 0.5, 0.75):
        yy = y0 + h - yv / ymax * h; d.line([x0, yy, x0 + w, yy], fill=(36, 40, 46)); d.text((x0 - 34, yy - 6), f"{yv:.2f}", fill=(140, 140, 140))
    xa, xb = x0 + w0 / T * w, x0 + min(w1, T) / T * w
    d.rectangle([xa, y0 + 1, xb, y0 + h - 1], fill=(30, 40, 30)); d.text((xa + 2, y0 + 2), "trained", fill=(110, 150, 110))
    for L, c, c2 in ([(lm, (240, 170, 70), (255, 210, 120))] + ([(lc, (90, 200, 120), (150, 240, 170))] if col else [])):
        pts = [(x0 + s / T * w, y0 + h - min(L[s], ymax) / ymax * h) for s in range(T + 1)]
        d.line(pts, fill=(70, 70, 70), width=1)
        if upto > 0: d.line(pts[:upto + 1], fill=c, width=2)
        px, py = pts[upto]; d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=c2)
    d.text((x0, y0 + h + 6), "0", fill=(140, 140, 140)); d.text((x0 + w - 20, y0 + h + 6), str(T), fill=(140, 140, 140))
    d.text((x0 + w / 2 - 20, y0 + h + 6), "step", fill=(140, 140, 140))
    d.text((x0 + 4, y0 + h + 22), "mass loss", fill=(240, 170, 70))
    if col: d.text((x0 + 90, y0 + h + 22), "colour loss", fill=(90, 200, 120))
frames = []
for s in sorted(imgs):
    fr = Image.new("RGB", (W, Z + 24 + PH), (10, 12, 14)); d = ImageDraw.Draw(fr)
    fr.paste(tim, (0, 20)); fr.paste(Image.fromarray(imgs[s]).resize((Z, Z), Image.NEAREST), (Z + 8, 20))
    d.text((4, 4), "target", fill=(220, 220, 220))
    d.text((Z + 12, 4), f"grown: step {s:3d}   mass {lm[s]:.3f}" + (f"   colour {lc[s]:.3f}" if col else ""), fill=(220, 220, 220))
    plot(d, s); frames.append(fr)
dst = os.path.join(HERE, stem + ".gif")
frames[0].save(dst, save_all=True, append_images=frames[1:] + [frames[-1]] * 20, duration=50, loop=0, optimize=True)
ss = [s for s in strip_steps if s <= T]
sheet = Image.new("RGB", (176 * (len(ss) + 1), 194), (10, 12, 14)); d = ImageDraw.Draw(sheet)
sheet.paste(Image.fromarray(crop(timg)), (0, 18)); d.text((4, 3), "target", fill=(220, 220, 220))
for i, s in enumerate(ss):
    sheet.paste(Image.fromarray(strip[s]), (176 * (i + 1), 18))
    d.text((176 * (i + 1) + 4, 3), f"t={s} m{lm[s]:.3f}" + (f" c{lc[s]:.3f}" if col else ""), fill=(220, 220, 220))
sheet.save(os.path.join(HERE, stem + "_strip.png"))
print(dst, {s: (round(lm[s], 4), round(lc[s], 4)) for s in (48, 64, 80, 96, 160, 300, 500) if s <= T})
