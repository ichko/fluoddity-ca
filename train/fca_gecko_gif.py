"""Growth GIF: target and the grown state side by side, from the seed to step T.

  python3 fca_gecko_gif.py TAG [T=500] [every=4]
A --colour checkpoint is drawn as the page draws it (hsv(sens*col + off, 0.8, 1) * mass);
anything else in the amber mass ramp. Writes fca_gecko_<TAG>_grow.gif.
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fluoddity_ca_gecko as f
from fca_gecko_ablate import build
from PIL import Image, ImageDraw

tag = sys.argv[1]
T = int(sys.argv[2]) if len(sys.argv) > 2 else 500
every = int(sys.argv[3]) if len(sys.argv) > 3 else 4
torch.set_num_threads(2)
m, a = build(tag)
tgt = f.load_target(a); total = float(tgt.sum())
steps = list(range(0, T + 1, every))
with torch.no_grad():
    S0, _ = m.init_state(total)
    out, _, _ = m.run(total, T, keep=set(steps), ckpt=False, full=True)
out[0] = S0
n = a.grid
if getattr(a, "colour", False):
    trgb = f.load_target_rgb(a); vmax = float(trgb.max()) * 1.2
    show = lambda S: f.to_rgb_img(m.render(S), vmax)
    timg = f.to_rgb_img(trgb, vmax)
    with torch.no_grad():
        Ls = {s: (float(f.ms_loss(out[s][0].reshape(n, n), tgt)), float(f.ms_loss(m.render(out[s]), trgb))) for s in steps}
    lab = lambda s: f"mass {Ls[s][0]:.3f}  colour {Ls[s][1]:.3f}"
else:
    vmax = float(tgt.max()) * 1.2
    show = lambda S: f.to_img(S[0].reshape(n, n), vmax)
    timg = f.to_img(tgt, vmax)
    Ls = {s: float(f.ms_loss(out[s][0].reshape(n, n), tgt)) for s in steps}
    lab = lambda s: f"mass loss {Ls[s]:.3f}"
c0 = (n - 176) // 2
crop = lambda img: img[c0:c0 + 176, c0:c0 + 176]
tim = Image.fromarray(crop(timg)).resize((352, 352), Image.NEAREST)
frames = []
for s in steps:
    fr = Image.new("RGB", (352 * 2 + 8, 388), (10, 12, 14)); d = ImageDraw.Draw(fr)
    with torch.no_grad():
        fr.paste(tim, (0, 20)); fr.paste(Image.fromarray(crop(show(out[s]))).resize((352, 352), Image.NEAREST), (360, 20))
    d.text((4, 4), "target", fill=(220, 220, 220)); d.text((364, 4), f"{tag}  step {s}", fill=(220, 220, 220))
    d.text((364, 374), lab(s), fill=(170, 170, 170))
    frames.append(fr)
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"fca_gecko_{tag}_grow.gif")
frames[0].save(dst, save_all=True, append_images=frames[1:] + [frames[-1]] * 15, duration=60, loop=0, optimize=True)
print(dst)
