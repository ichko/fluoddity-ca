"""Ablation: what does the brain do, and what does the learned seed velocity do?

For a checkpoint of fluoddity_ca_gecko.py, grow the gecko four ways and score each with the
training loss (multiscale MSE / target power, mean over t = 48, 64, 80, 96):
  full       the trained model as it is
  nobrain    learned seed velocity, brain amplitudes zeroed (no force, no strafe): the seed
             coasts under drag, pressure and the trail only
  noseed     the trained brain, the seed velocity reset to the untrained default
             (radial, --seed_speed) and seed hidden values (if any) unchanged
  neither    both at once

  python3 fca_gecko_ablate.py faithful see_mass
"""
import sys, os, json, argparse, math
import torch
import fluoddity_ca_gecko as f
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(1)
STEPS = (48, 64, 80, 96)


def build(tag):
    ck = torch.load(os.path.join(HERE, f"fca_gecko_{tag}.pt"), weights_only=False)
    a = argparse.Namespace(**ck["args"])
    for k, v in dict(colour=False, pool=0, wcol=0.5, hue_sens=None, hue_off=0.33, train_hue_off=1, seg_min=32, seg_max=64, score_every=4, eval_every=50, learn_seed_h=1, seed_lowres=0, rollback=0, ckpt_min=10, opt="grad").items():
        if not hasattr(a, k): setattr(a, k, v)
    pre = {p["name"]: p for p in f.load_presets()}
    m = f.FCA(a, pre[ck["preset"]]); m.load_state_dict(ck["state"]); m.eval()
    return m, a


def grow(m, tgt, total, steps):
    with torch.no_grad():
        out, _, _ = m.run(total, max(steps), keep=set(steps), ckpt=False)
    return out


def main(tags):
    res = {}
    for tag in tags:
        m, a = build(tag)
        tgt = f.load_target(a); total = float(tgt.sum())
        amp0 = m.amp.detach().clone()
        if a.seed_lowres:   # bake the coarse correction into the full field, drop the correction
            with torch.no_grad(): m.seed_v.copy_(m.seed_vel()); m.seed_c.zero_()
        sv0 = m.seed_v.detach().clone()
        n = m.N; c = (n - 1) / 2.0
        sx = m.xx[m.s0:m.s1, m.s0:m.s1] - c; sy = m.yy[m.s0:m.s1, m.s0:m.s1] - c
        rr = torch.sqrt(sx ** 2 + sy ** 2).clamp_min(1e-3)
        radial = torch.stack([sx / rr, sy / rr]) * a.seed_speed
        cases = {"full": (amp0, sv0), "nobrain": (torch.zeros_like(amp0), sv0),
                 "noseed": (amp0, radial), "neither": (torch.zeros_like(amp0), radial)}
        tiles, r = [], {}
        for name, (amp, sv) in cases.items():
            with torch.no_grad():
                m.amp.copy_(amp); m.seed_v.copy_(sv)
            out = grow(m, tgt, total, sorted(set(STEPS) | {32, 160}))
            L = sum(float(f.ms_loss(out[t], tgt)) for t in STEPS) / len(STEPS)
            r[name] = round(L, 4)
            tiles.append((name, L, out))
            vmax = float(tgt.max()) * 1.2; c0 = (n - 160) // 2
            vs = Image.new("RGB", (512 * 2 + 8, 530), (10, 12, 14)); dd = ImageDraw.Draw(vs)
            vs.paste(Image.fromarray(f.to_img(tgt, vmax)[c0:c0 + 160, c0:c0 + 160]).resize((512, 512), Image.NEAREST), (0, 18))
            vs.paste(Image.fromarray(f.to_img(out[80], vmax)[c0:c0 + 160, c0:c0 + 160]).resize((512, 512), Image.NEAREST), (520, 18))
            dd.text((4, 3), "target", fill=(220, 220, 220)); dd.text((524, 3), f"{tag} / {name}: t=80, loss {L:.3f}", fill=(220, 220, 220))
            vs.save(os.path.join(HERE, f"fca_gecko_{tag}_ablate_{name}.png"))
            print(tag, name, f"{L:.4f}", flush=True)
        res[tag] = r
        with torch.no_grad():
            m.amp.copy_(amp0); m.seed_v.copy_(sv0)
        # one image per tag: rows = case, columns = target, t = 32, 48, 80, 160
        vmax = float(tgt.max()) * 1.2; c0 = (n - 160) // 2
        crop = lambda im: im[c0:c0 + 160, c0:c0 + 160]
        cols = [32, 48, 80, 160]
        W, H = 160 * (len(cols) + 1) + 110, 18 + 162 * len(tiles)
        sheet = Image.new("RGB", (W, H), (10, 12, 14)); d = ImageDraw.Draw(sheet)
        d.text((114, 3), "target", fill=(220, 220, 220))
        for j, t in enumerate(cols): d.text((114 + 160 * (j + 1), 3), f"t={t}", fill=(220, 220, 220))
        for i, (name, L, out) in enumerate(tiles):
            y = 18 + 162 * i
            d.text((4, y + 60), name, fill=(220, 220, 220)); d.text((4, y + 76), f"loss {L:.3f}", fill=(220, 220, 220))
            sheet.paste(Image.fromarray(crop(f.to_img(tgt, vmax))), (110, y))
            for j, t in enumerate(cols):
                sheet.paste(Image.fromarray(crop(f.to_img(out[t], vmax))), (110 + 160 * (j + 1), y))
        p = os.path.join(HERE, f"fca_gecko_{tag}_ablate.png"); sheet.save(p); print(p)
    old = {}
    jp = os.path.join(HERE, "fca_gecko_ablate.json")
    if os.path.exists(jp): old = json.load(open(jp))
    old.update(res); json.dump(old, open(jp, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["faithful", "see_mass"])
