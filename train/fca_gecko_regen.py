"""Regeneration: the robustness benchmark and the damage GIF (Growing NCA, experiment 3).

A gecko is grown from the seed to step 200, cut (fluoddity_ca_gecko.std_damages: tail, head,
a 24-cell-wide hole in the body, 30% of the cells at random), every per-cell number and the
trail under the cut zeroed, the lost mass given back by scaling the rest up (--reinject scale,
see damage_state), and run on. Mass loss (multiscale MSE / target power, the training loss)
at +50, +100, +300 steps after the cut, against the same model left uncut.

  python3 fca_gecko_regen.py bench OUT.json TAG [TAG ...]
  python3 fca_gecko_regen.py gif TAG [CUT=head] [OUT_STEM] [LABEL] [MODE=edge]
"""
import sys, os, json, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fluoddity_ca_gecko as f
from fca_gecko_ablate import build
from PIL import Image, ImageDraw
import numpy as np

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
T_CUT, AFTER, T_END = 200, (50, 100, 300), 500


def bench(tag, modes=("edge", "scale")):
    m, a = build(tag)
    tgt = f.load_target(a); total = float(tgt.sum()); tv = f.view(a, tgt)
    L = lambda S: round(float(f.ms_loss(f.view(a, S[0].reshape(m.N, m.N)), tv)), 4)
    res = {"undamaged": {str(k): v for k, v in f.long_eval(m, a, tgt, total).items()}}
    with torch.no_grad():
        _, S, T = m.run(total, T_CUT, ckpt=False)
        res["at_cut"] = L(S)
        out, _, _ = m.run(total, max(AFTER), keep=set(AFTER), ckpt=False, full=True, state=(S, T))
        res["uncut"] = {f"+{k}": L(out[k]) for k in AFTER}
        for mode in modes:
            for nm, cut in f.std_damages(m, a).items():
                frac = float((S[0] * cut).sum()) / total
                S2, T2 = f.damage_state(m, S, T, cut, total, mode)
                out, _, _ = m.run(total, max(AFTER), keep=set(AFTER), ckpt=False, full=True, state=(S2, T2))
                res[f"{nm}/{mode}"] = {"cut_frac": round(frac, 3), "+0": L(S2), **{f"+{k}": L(out[k]) for k in AFTER}}
    return res


def gif(tag, cut_name="head", stem=None, label=None, mode="edge"):
    stem = stem or f"fca_gecko_{tag}_regen"
    label = label or tag
    m, a = build(tag)
    n = a.grid
    tgt = f.load_target(a); total = float(tgt.sum())
    col = f.has_col(a)
    trgb = f.load_target_rgb(a) if col else None
    if col:
        vmax = float(trgb.max()); show = lambda S: f.to_rgb_img(f.view(a, m.render(S)), vmax)
        timg = f.to_rgb_img(f.view(a, trgb), vmax)
    else:
        vmax = float(tgt.max()) * 1.2; show = lambda S: f.to_img(f.view(a, S[0].reshape(n, n)), vmax)
        timg = f.to_img(f.view(a, tgt), vmax)
    z = getattr(a, "zoom", 1)
    c0 = (n // z - 176) // 2
    crop = lambda img: img[c0:c0 + 176, c0:c0 + 176]
    cut = f.std_damages(m, a)[cut_name]
    # the cut's outline, in the shown (flipped, cropped) picture
    cm = f.view(a, cut.reshape(n, n)).numpy()[::-1] > 0.5
    edge = cm & ~(np.roll(cm, 1, 0) & np.roll(cm, -1, 0) & np.roll(cm, 1, 1) & np.roll(cm, -1, 1))
    edge = crop(edge)
    lm, lc, lm_ref, imgs, strip = [], [], [], {}, {}
    strip_steps = [0, 48, 96, 200, "cut", 210, 225, 250, 300, 400, 500]
    EVERY = 2
    with torch.no_grad():
        S, Tr = m.init_state(total)
        for s in range(T_END + 1):
            if s > 0: S, Tr = m.step(S, Tr)
            _, l1, l2 = f.frame_loss(m, S, tgt, trgb, a)
            lm.append(float(l1)); lc.append(float(l2))
            if s == T_CUT:
                Sref, Tref = S.clone(), Tr.clone()
                pre = crop(show(S))
                S, Tr = f.damage_state(m, S, Tr, cut, total, mode)
                frac_txt = ""
                _, l1, l2 = f.frame_loss(m, S, tgt, trgb, a)
                lm[-1], lc[-1] = float(l1), float(l2)   # the curve shows the cut at 200
                post = crop(show(S)); strip["cut"] = post
                imgs["cut"] = post
            if s % EVERY == 0 and s != T_CUT: imgs[s] = crop(show(S))
            if s in strip_steps: strip[s] = crop(show(S)) if s != T_CUT else pre
        # the same gecko left uncut, for reference
        lm_ref = lm[:T_CUT]
        S, Tr = Sref, Tref
        lm_ref.append(lm_pre := float(f.frame_loss(m, S, tgt, trgb, a)[1]))
        for s in range(T_CUT + 1, T_END + 1):
            S, Tr = m.step(S, Tr); lm_ref.append(float(f.frame_loss(m, S, tgt, trgb, a)[1]))
    Z = 352; PH = 170; W = Z * 2 + 8
    tim = Image.fromarray(crop(timg)).resize((Z, Z), Image.NEAREST)
    ymax = 0.5
    T = T_END
    def plot(d, upto):
        x0, y0, w, h = 50, Z + 32, W - 70, PH - 60
        d.rectangle([x0, y0, x0 + w, y0 + h], outline=(60, 64, 70))
        for yv in (0.1, 0.2, 0.3, 0.4):
            yy = y0 + h - yv / ymax * h; d.line([x0, yy, x0 + w, yy], fill=(36, 40, 46)); d.text((x0 - 30, yy - 6), f"{yv:.1f}", fill=(140, 140, 140))
        xc = x0 + T_CUT / T * w
        d.line([xc, y0 + 1, xc, y0 + h - 1], fill=(200, 60, 60), width=2); d.text((xc + 4, y0 + 2), f"cut: {cut_name}", fill=(230, 90, 90))
        pref = [(x0 + s / T * w, y0 + h - min(lm_ref[s], ymax) / ymax * h) for s in range(T_CUT, T + 1)]
        for k in range(0, len(pref) - 1, 6): d.line(pref[k:k + 3], fill=(120, 120, 120), width=1)   # dashed: uncut
        curves = [(lm, (240, 170, 70), (255, 210, 120))] + ([(lc, (90, 200, 120), (150, 240, 170))] if col else [])
        for L, c, c2 in curves:
            pts = [(x0 + s / T * w, y0 + h - min(L[s], ymax) / ymax * h) for s in range(T + 1)]
            d.line(pts, fill=(60, 60, 60), width=1)
            if upto > 0: d.line(pts[:upto + 1], fill=c, width=2)
            px, py = pts[upto]; d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=c2)
        d.text((x0, y0 + h + 6), "0", fill=(140, 140, 140)); d.text((x0 + w - 20, y0 + h + 6), str(T), fill=(140, 140, 140))
        d.text((x0 + w / 2 - 20, y0 + h + 6), "step", fill=(140, 140, 140))
        d.text((x0 + 4, y0 + h + 22), "mass loss", fill=(240, 170, 70))
        if col: d.text((x0 + 90, y0 + h + 22), "colour loss", fill=(90, 200, 120))
        d.text((x0 + 190, y0 + h + 22), "- - mass loss, same gecko uncut", fill=(140, 140, 140))
    def outlined(im):
        im = im.copy(); im[edge] = (230, 60, 60); return im
    frames = []
    def frame(s, im, txt, mark=False):
        fr = Image.new("RGB", (W, Z + 24 + PH), (10, 12, 14)); d = ImageDraw.Draw(fr)
        fr.paste(tim, (0, 20)); fr.paste(Image.fromarray(outlined(im) if mark else im).resize((Z, Z), Image.NEAREST), (Z + 8, 20))
        d.text((4, 4), "target", fill=(220, 220, 220))
        d.text((Z + 12, 4), f"{label}: step {s:3d}  {txt}  mass {lm[s]:.3f}" + (f"  colour {lc[s]:.3f}" if col else ""), fill=(220, 220, 220))
        plot(d, s); return fr
    for s in range(0, T_CUT, EVERY): frames.append(frame(s, imgs[s], "growing"))
    fpre = frame(T_CUT, pre, "about to cut", mark=True); fpre_l = lm[T_CUT]
    frames += [fpre] * 12
    frames += [frame(T_CUT, imgs["cut"], "CUT", mark=True)] * 16
    for s in range(T_CUT + EVERY, T + 1, EVERY): frames.append(frame(s, imgs[s], "regrowing", mark=s < T_CUT + 20))
    dst = os.path.join(HERE, stem + ".gif")
    frames[0].save(dst, save_all=True, append_images=frames[1:] + [frames[-1]] * 20, duration=50, loop=0, optimize=True)
    ss = strip_steps
    sheet = Image.new("RGB", (176 * (len(ss) + 1), 194), (10, 12, 14)); d = ImageDraw.Draw(sheet)
    sheet.paste(Image.fromarray(crop(timg)), (0, 18)); d.text((4, 3), "target", fill=(220, 220, 220))
    for i, s in enumerate(ss):
        im = strip[s]
        if s in ("cut", T_CUT): im = outlined(im)
        sheet.paste(Image.fromarray(im), (176 * (i + 1), 18))
        lab = f"cut {cut_name} m{lm[T_CUT]:.3f}" if s == "cut" else (f"t={s} (pre-cut)" if s == T_CUT else f"t={s} m{lm[s]:.3f}")
        d.text((176 * (i + 1) + 4, 3), lab, fill=(220, 220, 220))
    sheet.save(os.path.join(HERE, stem + "_strip.png"))
    print(dst, {"pre": round(lm_pre, 4), **{f"+{k}": round(lm[T_CUT + k], 4) for k in (0, 50, 100, 300)},
                "uncut+300": round(lm_ref[T_END], 4)})


if __name__ == "__main__":
    if sys.argv[1] == "bench":
        outp = sys.argv[2]
        res = {}
        if os.path.exists(outp): res = json.load(open(outp))
        for tag in sys.argv[3:]:
            res[tag] = bench(tag); print(tag, json.dumps(res[tag]), flush=True)
            json.dump(res, open(outp, "w"), indent=1)
    else:
        gif(sys.argv[2], *(sys.argv[3:]))
