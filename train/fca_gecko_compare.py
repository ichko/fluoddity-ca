"""Every variant's best checkpoint grown side by side, and a summary JSON.

  python3 fca_gecko_compare.py faithful see_mass centres48 hidden6 seedlow3
writes fca_gecko_compare.png (rows = variants, columns = target and t = 16 ... 160) and
fca_gecko_summary.json (loss per variant at each step, trainable numbers, iterations).
"""
import sys, os, json, argparse
import torch
import fluoddity_ca_gecko as f
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(1)
COLS = [16, 32, 48, 64, 80, 96, 128, 160]
DEFAULTS = dict(colour=False, pool=0, wcol=0.5, hue_sens=None, hue_off=0.33, train_hue_off=1, seg_min=32, seg_max=64, score_every=4, eval_every=50, learn_seed_h=1, seed_lowres=0, rollback=0, ckpt_min=10, opt="grad")


def main(tags):
    rows, summ = [], {}
    for tag in tags:
        p = os.path.join(HERE, f"fca_gecko_{tag}.pt")
        if not os.path.exists(p): continue
        ck = torch.load(p, weights_only=False)
        a = argparse.Namespace(**{**DEFAULTS, **ck["args"]})
        pre = {q["name"]: q for q in f.load_presets()}
        m = f.FCA(a, pre[ck["preset"]]); m.load_state_dict(ck["state"])
        tgt = f.load_target(a); total = float(tgt.sum())
        with torch.no_grad():
            out, S, _ = m.run(total, max(COLS), keep=set(COLS), ckpt=False)
        Ls = {t: float(f.ms_loss(out[t], tgt)) for t in COLS}
        fit = sum(Ls[t] for t in (48, 64, 80, 96)) / 4
        ntrain = sum(q.numel() for q in m.parameters() if q.requires_grad)
        summ[tag] = dict(loss_fit=round(fit, 4), loss_at={t: round(v, 4) for t, v in Ls.items()},
                         train_best_ema=ck.get("best"), iterations=ck.get("it", -1) + 1, trainable=ntrain,
                         preset=ck["preset"], mass_drift=float(S[0].sum()) / total - 1,
                         flags={k: ck["args"].get(k) for k in ("see_mass", "centres", "hidden", "seed_lowres", "learn_seed_h", "learn_seed_v", "init_from", "gdamp", "lr")})
        rows.append((tag, fit, ntrain, tgt, out))
        print(tag, summ[tag]["loss_fit"], ntrain, flush=True)
    n = 256; c0 = (n - 160) // 2
    crop = lambda im: im[c0:c0 + 160, c0:c0 + 160]
    W, H = 120 + 160 * (len(COLS) + 1), 18 + 162 * len(rows)
    sheet = Image.new("RGB", (W, H), (10, 12, 14)); d = ImageDraw.Draw(sheet)
    d.text((124, 3), "target", fill=(220, 220, 220))
    for j, t in enumerate(COLS): d.text((124 + 160 * (j + 1), 3), f"t={t}", fill=(220, 220, 220))
    for i, (tag, fit, ntrain, tgt, out) in enumerate(rows):
        y = 18 + 162 * i; vmax = float(tgt.max()) * 1.2
        d.text((4, y + 56), tag, fill=(220, 220, 220)); d.text((4, y + 72), f"loss {fit:.3f}", fill=(220, 220, 220))
        d.text((4, y + 88), f"{ntrain} numbers", fill=(160, 160, 160))
        sheet.paste(Image.fromarray(crop(f.to_img(tgt, vmax))), (120, y))
        for j, t in enumerate(COLS):
            sheet.paste(Image.fromarray(crop(f.to_img(out[t], vmax))), (120 + 160 * (j + 1), y))
    sheet.save(os.path.join(HERE, "fca_gecko_compare.png"))
    json.dump(summ, open(os.path.join(HERE, "fca_gecko_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["faithful", "see_mass", "centres48", "hidden6", "seedlow3"])
