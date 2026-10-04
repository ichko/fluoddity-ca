"""Export a trained Fluoddity-CA gecko (fluoddity_ca_gecko.py checkpoint with --paint) for the
browser page gecko.html: everything the forward pass uses, as float32, in one JSON file.

  python3 export_gecko_web.py fca_gecko_web.pt ../../fluoddity-ca/gecko.json
  python3 export_gecko_web.py fca_gecko_web.pt out.json --ref 16,48,96,300 --ref_out ref.npz

Layout (the page's textures follow the state's own channel order, 4 per RGBA32F texture):
  state  16 channels: m, m*vx, m*vy, m*h0..h5, m*p0..p6   (p0..p2 = RGB, p3..p6 = paint-hidden)
  seed   that state at t=0 over the seed patch [s0, s0+S) x [s0, s0+S) (zero elsewhere); trail 0
  brain  freq [C, 25] over the inputs  L(2) R(2) Lm Rm m h(6) Lh(6) Rh(6),  amp [C, 4]
  hidden wh [6, C, 4], bh [6]    brake wb [C, 4], bb    paint w1 [W, 54], b1, w2 [7, W], b2
Arrays are base64 of little-endian float32, row-major, with their shape alongside.
"""
import argparse, base64, json, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fluoddity_ca_gecko as f

for p in (f.PRESETS_JS, os.path.join(HERE, "..", "src", "presets.js"), "/home/user/fluoddity-ca/src/presets.js"):
    if os.path.exists(p): f.PRESETS_JS = p; break
DEFAULTS = dict(press_mode=0, lr_brake=2e-2, paint=-1, paint_w=64, freeze_dyn=False, rgb=False, brake=False, brake_init=5.0,
                wpix=0.0, batch=2, max_age=600, age_ramp=200, colour=False, pool=0, wcol=0.5, hue_sens=None, hue_off=0.33,
                train_hue_off=1, seg_min=32, seg_max=64, score_every=4, eval_every=50, learn_seed_h=1, seed_lowres=0,
                rollback=0, ckpt_min=10, opt="grad")


def build(path):
    ck = torch.load(path, weights_only=False)
    a = argparse.Namespace(**ck["args"])
    for k, v in DEFAULTS.items():
        if not hasattr(a, k): setattr(a, k, v)
    pre = {p["name"]: p for p in f.load_presets()}
    m = f.FCA(a, pre[ck["preset"]]); m.load_state_dict(ck["state"]); m.eval()
    return m, a, ck


def b64(x):
    x = np.ascontiguousarray(np.asarray(x, dtype="<f4"))
    return {"shape": list(x.shape), "b64": base64.b64encode(x.tobytes()).decode()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--ref", default="", help="steps at which to save the torch state (comma list) for a parity check")
    ap.add_argument("--ref_out", default="")
    A = ap.parse_args()
    torch.set_num_threads(1)
    m, a, ck = build(A.ckpt)
    m.feed_total = float(f.load_target(a).sum())
    EL = getattr(m, "elastic", False)
    assert a.see_mass and m.Hd == 6 and m.Pc == (5 if EL else 7) and a.brake, "the page is written for see_mass, hidden 6, paint 4 (2 with --elastic), brake"
    assert getattr(a, "zoom", 1) == 1 and not a.colour and not a.rgb and a.press_mode == 0
    n = m.N
    tgt = f.load_target(a); total = float(tgt.sum())
    with torch.no_grad():
        S0, _ = m.init_state(total)
        Pt = m.P()
        P = {k: float(v) for k, v in Pt.items()}
        # the step's derived constants, computed in float32 exactly as the step computes them
        import math
        ang = Pt["angle"] * math.pi
        pc = Pt["persist"].clamp(0.001, 0.999); dd = Pt["diff"] ** 2
        derived = dict(cos_a=float(torch.cos(ang)), sin_a=float(torch.sin(ang)),
                       sd=float(1.0 / m.SQ * 0.005 * Pt["dist"]), scaling=float(m.SQ * 38.855 * Pt["gain"]),
                       fcoef=float(1.0 / m.SQ * Pt["gfm"] / 400.0), scoef=float(1.0 / m.SQ * Pt["gfm"] / 20.0),
                       pcoef=float(Pt["press"] * (2.0 / n)), K=float(4.0 / (torch.pow(5.0, dd) - 1.0)),
                       persist=float(Pt["persist"]), deposit=float(1.0 - pc), cells=n / 2.0)
    S0 = S0.reshape(-1, n, n).numpy()
    # the page's channel order: m, m*v, m*h0..5, m*paint (RGB first), then m*X, m*Y with --elastic
    order = list(range(9)) + (list(range(11, 16)) + [9, 10] if EL else list(range(9, 16)))
    S0 = S0[order]
    s0, s1 = m.s0, m.s1
    out_ = np.abs(S0).copy(); out_[:, s0:s1, s0:s1] = 0
    assert out_.max() == 0, "seed outside its patch"
    reps = getattr(a, "trail_reps", 0) or getattr(a, "zoom", 1) ** 2
    assert reps == 1
    from PIL import Image
    im = np.asarray(Image.open(os.path.join(HERE, "gecko.png")).convert("RGBA"))
    d = dict(
        kind="fluoddity-ca gecko", source=os.path.basename(A.ckpt), preset=ck["preset"], it=ck.get("it"), best=ck.get("best"),
        N=n, SQ=m.SQ, total_mass=total, nt=m.nt, sym=bool(m.sym), absori=int(m.absori), orimix=float(m.orimix),
        rest=float(a.rest), pressure_on=bool(a.train_pressure or a.pressure > 0),
        phys=P, derived=derived, mgain=float(m.mgain.detach()), hgain=float(m.hgain.detach()), bb=float(m.bb.detach()),
        seed_s0=s0, seed=b64(S0[:, s0:s1, s0:s1]),
        freq=b64(m.freq.detach()), amp=b64(m.amp.detach()), wh=b64(m.wh.detach()), bh=b64(m.bh.detach()),
        wb=b64(m.wb.detach()), pw1=b64(m.pw1.detach()), pb1=b64(m.pb1.detach()), pw2=b64(m.pw2.detach()), pb2=b64(m.pb2.detach()),
        target=dict(w=im.shape[1], h=im.shape[0], off=(n - im.shape[0]) // 2,
                    rgba=base64.b64encode(np.ascontiguousarray(im).tobytes()).decode()),
        channels=["m", "m*vx", "m*vy"] + [f"m*h{k}" for k in range(6)] + [f"m*p{k}" for k in range(7)],
        render="rgb = (m + m*p[0:3]) / 2, shown / max(target rgb premultiplied)",
    )
    # optional pieces of the newer models (each off = absent)
    if getattr(m, "nbr", 0): d["nbr"] = int(m.nbr)
    if getattr(m, "anchor", 0): d["anchor"] = int(m.anchor); d["wa"] = b64(m.wa.detach())
    if getattr(a, "pmod", False): d["wp"] = b64(m.wp.detach()); d["bp"] = float(m.bp.detach())
    if getattr(a, "feed", 0): d["feed"] = float(a.feed)
    if EL:
        d["el"] = dict(wr=b64(m.wr.detach()), br=float(m.br.detach()), wk=b64(m.wk.detach()), bk=float(m.bk.detach()), kmax=float(a.el_kmax),
                       t0=float(a.el_t0), tau=float(a.el_tau))
        d["channels"] = ["m", "m*vx", "m*vy"] + [f"m*h{k}" for k in range(6)] + [f"m*p{k}" for k in range(5)] + ["m*X", "m*Y"]
    if A.ref:
        steps = sorted(int(s) for s in A.ref.split(","))
        trgb = f.load_target_rgb(a)
        with torch.no_grad():
            out, _, _ = m.run(total, max(steps), keep=set(steps), ckpt=False, full=True)
        ref, losses = {}, {}
        for t in steps:
            S = out[t]; R = m.render(S)
            lm = float(f.ms_loss(S[0].reshape(n, n), tgt)); lc = float(f.ms_loss(R, trgb))
            losses[t] = (round(lm, 4), round(lc, 4))
            ref[f"S{t}"] = S.reshape(-1, n, n).numpy(); ref[f"R{t}"] = R.numpy()
            print(f"torch t={t}: mass loss {lm:.4f} colour loss {lc:.4f}", flush=True)
        d["torch_loss"] = {str(k): v for k, v in losses.items()}
        if A.ref_out: np.savez_compressed(A.ref_out, **ref, tgt=tgt.numpy(), trgb=trgb.numpy())
    json.dump(d, open(A.out, "w"), separators=(",", ":"))
    print("wrote", A.out, os.path.getsize(A.out), "bytes")


if __name__ == "__main__":
    main()
