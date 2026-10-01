"""Fluoddity as a CA, differentiable, fitted so that a seed grows the gecko.

The step is the page's (fluoddity-ca-next): every cell holds a mass, a momentum
and nothing else; it reads the trail with two eyes along its own velocity, runs
the ten-centre Fourier brain, accelerates, and pushes its mass to where it lands
as a box of side `spread` cells, split over the cells under it by overlap. The
trail blurs, fades and takes every cell's momentum. Units are the page's: world
spans [-1,1] across the grid, the grid is 1024*sqrt(ws) cells on a side, mass
is in particles. At 256 cells ws = 1/16, and every rate in CELLS per step is
the same as the page's at any world size, so a fit here is a fit there.

The target is gecko.png at its own resolution, one pixel one cell, in the
middle of a 256 grid. Mass is conserved exactly, so the seed holds exactly what
the picture costs: a disc of mass, and a velocity field over it that the fit
may change, because a cell at rest has no heading and the brain cannot steer it.

Variants, each a flag on top of the faithful one:
  --see_mass     the eyes also read mass, and the cell its own
  --centres N    more Fourier centres than ten
  --hidden H     H numbers carried with the mass, read by the eyes, rewritten
                 by the brain; their seed values are learned too

  python3 fluoddity_ca_gecko.py --tag faithful --iters 600
  python3 fluoddity_ca_gecko.py --tag hidden --see_mass --centres 48 --hidden 6
"""
import argparse, json, math, os, re, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
PRESETS_JS = "/tmp/claude-0/-home-user-field-life/eee7b4cc-fe9f-5306-965d-dbe5dfbf32a2/scratchpad/presets.js"
PHYS = ["axial_force", "lateral_force", "sensor_gain", "drag", "strafe_power", "sensor_angle",
        "global_force_mult", "sensor_distance", "trail_persistence", "trail_diffusion"]


# ---------------------------------------------------------------- presets, and the page's rule bake
def load_presets(path=PRESETS_JS):
    s = open(path).read()
    s = s[s.index("["): s.rindex("]") + 1]
    return json.loads(s)


def _pcg(x):
    x = np.uint64(x) & np.uint64(0xffffffff)
    st = (x * np.uint64(747796405) + np.uint64(2891336453)) & np.uint64(0xffffffff)
    w = (((st >> ((st >> np.uint64(28)) + np.uint64(4))) ^ st) * np.uint64(277803737)) & np.uint64(0xffffffff)
    return (w >> np.uint64(22)) ^ w


def _hash(a, b):
    ua = int(np.float32(a).view(np.uint32)); ub = int(np.float32(b).view(np.uint32))
    h = _pcg(np.uint64(ua) ^ _pcg(ub))
    return np.float32(np.float32(int(h)) / np.float32(4294967295.0))


def bake_rule(rule, seed, mut, cohort=0):
    """FS_RULES for one cohort: the preset's 80 numbers, mutated as the page mutates them."""
    f32 = np.float32
    r = np.asarray(rule, np.float32).reshape(10, 2, 4)
    fr, am = r[:, 0].copy(), r[:, 1].copy()
    sd = f32(f32(seed) + f32(cohort))
    if np.all(fr[0] == 0) and np.all(am[5] == 0):
        for i in range(10):
            fs = f32(1 + 2 * _hash(sd, i * 8) ** 2)
            fr[i] = [(_hash(sd, i * 8 + k) * 2 - 1) * fs for k in range(4)]
            am[i] = [_hash(sd, i * 8 + 4 + k) * 2 - 1 for k in range(4)]
    co = fr[4][:2] + am[7][[1, 0]] + fr[1][2:]
    s = f32(_hash(co[0], co[1]) + sd)
    mut = f32(mut)
    for i in range(10):
        cx, cy = f32(f32(-0.5) + f32(f32(-i) + s)), f32(f32(-0.5) + f32(i))
        h4 = [_hash(cx, cy), _hash(f32(cx * -1 + 5), f32(cy * -1 + 5)),
              _hash(f32(cy - 100), f32(cx - 100)), _hash(f32(cy * -1 + 25), f32(cx * -1 + 25))]
        am[i] = am[i] + mut * (-1 + 2 * np.array(h4, np.float32))
        fr[i] = fr[i] * (1 + mut * f32(0.5) * (_hash(s, i) - f32(0.5)))
    return fr, am


# ---------------------------------------------------------------- the step
def isp(y):  # inverse softplus
    return math.log(math.expm1(max(y, 1e-6)))


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4); return math.log(p / (1 - p))


class FCA(nn.Module):
    def __init__(self, a, preset):
        super().__init__()
        self.a = a
        self.N = a.grid
        self.ws = (a.grid / 1024.0) ** 2
        self.SQ = math.sqrt(self.ws)
        nC, Hd = a.centres, a.hidden
        # brain inputs: L(2) R(2) [Lm Rm m] [h(H) Lh(H) Rh(H)]
        self.din = 4 + (3 if a.see_mass else 0) + 3 * Hd
        fr, am = bake_rule(preset["rule"], preset["rule_seed"], preset["mutation_scale"] * a.mut_mult)
        freq = torch.zeros(nC, self.din); amp = torch.zeros(nC, 4)
        g = torch.Generator().manual_seed(a.seed)
        freq[:10, [0, 1, 2, 3]] = torch.tensor(fr)
        amp[:10] = torch.tensor(am)
        if nC > 10:   # extra centres: random frequencies like the page's own, quiet amplitudes
            fs = 1 + 2 * torch.rand(nC - 10, 1, generator=g) ** 2
            freq[10:, :4] = (torch.rand(nC - 10, 4, generator=g) * 2 - 1) * fs
            amp[10:] = (torch.rand(nC - 10, 4, generator=g) * 2 - 1) * a.extra_amp
            amp[10:, 3] = torch.rand(nC - 10, generator=g) * 2 - 1     # .w is also a phase
        if self.din > 4:  # extra inputs start with small random frequencies so they are heard at all
            freq[:, 4:] = torch.randn(nC, self.din - 4, generator=g) * a.extra_freq
        self.freq = nn.Parameter(freq)
        self.amp = nn.Parameter(amp)
        if Hd:
            self.wh = nn.Parameter(torch.randn(Hd, nC, 4, generator=g) * 0.3 / math.sqrt(nC))
            self.bh = nn.Parameter(torch.zeros(Hd))
            self.hgain = nn.Parameter(torch.tensor(1.0))
        if a.see_mass:
            self.mgain = nn.Parameter(torch.tensor(1.0))
        p = {k: preset[k] for k in PHYS}
        self.raw = nn.ParameterDict({
            "axial": nn.Parameter(torch.tensor(float(p["axial_force"]))),
            "lateral": nn.Parameter(torch.tensor(float(p["lateral_force"]))),
            "gain": nn.Parameter(torch.tensor(float(p["sensor_gain"]))),
            "drag": nn.Parameter(torch.tensor(float(p["drag"]))),
            "strafe": nn.Parameter(torch.tensor(float(p["strafe_power"]))),
            "angle": nn.Parameter(torch.tensor(float(p["sensor_angle"]))),
            "gfm": nn.Parameter(torch.tensor(float(p["global_force_mult"]))),
            "dist": nn.Parameter(torch.tensor(float(p["sensor_distance"]))),
            "persist": nn.Parameter(torch.tensor(logit(min(p["trail_persistence"], 0.999) / 0.999))),
            "diff": nn.Parameter(torch.tensor(logit((max(p["trail_diffusion"], 0.001) - 0.001) / 0.999))),
            "press": nn.Parameter(torch.tensor(isp(a.pressure))),
            "spread": nn.Parameter(torch.tensor(isp(a.spread - 1.0))),
        })
        if not a.train_spread: self.raw["spread"].requires_grad_(False)
        if not a.train_pressure: self.raw["press"].requires_grad_(False)
        self.sym = not preset.get("disable_symmetry", False)
        self.absori = a.absori if a.absori >= 0 else int(preset.get("absolute_orientation", 0))
        self.orimix = float(preset.get("orientation_mix", 1.0))
        # the seed: a disc of mass, with a learnable velocity field (and hidden values) over it
        n = self.N; c = (n - 1) / 2.0
        yy, xx = torch.meshgrid(torch.arange(n, dtype=torch.float32), torch.arange(n, dtype=torch.float32), indexing="ij")
        self.register_buffer("yy", yy); self.register_buffer("xx", xx)
        r = torch.sqrt((xx - c) ** 2 + (yy - c) ** 2)
        disc = (a.seed_r + 0.5 - r).clamp(0, 1)
        self.register_buffer("disc", disc)
        self.register_buffer("pos", torch.stack([(xx + 0.5) / n * 2 - 1, (yy + 0.5) / n * 2 - 1]))
        self.register_buffer("xf", xx.reshape(-1).clone()); self.register_buffer("yf", yy.reshape(-1).clone())
        self.nt = int(math.ceil(a.max_spread)) + 1
        R0 = int(math.ceil(a.seed_r)) + 1
        self.s0, self.s1 = int(c - R0 + 0.5), int(c + R0 + 1.5)
        S = self.s1 - self.s0
        sv = torch.zeros(2, S, S)
        sx = xx[self.s0:self.s1, self.s0:self.s1] - c; sy = yy[self.s0:self.s1, self.s0:self.s1] - c
        rr = torch.sqrt(sx ** 2 + sy ** 2).clamp_min(1e-3)
        if a.seed_vel == "radial":
            sv = torch.stack([sx / rr, sy / rr]) * a.seed_speed
        elif a.seed_vel == "rand":
            ang = torch.rand(S, S, generator=g) * 2 * math.pi
            sv = torch.stack([ang.cos(), ang.sin()]) * a.seed_speed
        else:
            sv = sv + torch.randn(2, S, S, generator=g) * 1e-5
        self.seed_v = nn.Parameter(sv)
        if not a.learn_seed_v: self.seed_v.requires_grad_(False)
        if Hd:
            self.seed_h = nn.Parameter(torch.randn(Hd, S, S, generator=g) * 0.3)

    # ---- params in the page's units
    def P(self):
        r = self.raw
        return dict(axial=r["axial"], lateral=r["lateral"], gain=r["gain"], drag=r["drag"], strafe=r["strafe"],
                    angle=r["angle"], gfm=r["gfm"], dist=r["dist"],
                    persist=torch.sigmoid(r["persist"]) * 0.999,
                    diff=0.001 + 0.999 * torch.sigmoid(r["diff"]),
                    press=F.softplus(r["press"]), spread=1.0 + F.softplus(r["spread"]))

    def init_state(self, total_mass):
        n = self.N
        m = self.disc * (total_mass / self.disc.sum())
        v = torch.zeros(2, n, n)
        v = F.pad(self.seed_v, (self.s0, n - self.s1, self.s0, n - self.s1))
        st = [m, m * v[0], m * v[1]]
        if self.a.hidden:
            h = torch.tanh(F.pad(self.seed_h, (self.s0, n - self.s1, self.s0, n - self.s1)))
            st += [m * h[k] for k in range(self.a.hidden)]
        return torch.stack(st).reshape(len(st), -1), torch.zeros(2, n * n)

    def sample(self, Fm, px, py):
        """bilinear taps of Fm[C, n*n] at absolute cell coords (px, py), wrapped: the eye."""
        n = self.N
        x0, y0 = torch.floor(px), torch.floor(py)
        fx, fy = px - x0, py - y0
        x0 = x0.long() % n; y0 = y0.long() % n; x1 = (x0 + 1) % n; y1 = (y0 + 1) % n
        idx = torch.cat([y0 * n + x0, y0 * n + x1, y1 * n + x0, y1 * n + x1])
        g = Fm[:, idx].reshape(Fm.shape[0], 4, -1)
        w = torch.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy])
        return (g * w).sum(1)

    def fourier_feats(self, pos):
        """pos [din, A] -> features [nC, 4, A], the page's four waves per centre."""
        ph = self.freq @ pos
        i = torch.arange(self.freq.shape[0], dtype=torch.float32)
        off = (2 * i * 0.6283 + self.amp[:, 3] * 3.14159)[:, None]
        return torch.stack([torch.sin(ph + off), torch.cos(ph + off * 0.7),
                            torch.sin(ph * 2 + off * 1.3), torch.cos(ph * 2 + off * 0.5)], 1)

    def step(self, S, T):
        """S [3+H, n*n] = mass, momentum, mass*hidden; T [2, n*n] = trail. Only cells holding
        mass run the brain and move (the page's m <= 1e-12 early-out)."""
        a, P, n, Hd = self.a, self.P(), self.N, self.a.hidden
        m = S[0]
        ia = torch.nonzero(m > 1e-12).squeeze(1)
        ma = m[ia]
        Sa = S[:, ia]
        xa, ya = self.xf[ia], self.yf[ia]
        vel = Sa[1:3] / ma
        sd = 1.0 / self.SQ * 0.005 * P["dist"]
        vn = torch.sqrt((vel ** 2).sum(0) + 1e-30)
        ori = vel / vn * (vn > 1e-12)
        if a.ori_eps > 0:   # exact forward; the backward sees a heading softened below |v| ~ ori_eps
            soft = vel / torch.sqrt((vel ** 2).sum(0) + a.ori_eps ** 2)
            ori = soft + (ori - soft).detach()
        if self.absori == 1:
            up = torch.zeros_like(ori); up[1] = 1
            ori = ori + (up - ori) * self.orimix
        elif self.absori == 2:
            pos = torch.stack([(xa + 0.5) / n * 2 - 1, (ya + 0.5) / n * 2 - 1])
            pn = pos / torch.sqrt((pos ** 2).sum(0) + 1e-12)
            ori = ori + (-pn - ori) * self.orimix
        on = torch.sqrt((ori ** 2).sum(0) + 1e-30)
        fwd = ori / on * (on > 1e-12)
        left = torch.stack([fwd[1], -fwd[0]])
        ang = P["angle"] * math.pi
        ca, sa = torch.cos(ang), torch.sin(ang)
        o = ori * sd
        oq = torch.stack([o[1], -o[0]])
        lo = ca * o + sa * oq      # pR(lo, +angle)
        ro = ca * o - sa * oq      # pR(ro, -angle)
        cells = n / 2.0
        fields = [T]
        if a.see_mass: fields.append(m[None])
        if Hd: fields.append(S[3:])
        Fm = torch.cat(fields) if len(fields) > 1 else T
        EL = self.sample(Fm, xa + lo[0] * cells, ya + lo[1] * cells)
        ER = self.sample(Fm, xa + ro[0] * cells, ya + ro[1] * cells)
        scaling = self.SQ * 38.855 * P["gain"]
        TL, TR = EL[:2] * scaling, ER[:2] * scaling
        L = torch.stack([(TL * fwd).sum(0), (TL * left).sum(0)])
        R = torch.stack([(TR * fwd).sum(0), (TR * left).sum(0)])
        flip = lambda u: torch.stack([u[0], -u[1]])
        ins_b, ins_m = [L, R], [flip(R), flip(L)]
        k = 2
        if a.see_mass:
            Lm, Rm = EL[k:k + 1] * self.mgain, ER[k:k + 1] * self.mgain
            mo = ma[None] * self.mgain
            ins_b += [Lm, Rm, mo]; ins_m += [Rm, Lm, mo]; k += 1
        if Hd:
            h = Sa[3:] / ma
            Lh, Rh = EL[k:] * self.hgain, ER[k:] * self.hgain
            ins_b += [h, Lh, Rh]; ins_m += [h, Rh, Lh]
        fb = self.fourier_feats(torch.cat(ins_b))
        base = (self.amp[:, :, None] * fb).sum(0)
        if self.sym:
            fm = self.fourier_feats(torch.cat(ins_m))
            mir = (self.amp[:, :, None] * fm).sum(0)
        else:
            fm, mir = None, torch.zeros_like(base)
        f0, f1 = base[0] + mir[0], base[1] - mir[1]
        s0, s1 = base[2] + mir[2], base[3] - mir[3]
        force = (fwd * (f0 * P["axial"]) + left * (f1 * P["lateral"])) * (1.0 / self.SQ * P["gfm"] / 400.0)
        strafe = (fwd * (s0 * P["axial"]) + left * (s1 * P["lateral"])) * (1.0 / self.SQ * P["gfm"] / 20.0)
        if a.train_pressure or a.pressure > 0:
            Pm = F.relu(m - a.rest).reshape(n, n)
            gx = torch.zeros_like(Pm); gy = torch.zeros_like(Pm)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0: continue
                    w = 2.0 if (dx == 0 or dy == 0) else 1.0
                    sh = torch.roll(Pm, (-dy, -dx), (0, 1))
                    gx = gx + w * sh * dx; gy = gy + w * sh * dy
            g = torch.stack([gx.reshape(-1)[ia], gy.reshape(-1)[ia]])
            force = force - P["press"] * (2.0 / n) * g / 8.0
        vel = vel * P["drag"] + force
        disp = (vel + strafe * P["strafe"]) * cells
        Q = [ma, ma * vel[0], ma * vel[1]]
        if Hd:
            feats = fb + fm if fm is not None else fb
            dh = torch.einsum("kcf,cfa->ka", self.wh, feats) + self.bh[:, None]
            hn = torch.tanh(h + dh)
            Q = Q + [ma * hn[i] for i in range(Hd)]
        S2 = self.move(torch.stack(Q), xa + disp[0], ya + disp[1], P["spread"], S.shape[1])
        # trail: blur, fade, take the momentum
        p = P["persist"]; d = P["diff"] ** 2
        K = 4.0 / (torch.pow(5.0, d) - 1.0)
        T2d = T.reshape(2, n, n)
        nb = sum(torch.roll(T2d, s, ax) for s in (1, -1) for ax in (1, 2)).reshape(2, -1)
        T2 = (T * K + nb) / (4.0 + K) * p + S2[1:3] * (1.0 - p.clamp(0.001, 0.999)) * 7.07
        return S2, T2

    def move(self, Q, qx, qy, s, size):
        """reintegration tracking: source j's Q[:, j] is a box of side s centred at (qx, qy) in
        cells; every cell gets the fraction of the box over it. nt x nt cells can be touched."""
        n, nt = self.N, self.nt
        s = s.clamp(max=nt - 1.0)
        h = s / 2
        def taps(q):
            c0 = torch.floor(q - h + 0.5).detach()
            c = c0[None] + torch.arange(nt, dtype=torch.float32)[:, None]
            w = (torch.minimum(q + h, c + 0.5) - torch.maximum(q - h, c - 0.5)).clamp_min(0) / s
            return c.long() % n, w
        cx, wx = taps(qx); cy, wy = taps(qy)
        tgt = (cy[:, None] * n + cx[None]).reshape(-1)
        w = (wy[:, None] * wx[None]).reshape(-1)
        vals = (Q[:, None] * w.reshape(nt * nt, -1)[None]).reshape(Q.shape[0], -1)
        return torch.zeros(Q.shape[0], size).index_add(1, tgt, vals)

    def run(self, total_mass, steps, keep=(), ckpt=True):
        S, T = self.init_state(total_mass)
        out = {}
        for t in range(1, steps + 1):
            if ckpt and torch.is_grad_enabled():
                S, T = checkpoint(self.step, S, T, use_reentrant=False)
            else:
                S, T = self.step(S, T)
            if t in keep: out[t] = S[0].reshape(self.N, self.N)
        return out, S, T


# ---------------------------------------------------------------- target, loss, pictures
def load_target(a):
    from PIL import Image
    im = np.asarray(Image.open(os.path.join(HERE, "gecko.png")).convert("RGBA")).astype(np.float32) / 255
    al = im[..., 3]
    if a.luma:
        al = al * (0.3 * im[..., 0] + 0.59 * im[..., 1] + 0.11 * im[..., 2])
    al = al[::-1].copy()          # row 0 at the bottom, as the page's y
    n, h = a.grid, al.shape[0]
    t = np.zeros((n, n), np.float32)
    o = (n - h) // 2
    t[o:o + h, o:o + al.shape[1]] = al * a.mass_scale
    return torch.tensor(t)


def ms_loss(m, tgt, scales=(1, 2, 4, 8)):
    L = 0
    for s in scales:
        a = F.avg_pool2d(m[None, None], s) if s > 1 else m[None, None]
        b = F.avg_pool2d(tgt[None, None], s) if s > 1 else tgt[None, None]
        L = L + ((a - b) ** 2).mean() / (b ** 2).mean()
    return L / len(scales)


def to_img(m, vmax):
    x = (m.detach().numpy()[::-1] / vmax).clip(0, 1)
    # a warm ramp: black -> amber -> white
    r = np.clip(x * 1.6, 0, 1); g = np.clip(x * 1.25 - 0.1, 0, 1); b = np.clip(x * 1.8 - 0.8, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def save_pictures(model, tgt, total, a, tag):
    from PIL import Image, ImageDraw
    n = a.grid; c0 = (n - 160) // 2
    crop = lambda img: img[c0:c0 + 160, c0:c0 + 160]
    T1 = a.t1
    steps = sorted(set([0, 8, 16, 24, 32, 48, a.t0, T1, T1 + 16, T1 + 48, 2 * T1]))
    with torch.no_grad():
        S0, _ = model.init_state(total)
        S0 = S0.reshape(-1, a.grid, a.grid)
        out, _, _ = model.run(total, max(steps), keep=set(steps), ckpt=False)
    out[0] = S0[0]
    vmax = float(tgt.max()) * 1.2
    tiles = [("target", crop(to_img(tgt, vmax)))] + [(f"t={s}", crop(to_img(out[s], vmax))) for s in steps]
    W = 160 * len(tiles)
    sheet = Image.new("RGB", (W, 178), (10, 12, 14)); d = ImageDraw.Draw(sheet)
    for i, (lab, im) in enumerate(tiles):
        sheet.paste(Image.fromarray(im), (160 * i, 18)); d.text((160 * i + 4, 3), lab, fill=(220, 220, 220))
    p1 = os.path.join(HERE, f"fca_gecko_{tag}_strip.png"); sheet.save(p1)
    big = Image.new("RGB", (512 * 2 + 8, 530), (10, 12, 14)); d = ImageDraw.Draw(big)
    big.paste(Image.fromarray(crop(to_img(tgt, vmax))).resize((512, 512), Image.NEAREST), (0, 18))
    big.paste(Image.fromarray(crop(to_img(out[T1], vmax))).resize((512, 512), Image.NEAREST), (520, 18))
    d.text((4, 3), "target", fill=(220, 220, 220)); d.text((524, 3), f"{tag}: grown, t={T1}", fill=(220, 220, 220))
    p2 = os.path.join(HERE, f"fca_gecko_{tag}_vs.png"); big.save(p2)
    return p1, p2


def save_params(model, path, extra):
    P = {k: float(v) for k, v in model.P().items()}
    d = dict(extra, phys=P, freq=model.freq.detach().tolist(), amp=model.amp.detach().tolist(),
             seed_v=model.seed_v.detach().tolist(), sym=model.sym, absori=model.absori, orimix=model.orimix)
    if model.a.hidden:
        d.update(wh=model.wh.detach().tolist(), bh=model.bh.detach().tolist(), hgain=float(model.hgain),
                 seed_h=model.seed_h.detach().tolist())
    if model.a.see_mass: d["mgain"] = float(model.mgain)
    # the page's 80 numbers, when the brain is the page's brain
    if model.freq.shape == (10, 4):
        d["rule80"] = torch.cat([model.freq, model.amp], 1).reshape(-1).tolist()
    json.dump(d, open(path, "w"))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="faithful")
    ap.add_argument("--grid", type=int, default=256)
    ap.add_argument("--preset", default="auto", help="name, comma list, or auto (screen them all)")
    ap.add_argument("--screen_top", type=int, default=3)
    ap.add_argument("--screen_iters", type=int, default=40)
    ap.add_argument("--mut_mult", type=float, default=1.0)
    ap.add_argument("--absori", type=int, default=-1, help="-1 = the preset's")
    ap.add_argument("--centres", type=int, default=10)
    ap.add_argument("--see_mass", action="store_true")
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--extra_amp", type=float, default=0.05)
    ap.add_argument("--extra_freq", type=float, default=0.3)
    ap.add_argument("--spread", type=float, default=1.5)
    ap.add_argument("--train_spread", type=int, default=1)
    ap.add_argument("--pressure", type=float, default=0.3)
    ap.add_argument("--train_pressure", type=int, default=1)
    ap.add_argument("--rest", type=float, default=1.0, help="mass above which pressure pushes")
    ap.add_argument("--max_spread", type=float, default=3.0, help="spread is capped here (sets the taps per axis)")
    ap.add_argument("--seed_r", type=float, default=12.0)
    ap.add_argument("--seed_vel", default="radial", choices=["radial", "rand", "zero"])
    ap.add_argument("--seed_speed", type=float, default=0.002)
    ap.add_argument("--learn_seed_v", type=int, default=1)
    ap.add_argument("--ori_eps", type=float, default=1e-3, help="world units; 0 = exact gradient")
    ap.add_argument("--luma", action="store_true")
    ap.add_argument("--mass_scale", type=float, default=1.0)
    ap.add_argument("--t0", type=int, default=48)
    ap.add_argument("--t1", type=int, default=80)
    ap.add_argument("--hold", type=int, default=16, help="also score this many steps past the end")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr_seed", type=float, default=1e-2)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--hours", type=float, default=0, help="stop after this long (0 = iters only)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    tgt = load_target(a)
    total = float(tgt.sum())
    presets = {p["name"]: p for p in load_presets()}
    log = open(os.path.join(HERE, f"fca_gecko_{a.tag}.log"), "a")
    def say(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()
    say(f"== {a.tag} {vars(a)}")

    def score(model, iters_eval=None):
        with torch.no_grad():
            out, _, _ = model.run(total, a.t1, keep={a.t0, (a.t0 + a.t1) // 2, a.t1}, ckpt=False)
            vals = [float(ms_loss(m, tgt)) for m in out.values()]
            bad = any(not math.isfinite(v) for v in vals)
            return 1e9 if bad else sum(vals) / len(vals)

    if a.resume:
        ck = torch.load(a.resume)
        model = FCA(a, presets[ck["preset"]]); model.load_state_dict(ck["state"]); start = ck["preset"]
    else:
        names = list(presets) if a.preset == "auto" else a.preset.split(",")
        if len(names) > 1:
            res = []
            for nm in names:
                t = time.time(); mdl = FCA(a, presets[nm]); sc = score(mdl)
                res.append((sc, nm)); say(f"screen {nm:22s} {sc:.4f}  {time.time() - t:.1f}s")
            res.sort()
            names = [nm for _, nm in res[:a.screen_top]]
            say("screen best:", res[:a.screen_top])
            if a.screen_top > 1 and a.screen_iters > 0:   # a short fit of each, keep the best
                res2 = []
                for nm in names:
                    mdl = FCA(a, presets[nm]); l = train(mdl, a, tgt, total, a.screen_iters, say, nm, quiet=True)
                    res2.append((score(mdl), nm)); say(f"short fit {nm}: {res2[-1][0]:.4f}")
                res2.sort(); names = [res2[0][1]]
        start = names[0]
        model = FCA(a, presets[start])
    say(f"start from preset {start}; params {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    train(model, a, tgt, total, a.iters, say, start)


def train(model, a, tgt, total, iters, say, name, quiet=False):
    seedp = [p for n, p in model.named_parameters() if n.startswith("seed_") and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith("seed_") and p.requires_grad]
    opt = torch.optim.Adam([{"params": rest, "lr": a.lr}, {"params": seedp, "lr": a.lr_seed * a.seed_speed}])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(i / max(iters, 1), 1)))))
    best, best_state = 1e9, None
    t_start = time.time(); hist = []
    for it in range(iters):
        t = time.time()
        tend = int(torch.randint(a.t0, a.t1 + 1, (1,)))
        keep = set(range(a.t0, tend + 1, 8)) | {tend, tend + a.hold} if a.hold else set(range(a.t0, tend + 1, 8)) | {tend}
        out, S, T = model.run(total, max(keep), keep=keep)
        losses = [ms_loss(m, tgt) for m in out.values()]
        loss = sum(losses) / len(losses)
        if not torch.isfinite(loss):
            say(f"it {it}: non-finite loss, restoring best");
            if best_state: model.load_state_dict(best_state)
            continue
        opt.zero_grad(); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], a.clip))
        if math.isfinite(gn): opt.step()
        sched.step()
        L = float(loss); hist.append(L)
        ema = sum(hist[-20:]) / len(hist[-20:])
        if ema < best and it > 10:
            best = ema; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        mass_err = float(S[0].sum()) / total - 1
        if not quiet and (it % 10 == 0 or it == iters - 1):
            P = {k: round(float(v), 4) for k, v in model.P().items()}
            say(f"it {it:5d} loss {L:.4f} ema {ema:.4f} best {best:.4f} T={tend} gn {gn:.2e} "
                f"dmass {mass_err:+.1e} {time.time() - t:.2f}s/it  {P}")
        if not quiet and (it % 100 == 0 or it == iters - 1) and best_state is not None:
            cur = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            torch.save({"preset": name, "state": best_state, "args": vars(a), "best": best},
                       os.path.join(HERE, f"fca_gecko_{a.tag}.pt"))
            save_params(model, os.path.join(HERE, f"fca_gecko_{a.tag}.json"), {"preset": name, "best": best, "iters": it + 1, "args": vars(a)})
            save_pictures(model, tgt, total, a, a.tag)
            model.load_state_dict(cur)
        if a.hours and time.time() - t_start > a.hours * 3600 and not quiet:
            say(f"time limit at it {it}"); iters = it + 1; break
    if best_state is not None:
        model.load_state_dict(best_state)
        if not quiet:
            torch.save({"preset": name, "state": best_state, "args": vars(a), "best": best},
                       os.path.join(HERE, f"fca_gecko_{a.tag}.pt"))
            save_params(model, os.path.join(HERE, f"fca_gecko_{a.tag}.json"), {"preset": name, "best": best, "iters": iters, "args": vars(a)})
            say("pictures:", *save_pictures(model, tgt, total, a, a.tag))
    return best


if __name__ == "__main__":
    main()
