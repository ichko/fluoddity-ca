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
            hv = _hash(sd, i * 8)   # GLSL pow(x, 2.0) is exp2(2*log2(x)), and its rounding reaches the mutation hash
            fs = f32(f32(1) + f32(2) * np.exp2(f32(2) * np.log2(hv, dtype=np.float32), dtype=np.float32))
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
class _Damp(torch.autograd.Function):
    """identity forward; the gradient flowing back through one step is scaled by g. The
    step is chaotic (backward growth ~1.5x a step), so an undamped unroll is all noise."""
    @staticmethod
    def forward(ctx, x, g):
        ctx.g = g; return x.view_as(x)
    @staticmethod
    def backward(ctx, gr):
        return gr * ctx.g, None


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
        self.Hd = a.hidden + (3 if getattr(a, "rgb", False) else 0)   # --rgb: three more carried channels, read as colour
        nC, Hd = a.centres, self.Hd
        # brain inputs: L(2) R(2) [Lm Rm m] [h(H) Lh(H) Rh(H)]
        self.din = 4 + (3 if a.see_mass else 0) + 3 * Hd
        self.nbr = getattr(a, "nbr", 0)   # --nbr R: the (2R+1)^2 box mean of mass and of hidden around the cell, read isotropically
        if self.nbr: self.din += 1 + Hd
        self.anchor = getattr(a, "anchor", 0)   # --anchor K: the cell's position in the world as two more inputs, K more centres that hear only it
        if self.anchor: self.din += 2
        self.elastic = bool(getattr(a, "elastic", False))   # --elastic: every cell carries a rest position X; a spring pulls it back there
        if self.elastic: self.din += 2
        self.ex = 3 + Hd + (1 if a.colour else 0)            # index of m*X, m*Y in the state
        if self.anchor: nC = nC + self.anchor   # --centres stays the count of the ordinary ones
        fr, am = bake_rule(preset["rule"], preset["rule_seed"], preset["mutation_scale"] * a.mut_mult)
        freq = torch.zeros(nC, self.din); amp = torch.zeros(nC, 4)
        g = torch.Generator().manual_seed(a.seed)
        freq[:10, [0, 1, 2, 3]] = torch.tensor(fr)
        amp[:10] = torch.tensor(am)
        if nC > 10:   # extra centres: random frequencies like the page's own, quiet amplitudes
            fs = 1 + 2 * torch.rand(nC - 10, 1, generator=g) ** 2
            freq[10:, :4] = (torch.rand(nC - 10, 4, generator=g) * 2 - 1) * fs
            amp[10:] = (torch.rand(nC - 10, 4, generator=g) * 2 - 1) * a.extra_amp
            # .w is also the strafe amplitude, so it starts as quiet as the rest: a warm start
            # from ten centres then begins where it left off
        if self.din > 4:  # extra inputs start with small random frequencies so they are heard at all
            freq[:, 4:] = torch.randn(nC, self.din - 4, generator=g) * a.extra_freq
        if self.anchor:   # the anchor centres: pure waves over the position, |k| log-uniform in [2, 60], silent at the start
            K = self.anchor; c0 = nC - K
            freq[:, -2:] = 0; freq[c0:] = 0
            th = torch.rand(K, generator=g) * 2 * math.pi
            km = torch.exp(math.log(2) + torch.rand(K, generator=g) * (math.log(60) - math.log(2)))
            freq[c0:, -2] = km * th.cos(); freq[c0:, -1] = km * th.sin()
            amp[c0:] = 0
        self.freq = nn.Parameter(freq)
        self.amp = nn.Parameter(amp)
        if Hd:
            self.wh = nn.Parameter(torch.randn(Hd, nC, 4, generator=g) * 0.3 / math.sqrt(nC))
            self.bh = nn.Parameter(torch.zeros(Hd))
            self.hgain = nn.Parameter(torch.tensor(1.0))
        if a.see_mass:
            self.mgain = nn.Parameter(torch.tensor(1.0))
        if getattr(a, "brake", False):   # per-cell gate on the velocity, from the brain's features
            self.wb = nn.Parameter(torch.zeros(nC, 4))
            self.bb = nn.Parameter(torch.tensor(float(a.brake_init)))
        if self.anchor:   # a force in the world's frame (cells/step^2), from the features: zero at the start
            self.wa = nn.Parameter(torch.zeros(2, nC, 4))
            with torch.no_grad():
                if Hd: self.wh[:, nC - self.anchor:] = 0
        if self.elastic:   # rest-position memory: X <- X + r (x - X), spring -k (x - X); r, k from the features
            self.wr = nn.Parameter(torch.zeros(nC, 4)); self.br = nn.Parameter(torch.tensor(float(a.el_r0)))
            self.wk = nn.Parameter(torch.zeros(nC, 4)); self.bk = nn.Parameter(torch.tensor(logit(a.el_k0 / a.el_kmax)))
        if getattr(a, "pmod", False):   # per-cell gain on the pressure push, exp(wp.feats + bp), 1 at the start
            self.wp = nn.Parameter(torch.zeros(nC, 4))
            self.bp = nn.Parameter(torch.tensor(0.0))
        # --paint K: a colour layer riding on the mass, 3 RGB + K paint-hidden channels, carried
        # mass-weighted like the hidden ones and rewritten every step by its own small head. It reads
        # the brain's inputs, its own paint, the paint under the two eyes and the 3x3 mean paint, but
        # the dynamics never read it and its gradient never reaches the dynamics.
        self.Pc = (3 + a.paint) if getattr(a, "paint", -1) >= 0 else 0
        assert not (self.Pc and a.colour), "--paint and --colour both use the last channels"
        if self.Pc:
            dp, W = self.din + 4 * self.Pc + 1, a.paint_w
            self.pw1 = nn.Parameter(torch.randn(W, dp, generator=g) / math.sqrt(dp))
            self.pb1 = nn.Parameter(torch.zeros(W))
            self.pw2 = nn.Parameter(torch.zeros(self.Pc, W))
            self.pb2 = nn.Parameter(torch.zeros(self.Pc))
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
        if a.colour:
            self.hue_sens = nn.Parameter(torch.tensor(float(a.hue_sens if a.hue_sens is not None else preset.get("hue_sensitivity", 0.0))))
            self.hue_off = nn.Parameter(torch.tensor(float(a.hue_off)))   # NOT on the page: hue = sens*col + off
            if not a.train_hue_off: self.hue_off.requires_grad_(False)
            self.cbright = nn.Parameter(torch.tensor(0.7))                # the page's brightness slider
        self.sym = not preset.get("disable_symmetry", False)
        self.absori = a.absori if a.absori >= 0 else int(preset.get("absolute_orientation", 0))
        self.orimix = float(preset.get("orientation_mix", 1.0))
        # the seed: a disc of mass, with a learnable velocity field (and hidden values) over it
        n = self.N; c = (n - 1) / 2.0
        yy, xx = torch.meshgrid(torch.arange(n, dtype=torch.float32), torch.arange(n, dtype=torch.float32), indexing="ij")
        self.register_buffer("yy", yy.contiguous(), persistent=False); self.register_buffer("xx", xx.contiguous(), persistent=False)
        r = torch.sqrt((xx - c) ** 2 + (yy - c) ** 2)
        disc = (a.seed_r + 0.5 - r).clamp(0, 1)
        self.register_buffer("disc", disc, persistent=False)
        self.register_buffer("xf", xx.reshape(-1).clone(), persistent=False); self.register_buffer("yf", yy.reshape(-1).clone(), persistent=False)
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
        if not a.learn_seed_v or a.seed_lowres: self.seed_v.requires_grad_(False)
        if self.Pc:
            self.pseed = nn.Parameter(torch.zeros(self.Pc, S, S))   # the seed's paint (pre-tanh)
        if a.seed_lowres:   # a seed hint of only 2*K*K numbers: a coarse velocity correction, upsampled
            self.seed_c = nn.Parameter(torch.zeros(2, a.seed_lowres, a.seed_lowres))
        if Hd:
            self.seed_h = nn.Parameter(torch.randn(Hd, S, S, generator=g) * 0.3)
            if not a.learn_seed_h:   # hidden values start at zero everywhere: no layout hidden in the seed
                with torch.no_grad(): self.seed_h.zero_()
                self.seed_h.requires_grad_(False)

    def seed_vel(self):
        if not self.a.seed_lowres: return self.seed_v
        S = self.seed_v.shape[-1]
        return self.seed_v + F.interpolate(self.seed_c[None], size=(S, S), mode="bilinear", align_corners=True)[0]

    # ---- params in the page's units
    def P(self):
        r = self.raw
        return dict(axial=r["axial"], lateral=r["lateral"], gain=r["gain"], drag=r["drag"], strafe=r["strafe"],
                    angle=r["angle"], gfm=r["gfm"], dist=r["dist"],
                    persist=torch.sigmoid(r["persist"]) * 0.999,
                    diff=0.001 + 0.999 * torch.sigmoid(r["diff"]),
                    press=F.softplus(r["press"]), spread=(1.0 + F.softplus(r["spread"])).clamp(max=self.nt - 1.0))

    def init_state(self, total_mass):
        n = self.N
        self.t_now = 0
        m = self.disc * (total_mass / self.disc.sum())
        v = torch.zeros(2, n, n)
        v = F.pad(self.seed_vel(), (self.s0, n - self.s1, self.s0, n - self.s1))
        st = [m, m * v[0], m * v[1]]
        if self.Hd:
            h = torch.tanh(F.pad(self.seed_h, (self.s0, n - self.s1, self.s0, n - self.s1)))
            st += [m * h[k] for k in range(self.Hd)]
        if self.a.colour:
            st += [torch.zeros_like(m)]   # the page's col, mass-weighted; nothing has a colour yet
        if self.elastic:
            st += [m * self.xx.reshape(n, n), m * self.yy.reshape(n, n)]   # rest position: where it is, in cells
        if self.Pc:
            pp = torch.tanh(F.pad(self.pseed, (self.s0, n - self.s1, self.s0, n - self.s1)))
            st += [m * pp[k] for k in range(self.Pc)]
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

    def step(self, S, T, t=None):
        """S [3+H, n*n] = mass, momentum, mass*hidden; T [2, n*n] = trail. Only cells holding
        mass run the brain and move (the page's m <= 1e-12 early-out)."""
        a, P, n, Hd = self.a, self.P(), self.N, self.Hd
        if t is None:   # called step by step: the model keeps the count (init_state sets it to 0)
            t = getattr(self, "t_now", 0); self.t_now = t + 1
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
        if Hd: fields.append(S[3:3 + Hd])
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
            h = Sa[3:3 + Hd] / ma
            Lh, Rh = EL[k:] * self.hgain, ER[k:] * self.hgain
            ins_b += [h, Lh, Rh]; ins_m += [h, Rh, Lh]
        if self.nbr:   # isotropic: the same for the mirrored world
            r_ = self.nbr; kk = 2 * r_ + 1
            X = torch.cat([m[None], S[3:3 + Hd]]).reshape(1 + Hd, n, n)
            B = F.avg_pool2d(F.pad(X[None], (r_,) * 4, mode="circular"), kk, 1)[0].reshape(1 + Hd, -1)[:, ia]
            bmn = B[0:1]
            nb = torch.cat([bmn * self.mgain, B[1:] / bmn.clamp_min(1e-6) * self.hgain])
            ins_b += [nb]; ins_m += [nb]
        if self.anchor:
            pos = torch.stack([(xa + 0.5) / n * 2 - 1, (ya + 0.5) / n * 2 - 1])
            ins_b += [pos]; ins_m += [pos]
        if self.elastic:   # how far the cell is from its rest position, in its own frame, in units of 8 cells
            Xr = Sa[self.ex:self.ex + 2] / ma
            dd = torch.stack([xa, ya]) - Xr
            dd = torch.remainder(dd + n / 2, n) - n / 2
            dl = torch.stack([(dd * fwd).sum(0), (dd * left).sum(0)]) / 8.0
            ins_b += [dl]; ins_m += [flip(dl)]
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
            free = getattr(a, "press_mode", 0) == 1
            if free:   # free surface: only differences with neighbours inside the body push (none into the vacuum)
                bmm = F.avg_pool2d(F.pad(m.reshape(1, 1, n, n), (1, 1, 1, 1), mode="circular"), 3, 1)[0, 0]
                occ = ((bmm - 0.4) / 0.3).clamp(0, 1)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0: continue
                    w = 2.0 if (dx == 0 or dy == 0) else 1.0
                    sh = torch.roll(Pm, (-dy, -dx), (0, 1))
                    if free: sh = (sh - Pm) * torch.roll(occ, (-dy, -dx), (0, 1))
                    gx = gx + w * sh * dx; gy = gy + w * sh * dy
            g = torch.stack([gx.reshape(-1)[ia], gy.reshape(-1)[ia]])
            feats = fb + fm if fm is not None else fb
            if getattr(a, "pmod", False):
                g = g * torch.exp((torch.einsum("cf,cfa->a", self.wp, feats) + self.bp).clamp(-4, 3))
            force = force - P["press"] * (2.0 / n) * g / 8.0
        feats = fb + fm if fm is not None else fb
        if self.elastic:
            # every cell is as old as the seed it came from: its age is the step count t. Young cells do not
            # remember where they were (r ~ 1); from about el_t0 on they do, unless the brain says otherwise
            rr = torch.sigmoid(torch.einsum("cf,cfa->a", self.wr, feats) + self.br + (a.el_t0 - t) / a.el_tau)
            kk = a.el_kmax * torch.sigmoid(torch.einsum("cf,cfa->a", self.wk, feats) + self.bk)
            Xn = Xr + rr * dd                     # the rest position follows the cell at rate r
            force = force - kk * (1 - rr) * dd / cells
        if self.anchor:
            force = force + torch.einsum("kcf,cfa->ka", self.wa, feats) / (cells * 4 * self.wa.shape[1])   # wa . feats / (4 nC): an average, in cells/step^2
        vel = vel * P["drag"] + force
        if getattr(a, "brake", False):   # the brain may stop a cell: velocity and strafe both times a gate in (0,1)
            gate = torch.sigmoid(torch.einsum("cf,cfa->a", self.wb, feats) + self.bb)
            vel = vel * gate; strafe = strafe * gate
            if getattr(self, "probe", None) is not None: self.probe.append((ia, gate.detach(), vel.detach()))
        disp = (vel + strafe * P["strafe"]) * cells
        Q = [ma, ma * vel[0], ma * vel[1]]
        if Hd:
            dh = torch.einsum("kcf,cfa->ka", self.wh, feats) + self.bh[:, None]
            hn = torch.tanh(h + dh)
            Q = Q + [ma * hn[i] for i in range(Hd)]
        if a.colour:   # FS_BRAIN: col = base.x + mir.x, carried with the mass through the move
            Q = Q + [ma * (base[0] + mir[0])]
        if self.elastic:
            Q = Q + [ma * Xn[0], ma * Xn[1]]
        if self.Pc:
            pinfo = self.paint_info(S, ma, ia, xa, ya, lo, ro, ins_b, ins_m); self._pinfo = pinfo
            Qp = self.paint_core(S[-self.Pc:], pinfo)
        else:
            Qp = None
        S2 = self.move(torch.stack(Q), xa + disp[0], ya + disp[1], P["spread"], S.shape[1], Qd=Qp)
        if getattr(a, "feed", 0) > 0 and getattr(self, "feed_total", None):   # --feed r: lost mass comes back, every cell alike, at most r of it per step
            fct = (self.feed_total / S2[0].sum().detach()).clamp(1.0, 1.0 + a.feed)
            S2 = S2 * fct
        # trail: blur, fade, take the momentum
        p = P["persist"]; d = P["diff"] ** 2
        K = 4.0 / (torch.pow(5.0, d) - 1.0)
        reps = getattr(a, "trail_reps", 0) or getattr(a, "zoom", 1) ** 2   # --zoom z: z^2 blur passes, the same diffusion in target pixels
        Tb = T
        for _ in range(reps):
            T2d = Tb.reshape(2, n, n)
            nb = sum(torch.roll(T2d, s, ax) for s in (1, -1) for ax in (1, 2)).reshape(2, -1)
            Tb = (Tb * K + nb) / (4.0 + K)
        T2 = Tb * p + S2[1:3] * (1.0 - p.clamp(0.001, 0.999)) * 7.07
        return S2, T2

    def paint_info(self, S, ma, ia, xa, ya, lo, ro, ins_b, ins_m):
        """what the paint reads from the dynamics in one step, all detached"""
        n, cells = self.N, self.N / 2.0
        box = lambda X: F.avg_pool2d(F.pad(X[None], (1, 1, 1, 1), mode="circular"), 3, 1)[0]
        return dict(ia=ia, md=ma.detach(),
                    lx=(xa + lo[0] * cells).detach(), ly=(ya + lo[1] * cells).detach(),
                    rx=(xa + ro[0] * cells).detach(), ry=(ya + ro[1] * cells).detach(),
                    bm=box(S[0].detach().reshape(1, n, n)).reshape(-1)[ia],
                    xb=torch.asinh(torch.cat(ins_b).detach()),
                    xm=torch.asinh(torch.cat(ins_m).detach()) if self.sym else None)

    def paint_core(self, Fp, d):
        """Fp [Pc, n*n] = mass * paint; returns mass * new paint at the active cells (before the move)"""
        n, Pc = self.N, self.Pc
        ia, md, bm = d["ia"], d["md"], d["bm"]
        pv = (Fp[:, ia] / md).clamp(-0.995, 0.995)
        PL = self.sample(Fp, d["lx"], d["ly"]); PR = self.sample(Fp, d["rx"], d["ry"])
        box = lambda X: F.avg_pool2d(F.pad(X[None], (1, 1, 1, 1), mode="circular"), 3, 1)[0]
        nbp = box(Fp.reshape(Pc, n, n)).reshape(Pc, -1)[:, ia] / bm.clamp_min(1e-6)
        hb = F.relu(self.pw1 @ torch.cat([d["xb"], pv, PL, PR, nbp, bm[None]]) + self.pb1[:, None])
        if d["xm"] is not None:
            hb = hb + F.relu(self.pw1 @ torch.cat([d["xm"], pv, PR, PL, nbp, bm[None]]) + self.pb1[:, None])
        dp = self.pw2 @ hb + self.pb2[:, None]
        pn = torch.tanh(torch.atanh(pv) + dp)
        return md[None] * pn

    def move(self, Q, qx, qy, s, size, Qd=None):
        """reintegration tracking: source j's Q[:, j] is a box of side s centred at (qx, qy) in
        cells; every cell gets the fraction of the box over it. nt x nt cells can be touched."""
        n, nt = self.N, self.nt
        s = s.clamp(max=nt - 1.0)
        h = s / 2
        def taps(q):
            c0 = torch.floor(q - h + 0.5).detach()
            u = q - c0                                  # local coordinates keep the overlaps exact
            k = torch.arange(nt, dtype=torch.float32)[:, None]
            w = (torch.minimum(u + h, k + 0.5) - torch.maximum(u - h, k - 0.5)).clamp_min(0)
            w = w / w.sum(0, keepdim=True)              # each source hands on exactly what it has
            return (c0.long()[None] + k.long()) % n, w
        cx, wx = taps(torch.remainder(qx, n)); cy, wy = taps(torch.remainder(qy, n))   # on the torus, so far jumps keep their precision
        tgt = (cy[:, None] * n + cx[None]).reshape(-1)
        w = (wy[:, None] * wx[None]).reshape(-1)
        if getattr(self, "rec", None) is not None:   # recording the trajectory for paint-only training
            self.rec.append(dict(self._pinfo, tgt=tgt, w=w.detach()))
        vals = (Q[:, None] * w.reshape(nt * nt, -1)[None]).reshape(Q.shape[0], -1)
        out = torch.zeros(Q.shape[0], size).index_add(1, tgt, vals)
        if Qd is not None:   # carried along by the same move, but blind to where it went (no gradient to the positions)
            vd = (Qd[:, None] * w.detach().reshape(nt * nt, -1)[None]).reshape(Qd.shape[0], -1)
            out = torch.cat([out, torch.zeros(Qd.shape[0], size).index_add(1, tgt, vd)])
        return out

    def render(self, S):
        """[3, n, n] premultiplied colour. --rgb: the carried colour channels, (h+1)/2 * m.
        Otherwise FS_CELLS: hsv(sens*col/m + off, 0.8, 1) * m (times a brightness), [3, n, n]"""
        n = self.N
        if self.Pc:   # the paint's first three channels, (p+1)/2 * mass
            k = S.shape[0] - self.Pc
            return ((S[0].detach()[None] + S[k:k + 3]) * 0.5).reshape(3, n, n)
        if getattr(self.a, "rgb", False):
            k = 3 + self.a.hidden
            return ((S[0][None] + S[k:k + 3]) * 0.5).reshape(3, n, n)
        m = S[0].reshape(n, n)
        col = (S[-1].reshape(n, n) / m.clamp_min(1e-12)) * (m > 1e-12)
        hue = self.hue_sens * col + self.hue_off
        K = torch.tensor([1.0, 2.0 / 3.0, 1.0 / 3.0])[:, None, None]
        p = (torch.frac(torch.remainder(hue[None] + K, 1.0)) * 6.0 - 3.0).abs()
        rgb = 1.0 + 0.8 * ((p - 1.0).clamp(0, 1) - 1.0)
        return rgb * m[None] * self.cbright

    def run(self, total_mass, steps, keep=(), ckpt=True, state=None, full=False, t0=None):
        """keep: steps whose mass (or, with full, whole state) is returned. state: (S, T) to start from,
        at age t0 (default: where the last run or step left off; 0 from the seed)."""
        S, T = self.init_state(total_mass) if state is None else state
        if state is None: t0 = 0
        elif t0 is None: t0 = getattr(self, "t_now", 0)
        out = {}
        for t in range(1, steps + 1):
            if ckpt and torch.is_grad_enabled():
                S, T = checkpoint(self.step, S, T, t0 + t - 1, use_reentrant=False)
                if self.a.gdamp < 1: S, T = _Damp.apply(S, self.a.gdamp), _Damp.apply(T, self.a.gdamp)
            else:
                S, T = self.step(S, T, t0 + t - 1)
            if t in keep: out[t] = S if full else S[0].reshape(self.N, self.N)
        self.t_now = t0 + steps
        return out, S, T


# ---------------------------------------------------------------- target, loss, pictures
def load_target(a):
    from PIL import Image
    im = np.asarray(Image.open(os.path.join(HERE, "gecko.png")).convert("RGBA")).astype(np.float32) / 255
    al = im[..., 3]
    if a.luma:
        al = al * (0.3 * im[..., 0] + 0.59 * im[..., 1] + 0.11 * im[..., 2])
    al = al[::-1].copy()          # row 0 at the bottom, as the page's y
    z = getattr(a, "zoom", 1)
    if z > 1: al = np.repeat(np.repeat(al, z, 0), z, 1)   # sub-pixel: every target pixel is z x z cells
    n, h = a.grid, al.shape[0]
    t = np.zeros((n, n), np.float32)
    o = (n - h) // 2
    t[o:o + h, o:o + al.shape[1]] = al * a.mass_scale
    return torch.tensor(t)


def load_target_rgb(a):
    """gecko.png's RGB premultiplied by alpha, placed like the mass target: [3, n, n]"""
    from PIL import Image
    im = np.asarray(Image.open(os.path.join(HERE, "gecko.png")).convert("RGBA")).astype(np.float32) / 255
    rgb = (im[..., :3] * im[..., 3:4])[::-1].copy()
    z = getattr(a, "zoom", 1)
    if z > 1: rgb = np.repeat(np.repeat(rgb, z, 0), z, 1)
    n, h = a.grid, rgb.shape[0]
    t = np.zeros((3, n, n), np.float32)
    o = (n - h) // 2
    t[:, o:o + h, o:o + rgb.shape[1]] = rgb.transpose(2, 0, 1) * a.mass_scale
    return torch.tensor(t)


def ms_loss(m, tgt, scales=(1, 2, 4, 8)):
    """multiscale MSE over the target's power; m, tgt [n, n] or [C, n, n]"""
    if m.dim() == 2: m, tgt = m[None], tgt[None]
    L = 0
    for s in scales:
        a = F.avg_pool2d(m[None], s) if s > 1 else m[None]
        b = F.avg_pool2d(tgt[None], s) if s > 1 else tgt[None]
        L = L + ((a - b) ** 2).mean() / (b ** 2).mean()
    return L / len(scales)


def to_rgb_img(rgb, vmax):
    x = (rgb.detach().numpy()[:, ::-1] / vmax).clip(0, 1)
    return (x.transpose(1, 2, 0) * 255).astype(np.uint8)


def to_img(m, vmax):
    x = (m.detach().numpy()[::-1] / vmax).clip(0, 1)
    # a warm ramp: black -> amber -> white
    r = np.clip(x * 1.6, 0, 1); g = np.clip(x * 1.25 - 0.1, 0, 1); b = np.clip(x * 1.8 - 0.8, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def save_pictures(model, tgt, total, a, tag):
    from PIL import Image, ImageDraw
    n = a.grid // getattr(a, "zoom", 1); c0 = (n - 160) // 2
    crop = lambda img: img[c0:c0 + 160, c0:c0 + 160]
    T1 = a.t1
    steps = sorted(set([0, 8, 16, 24, 32, 48, a.t0, T1, T1 + 16, 160, 300, 500]))
    with torch.no_grad():
        S0, T0 = model.init_state(total)
        out, _, _ = model.run(total, max(steps), keep=set(steps), ckpt=False, full=True)
    out[0] = S0
    vmax = float(tgt.max()) * 1.2
    N = model.N
    if has_col(a):
        trgb = load_target_rgb(a)
        cv = float(trgb.max())
        show = lambda S: to_rgb_img(view(a, model.render(S)), cv)
        tshow = to_rgb_img(view(a, trgb), cv)
    else:
        show = lambda S: to_img(view(a, S[0].reshape(N, N)), vmax)
        tshow = to_img(view(a, tgt), vmax)
    out = {k: show(v) for k, v in out.items()}
    tiles = [("target", crop(tshow))] + [(f"t={s}", crop(out[s])) for s in steps]
    W = 160 * len(tiles)
    sheet = Image.new("RGB", (W, 178), (10, 12, 14)); d = ImageDraw.Draw(sheet)
    for i, (lab, im) in enumerate(tiles):
        sheet.paste(Image.fromarray(im), (160 * i, 18)); d.text((160 * i + 4, 3), lab, fill=(220, 220, 220))
    p1 = os.path.join(HERE, f"fca_gecko_{tag}_strip.png"); sheet.save(p1)
    big = Image.new("RGB", (512 * 2 + 8, 530), (10, 12, 14)); d = ImageDraw.Draw(big)
    big.paste(Image.fromarray(crop(tshow)).resize((512, 512), Image.NEAREST), (0, 18))
    big.paste(Image.fromarray(crop(out[T1])).resize((512, 512), Image.NEAREST), (520, 18))
    d.text((4, 3), "target", fill=(220, 220, 220)); d.text((524, 3), f"{tag}: grown, t={T1}", fill=(220, 220, 220))
    p2 = os.path.join(HERE, f"fca_gecko_{tag}_vs.png"); big.save(p2)
    return p1, p2


def save_params(model, path, extra):
    P = {k: float(v) for k, v in model.P().items()}
    d = dict(extra, phys=P, freq=model.freq.detach().tolist(), amp=model.amp.detach().tolist(),
             seed_v=model.seed_vel().detach().tolist(), sym=model.sym, absori=model.absori, orimix=model.orimix)
    if model.Hd:
        d.update(wh=model.wh.detach().tolist(), bh=model.bh.detach().tolist(), hgain=float(model.hgain),
                 seed_h=model.seed_h.detach().tolist())
    if model.a.see_mass: d["mgain"] = float(model.mgain)
    if getattr(model.a, "brake", False): d.update(brake_w=model.wb.detach().tolist(), brake_b=float(model.bb))
    if getattr(model, "elastic", False): d.update(el_wr=model.wr.detach().tolist(), el_br=float(model.br), el_wk=model.wk.detach().tolist(),
                                                  el_bk=float(model.bk), el_kmax=model.a.el_kmax)
    if getattr(model, "anchor", 0): d.update(anchor_w=model.wa.detach().tolist(), anchor=model.anchor)
    if getattr(model.a, "pmod", False): d.update(pmod_w=model.wp.detach().tolist(), pmod_b=float(model.bp))
    if getattr(model.a, "nbr", 0): d["nbr"] = model.a.nbr
    if getattr(model.a, "feed", 0): d["feed"] = model.a.feed
    if model.Pc: d.update(paint_w1=model.pw1.detach().tolist(), paint_b1=model.pb1.detach().tolist(),
                          paint_w2=model.pw2.detach().tolist(), paint_b2=model.pb2.detach().tolist(),
                          paint_seed=model.pseed.detach().tolist())
    if getattr(model.a, "rgb", False): d["rgb_channels"] = [3 + model.a.hidden + k for k in range(3)]
    if model.a.colour: d.update(hue_sensitivity=float(model.hue_sens), hue_offset=float(model.hue_off), brightness=float(model.cbright))
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
    ap.add_argument("--spread_to", type=float, default=0, help="pool training: anneal the spread linearly from --spread to this (0 = off)")
    ap.add_argument("--spread_anneal", type=int, default=100, help="its of that anneal")
    ap.add_argument("--pressure", type=float, default=0.3)
    ap.add_argument("--train_pressure", type=int, default=1)
    ap.add_argument("--press_mode", type=int, default=0, help="0: the page's pressure; 1: free surface (differences with in-body neighbours only)")
    ap.add_argument("--rest", type=float, default=1.0, help="mass above which pressure pushes")
    ap.add_argument("--max_spread", type=float, default=3.0, help="spread is capped here (sets the taps per axis)")
    ap.add_argument("--seed_r", type=float, default=12.0)
    ap.add_argument("--seed_vel", default="radial", choices=["radial", "rand", "zero"])
    ap.add_argument("--seed_speed", type=float, default=0.002)
    ap.add_argument("--learn_seed_v", type=int, default=1)
    ap.add_argument("--learn_seed_h", type=int, default=1, help="0: seed hidden values fixed at zero")
    ap.add_argument("--seed_lowres", type=int, default=0, help="K>0: seed velocity = default + a learned KxK correction, upsampled (2K^2 numbers in place of the full field)")
    ap.add_argument("--ori_eps", type=float, default=1e-2, help="world units; 0 = exact gradient")
    ap.add_argument("--gdamp", type=float, default=1.0, help="backward gain through each step (0.7 measured: worse than a random direction; 1.0 = exact, descends)")
    ap.add_argument("--zoom", type=int, default=1, help="sub-pixel: every target pixel is z x z cells (use --grid 256*z, --seed_r 12*z); losses on z x z averages")
    ap.add_argument("--trail_reps", type=int, default=0, help="trail blur passes per step (0 = zoom^2)")
    ap.add_argument("--luma", action="store_true")
    ap.add_argument("--mass_scale", type=float, default=1.0)
    ap.add_argument("--t0", type=int, default=48)
    ap.add_argument("--t1", type=int, default=80)
    ap.add_argument("--hold", type=int, default=16, help="also score this many steps past the end")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_seed", type=float, default=1e-2)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--init_from", default="", help="checkpoint of another variant to start from")
    ap.add_argument("--hours", type=float, default=0, help="stop after this long (0 = iters only)")
    ap.add_argument("--colour", action="store_true", help="carry the page's col channel; train hue_sensitivity on the RGB target")
    ap.add_argument("--wcol", type=float, default=0.5, help="weight of the colour loss against the mass loss")
    ap.add_argument("--hue_sens", type=float, default=None, help="initial hue_sensitivity (default: the preset's)")
    ap.add_argument("--hue_off", type=float, default=0.33, help="initial hue offset (not on the page; 0.33 = green)")
    ap.add_argument("--train_hue_off", type=int, default=1)
    ap.add_argument("--rgb", action="store_true", help="carry 3 colour channels (hidden-like, read by the eyes, rewritten by the brain); render (h+1)/2 * mass")
    ap.add_argument("--brake", action="store_true", help="the brain gates each cell's velocity: v *= sigmoid(w.feats + b)")
    ap.add_argument("--lr_brake", type=float, default=2e-2, help="Adam lr of the brake's weights and bias")
    ap.add_argument("--brake_init", type=float, default=5.0, help="initial brake bias (5: gate 0.993, nearly off)")
    ap.add_argument("--paint", type=int, default=-1, help="K>=0: a passive colour layer of 3 RGB + K paint-hidden channels (see FCA)")
    ap.add_argument("--paint_w", type=int, default=64, help="width of the paint head")
    ap.add_argument("--freeze_dyn", action="store_true", help="train only the paint layer; the dynamics stay as loaded")
    ap.add_argument("--wpix", type=float, default=0.0, help="extra weight of the full-resolution pixel MSE (mass and colour)")
    ap.add_argument("--batch", type=int, default=2, help="pool samples per iteration")
    ap.add_argument("--max_age", type=int, default=600, help="pool states older than this go back to the seed")
    ap.add_argument("--age_ramp", type=int, default=200, help="pool: the age cap grows from t1+hold to max_age over this many its")
    ap.add_argument("--pool", type=int, default=0, help="sample pool size (Growing NCA); 0 = train from the seed only")
    ap.add_argument("--seg_min", type=int, default=32)
    ap.add_argument("--seg_max", type=int, default=64)
    ap.add_argument("--score_every", type=int, default=4, help="pool training scores every this many frames once grown")
    ap.add_argument("--eval_every", type=int, default=25, help="pool training: long evaluation from the seed every N its")
    ap.add_argument("--rollback", type=float, default=1.25, help="if the running loss passes this x best, reload best and halve lr (0 = off)")
    ap.add_argument("--ckpt_min", type=float, default=10.0, help="save params+optimizer+iteration every this many minutes")
    ap.add_argument("--damage", type=int, default=0, help="pool: at most K grown samples of each batch get a random disc cut (Growing NCA exp. 3)")
    ap.add_argument("--damage_r", default="10,26", help="radius range of the cut disc, in cells")
    ap.add_argument("--damage_p", type=float, default=0.5, help="chance that an eligible sample is cut")
    ap.add_argument("--damage_age", type=int, default=96, help="only samples at least this old are cut")
    ap.add_argument("--damage_gap", type=int, default=128, help="steps after a cut before the same sample can be cut again")
    ap.add_argument("--damage_protect", type=int, default=128, help="steps after a cut during which a sample is not sent back to the seed for being the worst")
    ap.add_argument("--reinject", default="scale", choices=["scale", "edge", "seed", "none"], help="what happens to the cut mass (see damage_state)")
    ap.add_argument("--state_noise", type=float, default=0.0, help="pool: per-cell noise on grown samples' velocity (relative) and hidden (absolute)")
    ap.add_argument("--wdmg", type=float, default=0.0, help="selection: score = (undamaged + wdmg * damage_eval) / (1 + wdmg); >0 also runs the damage eval")
    ap.add_argument("--disturb", type=int, default=0, help="pool: at most K grown samples of each batch get a random mass-conserving disturbance (uses --damage_age/gap/protect)")
    ap.add_argument("--disturb_p", type=float, default=0.7, help="chance that an eligible sample is disturbed")
    ap.add_argument("--dist_u", default="0.2,1.0", help="range of disturbance strength u (1 = the old strongest; up to 2)")
    ap.add_argument("--mix_p", type=float, default=0.25, help="chance a disturbance is 2-3 milder ones at once")
    ap.add_argument("--settle", type=int, default=0, help="T>0: after a disturbance, frames are scored with weight (steps since)/T, capped at 1: time to go back")
    ap.add_argument("--nbr", type=int, default=0, help="R>0: brain also reads the (2R+1)^2 box mean of mass and hidden (isotropic)")
    ap.add_argument("--elastic", action="store_true", help="every cell carries a rest position (2 channels) that follows it at a learned rate r; a learned spring k pulls it back")
    ap.add_argument("--el_r0", type=float, default=2.0, help="initial bias of r (logit): 2 = rest position follows at 0.88 per step")
    ap.add_argument("--el_k0", type=float, default=0.02, help="initial spring, cells/step^2 per cell")
    ap.add_argument("--el_kmax", type=float, default=0.2, help="largest spring")
    ap.add_argument("--el_t0", type=float, default=140, help="age (steps) around which cells start to remember their rest position")
    ap.add_argument("--el_tau", type=float, default=8, help="how sharply (steps)")
    ap.add_argument("--anchor", type=int, default=0, help="K>0: the brain hears the cell's position in the world (K more centres of pure position waves) and adds a force in the world's frame")
    ap.add_argument("--lr_pmod", type=float, default=2e-3, help="Adam lr of the pressure gain and the anchor force")
    ap.add_argument("--pmod", action="store_true", help="the brain scales the pressure push per cell: exp(wp.feats + bp)")
    ap.add_argument("--feed", type=float, default=0.0, help="r>0: mass below the total comes back, every cell scaled alike, at most r per step")
    ap.add_argument("--chop_p", type=float, default=0.0, help="damage: chance the cut is a straight amputation instead of a disc")
    ap.add_argument("--eval_none", type=int, default=0, help="damage eval with the run's own --reinject even when it is none")
    ap.add_argument("--dmg_after", default="50,100", help="damage eval: steps after the cut that are scored")
    ap.add_argument("--opt", default="grad", choices=["grad", "es"],
                    help="grad: backprop through the damped unroll; es: antithetic evolution strategy (no gradient)")
    ap.add_argument("--es_pop", type=int, default=12, help="antithetic pairs per ES generation")
    ap.add_argument("--es_sigma", type=float, default=0.03, help="ES perturbation, relative to each tensor's scale")
    ap.add_argument("--es_lr", type=float, default=0.02, help="ES Adam step, in the same relative units")
    ap.add_argument("--es_seed", type=int, default=1, help="ES also perturbs the seed velocity (and seed hidden)")
    a = ap.parse_args()
    global DIST_U, MIX_P
    DIST_U = tuple(float(x) for x in a.dist_u.split(",")); MIX_P = a.mix_p
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

    ck_resume = None
    if a.resume == "auto":   # restart-proof: continue our own checkpoint if there is one
        p = os.path.join(HERE, f"fca_gecko_{a.tag}.pt")
        a.resume = p if os.path.exists(p) else ""
    if a.resume:
        ck = torch.load(a.resume, weights_only=False)
        model = FCA(a, presets[ck["preset"]]); model.load_state_dict(ck.get("cur", ck["state"])); start = ck["preset"]
        if "cur" in ck: ck_resume = ck
        say(f"resumed {a.resume} at it {ck.get('it', '?')} best {ck.get('best')}")
    else:
        names = list(presets) if a.preset == "auto" else a.preset.split(",")
        if a.init_from: names = names[:1]   # the warm start decides the preset; no screening
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
    if a.init_from and not ck_resume:   # warm start from another variant: same tensors copied, bigger ones filled in the corner
        ck = torch.load(a.init_from, weights_only=False); src = ck["state"]
        if not a.resume:
            start = ck["preset"]; model = FCA(a, presets[start])
        own = model.state_dict()
        sa = ck["args"]
        if sa.get("zoom", 1) != a.zoom or sa.get("grid", 256) != a.grid:
            src = rezoom(src, sa, a, model); say(f"rescaled {a.init_from} from grid {sa.get('grid')} zoom {sa.get('zoom', 1)} to grid {a.grid} zoom {a.zoom}")
        Hs = sa.get("hidden", 0) + (3 if sa.get("rgb", False) else 0); Hd = model.Hd
        if Hs != Hd and "freq" in src and sa.get("see_mass", False) == a.see_mass:
            b0 = 4 + (3 if a.see_mass else 0); f_src = src["freq"]; f_new = own["freq"].clone()
            nc = min(f_src.shape[0], f_new.shape[0]); H = min(Hs, Hd)
            f_new[:nc, :b0] = f_src[:nc, :b0]
            for blk in range(3):
                f_new[:nc, b0 + blk * Hd: b0 + blk * Hd + H] = f_src[:nc, b0 + blk * Hs: b0 + blk * Hs + H]
            src = dict(src); src["freq"] = f_new
        with torch.no_grad():
            for k, v in src.items():
                if k not in own: continue
                if own[k].shape == v.shape: own[k].copy_(v)
                elif own[k].dim() == v.dim():
                    sl = tuple(slice(0, min(x, y)) for x, y in zip(own[k].shape, v.shape))
                    own[k][sl] = v[sl]
        if not a.train_spread: own["raw.spread"] = torch.tensor(isp(a.spread - 1.0))   # an untrained spread is the one asked for
        if "freq" in src and src["freq"].shape[1] < own["freq"].shape[1] and Hs == Hd:   # new inputs (--nbr) start unheard: exact warm start
            own["freq"][:src["freq"].shape[0], src["freq"].shape[1]:] = 0
        model.load_state_dict(own)
        say(f"warm start from {a.init_from} ({ck['preset']}, best {ck.get('best')})")
    model.feed_total = total
    if a.freeze_dyn:
        for nm, p in model.named_parameters():
            if nm not in ("pw1", "pb1", "pw2", "pb2", "pseed"): p.requires_grad_(False)
    say(f"start from preset {start}; params {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    if a.pool:
        train_pool(model, a, tgt, total, a.iters, say, start, ck_resume)
    elif a.opt == "es":
        train_es(model, a, tgt, total, a.iters, say, start, ck_resume)
    else:
        train(model, a, tgt, total, a.iters, say, start, ck=ck_resume)


def rezoom(src, sa, a, model):
    """a state fitted at (grid n0, zoom z0), rewritten so that every length in cells is
    k = z/z0 times longer at (grid n, zoom z): eye distance x k, displacement per step x k
    (velocities in world units x k*n0/n, forces via global_force_mult x k), the trail the
    brain reads unchanged (sensor_gain / k), the pressure push x k over gradients k times
    shallower (pressure x k^2), the seed fields upsampled. Trail blur per step and the
    one-cell box stay in cells: that is the point."""
    n0, z0, n, z = sa.get("grid", 256), sa.get("zoom", 1), a.grid, a.zoom
    k = z / z0
    src = dict(src)
    src["raw.gfm"] = src["raw.gfm"] * k
    src["raw.dist"] = src["raw.dist"] * k
    src["raw.gain"] = src["raw.gain"] / k
    src["raw.press"] = torch.tensor(isp(float(F.softplus(src["raw.press"])) * k * k))
    src["raw.spread"] = torch.tensor(isp(max(a.spread - 1.0, 1e-6)))   # the spread asked for (in new cells)
    S = model.s1 - model.s0
    up = lambda X: F.interpolate(X[None], size=(S, S), mode="bilinear", align_corners=True)[0]
    src["seed_v"] = up(src["seed_v"]) * (k * n0 / n)
    for key in ("seed_h", "pseed"):
        if key in src: src[key] = up(src[key])
    return src


def lr_factor(i, iters):
    return min(1.0, (i + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(i / max(iters, 1), 1))))


def save_ckpt(path, **kw):
    """atomic: a container restart mid-write must not leave half a checkpoint"""
    torch.save(kw, path + ".tmp"); os.replace(path + ".tmp", path)


def finish(model, a, tgt, total, say, name, best, best_state, iters):
    if best_state is None: return
    model.load_state_dict(best_state)
    r = long_eval(model, a, tgt, total)
    save_params(model, os.path.join(HERE, f"fca_gecko_{a.tag}.json"), {"preset": name, "best": best, "iters": iters, "args": vars(a),
                                                                         "long_eval": {str(k): v for k, v in r.items()}})
    say(f"best params, from the seed: {r}")
    say("pictures:", *save_pictures(model, tgt, total, a, a.tag))


def has_col(a):
    return a.colour or getattr(a, "rgb", False) or getattr(a, "paint", -1) >= 0


def pix_loss(m, tgt):
    """full-resolution MSE over the target's power"""
    return ((m - tgt) ** 2).mean() / (tgt ** 2).mean()


def view(a, X):
    """--zoom z: what is compared with (and shown as) the picture, the z x z average of the cells"""
    z = getattr(a, "zoom", 1)
    if z <= 1: return X
    return F.avg_pool2d(X[None, None], z)[0, 0] if X.dim() == 2 else F.avg_pool2d(X[None], z)[0]


def frame_loss(model, S, tgt, trgb, a):
    """(training objective, mass ms_loss, colour ms_loss) for one full state. --zoom: on the
    z x z averages (the target's own resolution), so blur below a target pixel costs nothing"""
    if getattr(a, "zoom", 1) > 1:
        tgt = view(a, tgt); trgb = view(a, trgb) if trgb is not None else None
        m = view(a, S[0].reshape(model.N, model.N))
        lm = ms_loss(m, tgt)
        L = lm + a.wpix * pix_loss(m, tgt) if a.wpix else lm
        if not has_col(a): return L, lm, torch.zeros(())
        R = view(a, model.render(S)); lc = ms_loss(R, trgb)
        return L + a.wcol * (lc + a.wpix * pix_loss(R, trgb) if a.wpix else lc), lm, lc
    m = S[0].reshape(model.N, model.N)
    lm = ms_loss(m, tgt)
    L = lm + a.wpix * pix_loss(m, tgt) if a.wpix else lm
    if not has_col(a): return L, lm, torch.zeros(())
    R = model.render(S)
    lc = ms_loss(R, trgb)
    L = L + a.wcol * (lc + a.wpix * pix_loss(R, trgb) if a.wpix else lc)
    return L, lm, lc


def train(model, a, tgt, total, iters, say, name, quiet=False, ck=None):
    trgb = load_target_rgb(a) if has_col(a) else None
    seedp = [p for n, p in model.named_parameters() if n.startswith("seed_") and p.requires_grad]
    brk = [p for n, p in model.named_parameters() if n in ("wb", "bb", "wp", "bp") and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith("seed_") and n not in ("wb", "bb", "wp", "bp") and p.requires_grad]
    groups = [{"params": rest, "lr": a.lr}, {"params": seedp, "lr": a.lr_seed * a.seed_speed}]
    if brk: groups.append({"params": brk, "lr": a.lr_brake})
    opt = torch.optim.Adam(groups)
    base_lr = [g["lr"] for g in opt.param_groups]
    best, best_state, it0, hist, lr_scale, best_it = 1e9, None, 0, [], 1.0, 0
    ckp = os.path.join(HERE, f"fca_gecko_{a.tag}.pt")
    if ck is not None:
        opt.load_state_dict(ck["opt"]); best, best_state = ck["best"], ck["state"]
        it0, hist = ck["it"] + 1, ck.get("hist", [])
        lr_scale, best_it = ck.get("lr_scale", 1.0), ck.get("best_it", ck["it"])
        if "rng" in ck: torch.set_rng_state(ck["rng"])
    def checkpoint_now(it):
        save_ckpt(ckp, preset=name, state=best_state, cur={k: v.detach().clone() for k, v in model.state_dict().items()},
                  opt=opt.state_dict(), it=it, best=best, hist=hist, args=vars(a), rng=torch.get_rng_state(), kind="grad",
                  lr_scale=lr_scale, best_it=best_it)
    t_start = t_ck = time.time()
    it = it0 - 1
    for it in range(it0, iters):
        t = time.time()
        for g, b in zip(opt.param_groups, base_lr): g["lr"] = b * lr_factor(it, iters) * lr_scale
        tend = int(torch.randint(a.t0, a.t1 + 1, (1,)))
        keep = set(range(a.t0, tend + 1, 8)) | {tend, tend + a.hold} if a.hold else set(range(a.t0, tend + 1, 8)) | {tend}
        out, S, T = model.run(total, max(keep), keep=keep, full=True)
        losses = [frame_loss(model, St, tgt, trgb, a)[0] for St in out.values()]
        loss = sum(losses) / len(losses)
        if not torch.isfinite(loss):
            say(f"it {it}: non-finite loss, restoring best");
            if best_state: model.load_state_dict(best_state)
            continue
        opt.zero_grad(); loss.backward()
        # every tensor's gradient to unit norm before Adam: the unroll is chaotic, its gradient
        # spans ten orders of magnitude from one iteration to the next, and one spike would
        # otherwise poison Adam's second moment for hundreds of iterations
        gn, ok = 0.0, True
        for g in opt.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                nrm = float(p.grad.norm()); gn = max(gn, nrm)
                if not math.isfinite(nrm): ok = False; break
                p.grad /= (nrm + 1e-12) / a.clip if nrm > 0 else 1
        if ok: opt.step()
        L = float(loss.detach()); hist.append(L)
        ema = sum(hist[-20:]) / len(hist[-20:])
        if ema < best and it > 10:
            best = ema; best_it = it; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if a.rollback and best_state is not None and it - best_it > 20 and ema > best * a.rollback:
            # the unroll has walked into a chaotic regime where the gradient is noise; go back
            model.load_state_dict(best_state); opt.state.clear(); lr_scale *= 0.5; best_it = it; hist.append(best)
            say(f"it {it}: ema {ema:.4f} > {a.rollback} x best {best:.4f}, back to best, lr x{lr_scale:.3g}")
        mass_err = float(S[0].sum()) / total - 1
        if not quiet and (it % 10 == 0 or it == iters - 1):
            P = {k: round(float(v), 4) for k, v in model.P().items()}
            say(f"it {it:5d} loss {L:.4f} ema {ema:.4f} best {best:.4f} T={tend} gn {gn:.2e} "
                f"dmass {mass_err:+.1e} {time.time() - t:.2f}s/it  {P}")
        if not quiet and time.time() - t_ck > a.ckpt_min * 60:
            checkpoint_now(it); t_ck = time.time()
        if not quiet and (it % 100 == 0) and best_state is not None:
            cur = {k: v.clone() for k, v in model.state_dict().items()}
            finish(model, a, tgt, total, say, name, best, best_state, it + 1)
            model.load_state_dict(cur)
        if a.hours and time.time() - t_start > a.hours * 3600 and not quiet:
            say(f"time limit at it {it}"); break
    if not quiet:
        checkpoint_now(it)
        finish(model, a, tgt, total, say, name, best, best_state, it + 1)
    elif best_state is not None:
        model.load_state_dict(best_state)
    return best


def long_eval(model, a, tgt, total, steps=(96, 160, 300, 500)):
    trgb = load_target_rgb(a) if has_col(a) else None
    with torch.no_grad():
        out, _, _ = model.run(total, max(steps), keep=set(steps), ckpt=False, full=True)
    r = {}
    for t in steps:
        L, lm, lc = frame_loss(model, out[t], tgt, trgb, a)
        r[t] = (round(float(lm), 4), round(float(lc), 4)) if has_col(a) else round(float(lm), 4)
    return r


# ---------------------------------------------------------------- damage (Growing NCA, experiment 3)
def damage_state(model, S, T, cut, total, reinject="scale"):
    """cut [n*n] in [0,1]: the fraction of each cell removed. Every per-cell number goes with it
    (mass, momentum, hidden, paint: all are carried mass-weighted, so S * (1 - cut)), and the
    trail under the cut is zeroed too, so nothing at the wound remembers what was there.
    Mass is conserved by the step, so what is cut is gone for good unless it comes back:
      scale  the remaining cells are scaled by total / remaining, every channel alike, so each
             keeps its velocity, hidden and paint per unit mass and is only denser. The added
             mass carries no hint of where the wound is; pressure (rest density) pushes the
             excess outwards, into the hole, and the brain has to finish the shape.
      edge   the lost mass goes back into the body cells within EDGE_W cells of the wound (each
             scaled up, keeping its velocity, hidden and paint per unit mass): a dense band at the
             cut, like a blastema; pressure pushes it into the hole, the brain shapes it
      seed   the removed mass comes back as a seed (the learned seed disc, velocity, hidden and
             paint) at the centre, added to whatever is there.
      none   gone (the target is then unreachable)."""
    keep = 1.0 - cut
    S2 = S * keep[None]; T2 = T * keep[None]
    if reinject == "scale":
        S2 = S2 * (total / S2[0].sum().clamp_min(1e-12))
    elif reinject == "edge":   # back into the body cells within `EDGE_W` cells of the wound, ~ their mass
        n = model.N
        c2 = (cut > 0.5).float().reshape(1, 1, n, n)
        k = 2 * EDGE_W + 1
        near = F.max_pool2d(F.pad(c2, (EDGE_W,) * 4, mode="circular"), k, 1).reshape(-1)
        w = S2[0] * near * (1 - (cut > 0.5).float())
        if float(w.sum()) <= 1e-9: w = S2[0]
        lost = total - S2[0].sum()
        S2 = S2 * (1 + lost * w / w.sum() / S2[0].clamp_min(1e-12))[None]
    elif reinject == "seed":
        lost = total - float(S2[0].sum())
        if lost > 0: S2 = S2 + model.init_state(lost)[0]
    return S2, T2


EDGE_W = 6


def disc_cut(model, cx, cy, r):
    d = torch.sqrt((model.xf - cx) ** 2 + (model.yf - cy) ** 2)
    return (r + 0.5 - d).clamp(0, 1)


def random_disc_cut(model, S, ra, rb, gen=None, max_frac=0.35):
    """a disc of radius U(ra, rb) cells centred on a cell drawn with probability ~ its mass
    (so on the body); re-drawn smaller if it would take more than max_frac of the mass"""
    m = S[0].clamp_min(0)
    j = int(torch.multinomial(m / m.sum(), 1, generator=gen))
    cx, cy = float(model.xf[j]), float(model.yf[j])
    r = ra + (rb - ra) * float(torch.rand(1, generator=gen))
    for _ in range(8):
        cut = disc_cut(model, cx, cy, r)
        if float((m * cut).sum() / m.sum()) <= max_frac: break
        r *= 0.8
    return cut, (cx, cy, r)


def random_chop_cut(model, S, gen=None, fr_range=(0.08, 0.3)):
    """an amputation: everything beyond a random straight line, the line placed so that a
    fraction U(fr_range) of the mass is cut"""
    m = S[0].clamp_min(0)
    th = float(torch.rand(1, generator=gen)) * 2 * math.pi
    proj = model.xf * math.cos(th) + model.yf * math.sin(th)
    want = fr_range[0] + (fr_range[1] - fr_range[0]) * float(torch.rand(1, generator=gen))
    o = torch.argsort(proj, descending=True)
    cm = torch.cumsum(m[o], 0) / m.sum()
    j = int(torch.searchsorted(cm, torch.tensor([want]))[0].clamp(max=len(o) - 1))
    thr = float(proj[o[j]])
    return (proj >= thr).float(), (th, want)


def std_damages(model, a):
    """the standard cuts of the robustness benchmark, fixed in target-picture coordinates
    (x right, y up, 0..127; the picture sits in the middle of the grid): fraction of the
    target's mass in brackets"""
    z = getattr(a, "zoom", 1); n = model.N; o = (n - 128 * z) // 2
    X = (model.xf - o) / z; Y = (model.yf - o) / z
    g = torch.Generator().manual_seed(1234)
    return {
        "tail": ((X >= 76) & (X < 128) & (Y >= 0) & (Y <= 30)).float(),                              # the tail curl (12.5%)
        "head": ((X >= 0) & (X <= 56) & (Y >= 88) & (Y < 128)).float(),                              # the head (18.7%)
        "disc24": ((torch.sqrt((X - 72) ** 2 + (Y - 60) ** 2) <= 12)).float(),  # a 24-cell-wide hole in the body (10%)
        "rand30": (torch.rand(n * n, generator=g) < 0.3).float(),             # 30% of the cells, salt-and-pepper
    }



# ---------------------------------------------------------------- mass-conserving disturbances
# Every disturbance keeps the mass: it moves it, or changes its velocity, hidden or colour.
# Velocities in the state are world units per step: CELLS per step / (n / 2).
def displace(model, S, dx, dy):
    """move every cell's contents by (dx, dy) cells ([n*n] fields; sub-cell by the model's own
    box deposit of side 1), every channel alike: mass, momentum, hidden, paint. Conserves."""
    n = model.N
    return model.move(S, model.xf + dx, model.yf + dy, torch.tensor(1.0), S.shape[1])


def brush_drag(model, S, x0, y0, r, dx, dy):
    """the page's push tool: the cells under a soft disc brush (smoothstep falloff to radius r)
    are displaced along the drag, in sub-steps of at most one cell with the brush moving along
    the drag (gecko.html displace()). x0, y0 in the page's cell coordinates (cell i spans [i, i+1])."""
    n = model.N; L = math.hypot(dx, dy)
    k = max(1, math.ceil(L))
    for i in range(k):
        cx, cy = x0 + dx * (i + 0.5) / k, y0 + dy * (i + 0.5) / k
        ddx = (model.xf + 0.5 - cx).abs(); ddy = (model.yf + 0.5 - cy).abs()
        ddx = torch.minimum(ddx, n - ddx); ddy = torch.minimum(ddy, n - ddy)
        w = (1 - torch.sqrt(ddx ** 2 + ddy ** 2) / r).clamp(0, 1); w = w * w * (3 - 2 * w)
        S = model.move(S, model.xf + w * dx / k, model.yf + w * dy / k, torch.tensor(1.0), S.shape[1])
    return S


def shake(model, S, dv, gen):
    """the page's 'shake everything': every cell a velocity kick, random direction, |kick| = dv * sqrt(U)"""
    a = torch.rand(S.shape[1], generator=gen) * 2 * math.pi; r = torch.sqrt(torch.rand(S.shape[1], generator=gen))
    S = S.clone(); S[1] += S[0] * dv * r * torch.cos(a) / (model.N / 2.0); S[2] += S[0] * dv * r * torch.sin(a) / (model.N / 2.0)
    return S


def far_body_point(model, S, gen, k=12):
    """a cell on the body far from its centre of mass: the farthest of k drawn by mass"""
    m = S[0].clamp_min(0); w = m / m.sum()
    cx, cy = float((w * model.xf).sum()), float((w * model.yf).sum())
    j = torch.multinomial(w, k, replacement=True, generator=gen)
    d = (model.xf[j] - cx) ** 2 + (model.yf[j] - cy) ** 2
    jj = int(j[int(d.argmax())])
    return float(model.xf[jj]), float(model.yf[jj]), cx, cy


def smooth_field(model, gen, k=4):
    """a smooth random scalar field over the grid, ~N(0,1), from a k x k grid, bicubic, periodic"""
    n = model.N
    z = torch.randn(1, 1, k, k, generator=gen)
    z = F.pad(z, (1, 1, 1, 1), mode="circular")
    u = F.interpolate(z, size=(n * (k + 2) // k, n * (k + 2) // k), mode="bicubic", align_corners=False)
    o = n // k
    return u[0, 0, o:o + n, o:o + n].reshape(-1)


def body_point(model, S, gen):
    m = S[0].clamp_min(0)
    j = int(torch.multinomial(m / m.sum(), 1, generator=gen))
    return float(model.xf[j]), float(model.yf[j])


DISTURB_KINDS = ("drag", "drag", "limbdrag", "limbdrag", "shake", "shake", "kick", "swirl", "shove", "partoff", "velnoise", "warp", "blur", "hscram")


def disturb_state(model, S, T, kind, gen, u=None):
    """one random disturbance of kind `kind` at strength u in [0,1] (random if None); returns (S, T, text)"""
    n, cells, Hd = model.N, model.N / 2.0, model.Hd
    R = lambda lo, hi: lo + (hi - lo) * float(torch.rand(1, generator=gen))
    if u is None: u = R(*DIST_U)
    S = S.clone(); m = S[0]
    ang = R(0, 2 * math.pi); cs, sn = math.cos(ang), math.sin(ang)
    if kind == "drag":       # the page's push brush: a soft disc dragged 4-30 cells
        cx, cy = body_point(model, S, gen); r = R(8, 32); k = 4 + u * 26
        return brush_drag(model, S, cx + 0.5, cy + 0.5, r, k * cs, k * sn), T, f"drag r{r:.0f} {k:.0f}c"
    if kind == "limbdrag":   # the brush on an end of the body (tail, head, a foot), dragged away from the body
        x, y, cx, cy = far_body_point(model, S, gen); r = R(8, 24); k = 8 + u * 22
        th = math.atan2(y - cy, x - cx) + R(-1.0, 1.0)
        return brush_drag(model, S, x + 0.5, y + 0.5, r, k * math.cos(th), k * math.sin(th)), T, f"limbdrag r{r:.0f} {k:.0f}c"
    if kind == "shake":      # the page's shake everything
        dv = 0.3 + u * 4.7
        return shake(model, S, dv, gen), T, f"shake {dv:.1f}"
    if kind == "kick":       # a velocity kick in a disc, then the move carries it
        cx, cy = body_point(model, S, gen); r = R(8, 32); sp = u * 12.0
        mk = disc_cut(model, cx, cy, r)
        S[1] += m * mk * sp * cs / cells; S[2] += m * mk * sp * sn / cells
        return S, T, f"kick r{r:.0f} {sp:.1f}c/s"
    if kind == "swirl":
        cx, cy = body_point(model, S, gen); r = R(10, 32); sp = u * 10.0 * (1 if R(0, 1) < 0.5 else -1)
        mk = disc_cut(model, cx, cy, r)
        S[1] += m * mk * (-(model.yf - cy) / r) * sp / cells; S[2] += m * mk * ((model.xf - cx) / r) * sp / cells
        return S, T, f"swirl r{r:.0f} {sp:.1f}"
    if kind == "shove":      # the contents of a disc shifted by k cells
        cx, cy = body_point(model, S, gen); r = R(8, 28); k = 3 + u * 15
        mk = disc_cut(model, cx, cy, r)
        S = S * (1 - mk) + displace(model, S * mk, k * cs, k * sn)
        return S, T, f"shove r{r:.0f} {k:.0f}c"
    if kind == "partoff":    # an end of the body (beyond a random line) moved 10-30 cells away from it
        mk, (th, fr) = random_chop_cut(model, S, gen, (0.06, 0.25)); k = 8 + u * 22
        th = th + R(-0.8, 0.8)
        S = S * (1 - mk) + displace(model, S * mk, k * math.cos(th), k * math.sin(th))
        return S, T, f"partoff {fr:.0%} {k:.0f}c"
    if kind == "velnoise":   # every cell a random velocity, white or smooth
        sp = u * 4.0
        if R(0, 1) < 0.5:
            nz = torch.randn(2, S.shape[1], generator=gen)
        else:
            nz = torch.stack([smooth_field(model, gen, 8), smooth_field(model, gen, 8)])
        S[1:3] += m * nz * sp / cells
        return S, T, f"velnoise {sp:.1f}"
    if kind == "warp":       # the whole field displaced by a smooth random field
        A = u * 6.0
        dx = smooth_field(model, gen, 4) * A; dy = smooth_field(model, gen, 4) * A
        return displace(model, S, dx, dy), T, f"warp {A:.1f}c"
    if kind == "blur":       # box blur of every channel, conserving
        k = 1 + int(u * 3.99)
        X = S.reshape(-1, n, n)
        for _ in range(k): X = F.avg_pool2d(F.pad(X[None], (1, 1, 1, 1), mode="circular"), 3, 1)[0]
        return X.reshape(S.shape[0], -1), T, f"blur x{k}"
    if kind == "hscram":     # hidden values scrambled (and the colour layer too, when there is one)
        sd = 0.2 + u * 0.8; mm = m.clamp_min(1e-12)
        for lo_, hi_ in ((3, 3 + Hd), (S.shape[0] - model.Pc, S.shape[0])):
            if hi_ <= lo_: continue
            h = (S[lo_:hi_] / mm + sd * torch.randn(hi_ - lo_, S.shape[1], generator=gen)).clamp(-0.99, 0.99)
            S[lo_:hi_] = h * m
        return S, T, f"hscram {sd:.2f}"
    raise ValueError(kind)


DIST_U = (0.2, 1.0)
MIX_P = 0.25


def random_disturb(model, S, T, gen, kinds=DISTURB_KINDS, mix_p=None):
    """one disturbance, or with chance mix_p two or three milder ones"""
    if mix_p is None: mix_p = MIX_P
    if float(torch.rand(1, generator=gen)) < mix_p:
        k = 2 + int(float(torch.rand(1, generator=gen)) < 0.5); txt = []
        for _ in range(k):
            kd = kinds[int(torch.randint(len(kinds), (1,), generator=gen))]
            S, T, t = disturb_state(model, S, T, kd, gen, u=(0.1 + 0.5 * float(torch.rand(1, generator=gen))) * DIST_U[1]); txt.append(t)
        return S, T, "+".join(txt)
    kd = kinds[int(torch.randint(len(kinds), (1,), generator=gen))]
    return disturb_state(model, S, T, kd, gen)


def std_disturbs(model, a, S, T):
    """the standard disturbances of the benchmark, fixed (target-picture coordinates as std_damages)"""
    n, cells = model.N, model.N / 2.0
    z = getattr(a, "zoom", 1); o = (n - 128 * z) // 2
    P = lambda X, Y: (o + X * z, o + Y * z)
    g = torch.Generator().manual_seed(4321)
    m = S[0]; d = std_damages(model, a); out = {}
    cx, cy = P(72, 60)
    mk = disc_cut(model, cx, cy, 20)
    X = S.clone(); X[1] += m * mk * 4.0 / cells; out["kick"] = X
    mk = disc_cut(model, cx, cy, 24); X = S.clone(); r = 24
    X[1] += m * mk * (-(model.yf - cy) / r) * 6.0 / cells; X[2] += m * mk * ((model.xf - cx) / r) * 6.0 / cells; out["swirl"] = X
    mk = disc_cut(model, cx, cy, 16); out["shove"] = S * (1 - mk) + displace(model, S * mk, 12.0, 0.0)
    q = 20 / math.sqrt(2)
    mk = d["tail"]; out["tail_off"] = S * (1 - mk) + displace(model, S * mk, q, -q)
    mk = d["head"]; out["head_off"] = S * (1 - mk) + displace(model, S * mk, -q, q)
    out["shake"] = shake(model, S, 2.0, g)
    out["drag16"] = brush_drag(model, S, cx + 0.5, cy + 0.5, 16, 16.0, 0.0)
    tx, ty = P(98, 16); out["tail_drag"] = brush_drag(model, S, tx, ty, 16, q, -q)
    hx, hy = P(26, 106); out["head_drag"] = brush_drag(model, S, hx, hy, 18, -q, q)
    out["warp"] = displace(model, S, smooth_field(model, g, 4) * 4.0, smooth_field(model, g, 4) * 4.0)
    Xb = S.reshape(-1, n, n)
    for _ in range(3): Xb = F.avg_pool2d(F.pad(Xb[None], (1, 1, 1, 1), mode="circular"), 3, 1)[0]
    out["blur"] = Xb.reshape(S.shape[0], -1)
    X = S.clone(); mm = m.clamp_min(1e-12); Hd = model.Hd
    X[3:3 + Hd] = (X[3:3 + Hd] / mm + 0.7 * torch.randn(Hd, S.shape[1], generator=g)).clamp(-0.99, 0.99) * m; out["hscram"] = X
    return {k: (v, T) for k, v in out.items()}


def disturb_eval(model, a, tgt, total, t_dmg=200, after=(50, 100), grown=None, kinds=None):
    """grow to t_dmg, apply each standard disturbance, run on; mass ms_loss at t_dmg + each of `after`"""
    with torch.no_grad():
        if grown is None:
            _, S, T = model.run(total, t_dmg, ckpt=False)
        else:
            S, T = grown
        r = {}
        for nm, st in std_disturbs(model, a, S, T).items():
            if kinds and nm not in kinds: continue
            out, _, _ = model.run(total, max(after), keep=set(after), ckpt=False, state=st, t0=t_dmg)
            r[nm] = {k: round(float(ms_loss(view(a, out[k]), view(a, tgt))), 4) for k in after}
    return r


def damage_eval(model, a, tgt, total, t_dmg=200, after=(50, 100), reinject="scale", grown=None):
    """grow to t_dmg, apply each standard cut, run on; mass ms_loss at t_dmg + each of `after`"""
    with torch.no_grad():
        if grown is None:
            _, S, T = model.run(total, t_dmg, ckpt=False)
        else:
            S, T = grown
        r = {}
        for nm, cut in std_damages(model, a).items():
            S2, T2 = damage_state(model, S, T, cut, total, reinject)
            out, _, _ = model.run(total, max(after), keep=set(after), ckpt=False, state=(S2, T2), t0=t_dmg)
            r[nm] = {k: round(float(ms_loss(view(a, out[k]), view(a, tgt))), 4) for k in after}
    return r


def damage_score(r):
    v = [x for d in r.values() for x in d.values()]
    v = [x if math.isfinite(x) else 1e3 for x in v]
    return sum(v) / len(v)


def eval_score(r):
    """one number from long_eval: mean over the steps of mass (+ colour) loss"""
    v = [sum(x) if isinstance(x, tuple) else x for x in r.values()]
    v = [x if math.isfinite(x) else 1e3 for x in v]
    return sum(v) / len(v)


def train_pool(model, a, tgt, total, iters, say, name, ck=None):
    """Growing NCA's sample pool. The pool holds whole states (mass, momentum, hidden, colour
    AND the trail) and how many steps each has lived. Each iteration takes `batch` at random;
    the worst of them (by its current loss) goes back to the seed, and so does any older than
    the age cap. A seed is grown to t0..t1+hold and scored every 8 steps from t0, like the plain
    fit; a grown state runs on from where it was for seg_min..seg_max steps and is scored every
    score_every frames. The gradient goes back through that run only; the end states are
    written back. Curriculum: the age cap grows from t1+hold to max_age over age_ramp its.
    Selection: the best params are the ones with the best long evaluation from the seed
    (mean loss at 96, 160, 300, 500), not the best training loss, which mixes ages.
    (The first version of this function forgot to pass the pool state to run(), so every
    'pool' segment silently re-grew from the seed and was scored as a grown gecko.)"""
    trgb = load_target_rgb(a) if has_col(a) else None
    seedp = [p for n, p in model.named_parameters() if n.startswith("seed_") and p.requires_grad]
    brk = [p for n, p in model.named_parameters() if n in ("wb", "bb") and p.requires_grad]
    PM = ("wp", "bp", "wa", "wr", "br", "wk", "bk")
    pm = [p for n, p in model.named_parameters() if n in PM and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith("seed_") and n not in ("wb", "bb") + PM and p.requires_grad]
    groups = [{"params": rest, "lr": a.lr}, {"params": seedp, "lr": a.lr_seed * a.seed_speed}]
    if brk: groups.append({"params": brk, "lr": a.lr_brake})
    if pm: groups.append({"params": pm, "lr": getattr(a, "lr_pmod", 2e-3)})
    opt = torch.optim.Adam(groups)
    base_lr = [g["lr"] for g in opt.param_groups]
    ckp = os.path.join(HERE, f"fca_gecko_{a.tag}.pt")
    pool = [None] * a.pool          # None = a fresh seed
    ages = [0] * a.pool
    cut_at = [-1] * a.pool          # age at the last cut, -1 = never cut
    dmg_r = [float(x) for x in str(getattr(a, "damage_r", "10,26")).split(",")]
    dgen = torch.Generator().manual_seed(a.seed + 77)
    wdmg = getattr(a, "wdmg", 0.0)
    def evaluate():
        r = long_eval(model, a, tgt, total); su = eval_score(r)
        if wdmg <= 0: return r, su, None, su
        if getattr(a, "disturb", 0):
            rd = disturb_eval(model, a, tgt, total, after=tuple(int(x) for x in str(getattr(a, "dmg_after", "50,100")).split(",")))
            sd = damage_score(rd)
            return r, su, rd, (su + wdmg * sd) / (1 + wdmg)
        rd = damage_eval(model, a, tgt, total, reinject=a.reinject if (a.reinject != "none" or getattr(a, "feed", 0) or getattr(a, "eval_none", 0)) else "scale",
                         after=tuple(int(x) for x in str(getattr(a, "dmg_after", "50,100")).split(","))); sd = damage_score(rd)
        return r, su, rd, (su + wdmg * sd) / (1 + wdmg)
    best, best_state, it0, hist, lr_scale, best_it, evals = 1e9, None, 0, [], 1.0, 0, []
    if ck is not None:
        opt.load_state_dict(ck["opt"]); best, best_state = ck["best"], ck["state"]
        it0, hist = ck["it"] + 1, ck.get("hist", [])
        lr_scale, best_it, evals = ck.get("lr_scale", 1.0), ck.get("best_it", ck["it"]), ck.get("evals", [])
        if "pool" in ck: pool, ages = ck["pool"], ck["ages"]
        if "cut_at" in ck: cut_at = ck["cut_at"]
        if "dgen" in ck: dgen.set_state(ck["dgen"])
        if "rng" in ck: torch.set_rng_state(ck["rng"])
    def checkpoint_now(it):
        save_ckpt(ckp, preset=name, state=best_state, cur={k: v.detach().clone() for k, v in model.state_dict().items()},
                  opt=opt.state_dict(), it=it, best=best, hist=hist, args=vars(a), rng=torch.get_rng_state(), kind="pool",
                  lr_scale=lr_scale, best_it=best_it, pool=pool, ages=ages, evals=evals, cut_at=cut_at, dgen=dgen.get_state())
    if ck is None:
        r, su, rd, best = evaluate(); best_it = 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        evals.append((0, r) if rd is None else (0, r, rd))
        say(f"pool {a.pool} x{a.batch}, segments {a.seg_min}..{a.seg_max}, score every {a.score_every}, "
            f"age cap {a.t1 + a.hold}->{a.max_age} over {a.age_ramp} its; before: {r} score {su:.4f}"
            + (f"; damaged: {rd} score {damage_score(rd):.4f}; selection score {best:.4f}" if rd is not None else ""))
    cap = lambda it: int(a.t1 + a.hold + (a.max_age - a.t1 - a.hold) * min(1.0, it / max(a.age_ramp, 1)))
    t_start = t_ck = time.time(); it = it0 - 1
    for it in range(it0, iters):
        t = time.time()
        for g, b in zip(opt.param_groups, base_lr): g["lr"] = b * lr_factor(it, iters) * lr_scale
        if getattr(a, "spread_to", 0) > 0:   # forced anneal of the deposit box: spread -> spread_to over spread_anneal its
            u = min(1.0, it / max(a.spread_anneal, 1)); sp = a.spread + (a.spread_to - a.spread) * u
            with torch.no_grad(): model.raw["spread"].fill_(isp(max(sp - 1.0, 1e-6)))
        idx = torch.randperm(a.pool)[:a.batch].tolist()
        with torch.no_grad():   # the worst of the batch goes back to the seed, and anything past the age cap
            cur = [float(frame_loss(model, pool[i][0], tgt, trgb, a)[0]) if pool[i] is not None else 1e9 for i in idx]
        # a sample still regrowing from a cut is not sent back for being the worst (it is the point)
        healthy = [k for k, i in enumerate(idx) if cut_at[i] < 0 or ages[i] - cut_at[i] >= getattr(a, "damage_protect", 0)]
        if healthy:
            worst = idx[max(healthy, key=lambda k: cur[k])]
            pool[worst], ages[worst], cut_at[worst] = None, 0, -1
        for i in idx:
            if ages[i] > cap(it): pool[i], ages[i], cut_at[i] = None, 0, -1
        cuts = []
        if getattr(a, "damage", 0):   # Growing NCA exp. 3: cut the best of the grown ones, give the mass back
            elig = sorted([k for k, i in enumerate(idx) if pool[i] is not None and ages[i] >= a.damage_age
                           and (cut_at[i] < 0 or ages[i] - cut_at[i] >= a.damage_gap)], key=lambda k: cur[k])
            for k in elig[:a.damage]:
                if float(torch.rand(1, generator=dgen)) >= a.damage_p: continue
                i = idx[k]
                if float(torch.rand(1, generator=dgen)) < getattr(a, "chop_p", 0.0):
                    cut, _ = random_chop_cut(model, pool[i][0], dgen); r = -1
                else:
                    cut, (cx, cy, r) = random_disc_cut(model, pool[i][0], dmg_r[0], dmg_r[1], dgen)
                fr = float((pool[i][0][0] * cut).sum()) / total
                pool[i] = damage_state(model, pool[i][0], pool[i][1], cut, total, a.reinject)
                cut_at[i] = ages[i]; cuts.append(f"{'chop' if r < 0 else f'cut r{r:.0f}'} {fr:.0%}")
        if getattr(a, "disturb", 0):   # mass-conserving disturbances of grown samples (the best of the batch first)
            elig = sorted([k for k, i in enumerate(idx) if pool[i] is not None and ages[i] >= a.damage_age
                           and (cut_at[i] < 0 or ages[i] - cut_at[i] >= a.damage_gap)], key=lambda k: cur[k])
            for k in elig[:a.disturb]:
                if float(torch.rand(1, generator=dgen)) >= a.disturb_p: continue
                i = idx[k]
                with torch.no_grad():
                    S_, T_, txt = random_disturb(model, pool[i][0], pool[i][1], dgen)
                pool[i] = (S_, T_); cut_at[i] = ages[i]; cuts.append(txt)
        opt.zero_grad(); info = [] + cuts
        tot_loss, ok = 0.0, True
        for i in idx:
            age = ages[i]
            if pool[i] is None:   # a seed keeps its gradient to the seed values
                st, age = None, 0
                L = int(torch.randint(a.t0, a.t1 + a.hold + 1, (1,)))
                keep = {k for k in range(a.t0, L + 1) if k % 8 == 0 or k == L}
            else:
                st = (pool[i][0].clone(), pool[i][1].clone())
                if getattr(a, "state_noise", 0) > 0:   # mass untouched (conserved); velocity x (1 + s N), hidden + s N
                    S0, sn = st[0], a.state_noise
                    mm = S0[0].clamp_min(1e-12)
                    S0[1:3] = S0[1:3] * (1 + sn * torch.randn(2, S0.shape[1], generator=dgen))
                    if model.Hd:
                        hh = (S0[3:3 + model.Hd] / mm + sn * torch.randn(model.Hd, S0.shape[1], generator=dgen)).clamp(-0.99, 0.99)
                        S0[3:3 + model.Hd] = hh * S0[0]
                L = int(torch.randint(a.seg_min, a.seg_max + 1, (1,)))
                keep = {k for k in range(1, L + 1) if k % a.score_every == 0 or k == L}
            out, S, T = model.run(total, L, keep=keep, full=True, state=st, t0=age)
            parts = [frame_loss(model, St, tgt, trgb, a) for St in out.values()]
            if getattr(a, "settle", 0) > 0 and cut_at[i] >= 0:   # time to go back: early frames after a disturbance count less
                wts = [min(1.0, (age + k - cut_at[i]) / a.settle) for k in out.keys()]
                loss = sum(w * p[0] for w, p in zip(wts, parts)) / len(parts)
            else:
                loss = sum(p[0] for p in parts) / len(parts)
            if not torch.isfinite(loss): ok = False; break
            (loss / len(idx)).backward()
            pool[i] = (S.detach(), T.detach()); ages[i] = age + L
            lm = float(sum(p[1] for p in parts) / len(parts)); lc = float(sum(p[2] for p in parts) / len(parts))
            info.append(f"{'seed' if age == 0 else age}->{age + L}{'*' if cut_at[i] >= 0 else ''}: {lm:.3f}" + (f"/{lc:.3f}" if has_col(a) else ""))
            tot_loss += float(loss) / len(idx)
        if not ok:
            say(f"it {it}: non-finite loss, back to best, pool reset"); opt.zero_grad()
            model.load_state_dict(best_state); pool = [None] * a.pool; ages = [0] * a.pool; cut_at = [-1] * a.pool
            continue
        gn = 0.0
        for g in opt.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                nrm = float(p.grad.norm()); gn = max(gn, nrm)
                if not math.isfinite(nrm): p.grad.zero_(); continue
                if nrm > 0: p.grad /= (nrm + 1e-12) / a.clip
        opt.step()
        hist.append(tot_loss)
        if it % 10 == 0 or it == iters - 1:
            say(f"it {it:5d} loss {tot_loss:.4f} [{'; '.join(info)}] gn {gn:.1e} maxage {max(ages)} cap {cap(it)} "
                f"lr {opt.param_groups[0]['lr']:.2e} {time.time() - t:.1f}s/it")
        if it % a.eval_every == 0 and it > 0:
            r, su, rd, sc = evaluate(); evals.append((it, r) if rd is None else (it, r, rd))
            tag = ""
            annealing = getattr(a, "spread_to", 0) > 0 and it < a.spread_anneal   # only states at the final spread can be best
            if annealing: tag = " (annealing spread)"
            elif sc < best or (getattr(a, "spread_to", 0) > 0 and best_it < a.spread_anneal):
                best, best_it = sc, it; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; tag = " NEW BEST"
            elif a.rollback and it - best_it > 4 * a.eval_every and sc > best * a.rollback:
                model.load_state_dict(best_state); opt.state.clear(); lr_scale *= 0.5; best_it = it
                pool = [None] * a.pool; ages = [0] * a.pool; cut_at = [-1] * a.pool
                tag = f" -> back to best, lr x{lr_scale:.3g}, pool reset"
            dm = "" if rd is None else f" undamaged {su:.4f} damaged {damage_score(rd):.4f} {rd}"
            say(f"eval it {it}: {r}{dm} score {sc:.4f} (best {best:.4f} at {best_it}){tag}")
        if time.time() - t_ck > a.ckpt_min * 60:
            checkpoint_now(it); t_ck = time.time()
        if it % 100 == 0 and it > it0 and best_state is not None:
            cur_s = {k: v.clone() for k, v in model.state_dict().items()}
            finish(model, a, tgt, total, say, name, best, best_state, it + 1); model.load_state_dict(cur_s)
        if a.hours and time.time() - t_start > a.hours * 3600:
            say(f"time limit at it {it}"); break
    checkpoint_now(it)
    finish(model, a, tgt, total, say, name, best, best_state, it + 1)
    say(f"after (best params, from the seed): {long_eval(model, a, tgt, total)}")
    return best


def train_es(model, a, tgt, total, iters, say, name, ck=None):
    """Antithetic evolution strategy over every trainable tensor, each perturbed relative to its
    own scale, ranks for fitness, Adam on the estimate. It needs no gradient through the chaotic
    unroll, which is the thing that makes the backprop fit noisy. Fitness is deterministic: the
    mean multiscale loss at t0, the middle, t1 and t1 + hold."""
    names = [n for n, p in model.named_parameters() if p.requires_grad and (a.es_seed or not n.startswith("seed_"))]
    prm = dict(model.named_parameters())
    scale = {}
    for n in names:
        p = prm[n].detach()
        if n == "seed_v": scale[n] = a.seed_speed
        elif p.numel() == 1: scale[n] = max(abs(float(p)), 0.1)
        else: scale[n] = max(float(p.pow(2).mean().sqrt()), 0.05)
    flat = lambda: torch.cat([prm[n].detach().reshape(-1) / scale[n] for n in names])
    sizes = [prm[n].numel() for n in names]
    def put(x):
        with torch.no_grad():
            for n, c in zip(names, torch.split(x, sizes)): prm[n].copy_(c.reshape(prm[n].shape) * scale[n])
    tk = sorted({a.t0, (a.t0 + a.t1) // 2, a.t1, a.t1 + a.hold})
    def fit(x):
        put(x)
        with torch.no_grad():
            out, _, _ = model.run(total, max(tk), keep=set(tk), ckpt=False)
        v = sum(float(ms_loss(m, tgt)) for m in out.values()) / len(tk)
        return v if math.isfinite(v) else 1e9
    x = flat(); D = x.numel()
    m1 = torch.zeros(D); m2 = torch.zeros(D); it0 = 0
    best, best_state, hist = 1e9, None, []
    ckp = os.path.join(HERE, f"fca_gecko_{a.tag}.pt")
    gen = torch.Generator().manual_seed(a.seed + 1)
    if ck is not None:
        x = ck["x"]; m1, m2 = ck["m1"], ck["m2"]; it0 = ck["it"] + 1
        best, best_state, hist = ck["best"], ck["state"], ck.get("hist", [])
        gen.manual_seed(a.seed + 1 + it0)
    def checkpoint_now(it):
        put(x)
        save_ckpt(ckp, preset=name, state=best_state, cur={k: v.detach().clone() for k, v in model.state_dict().items()},
                  x=x, m1=m1, m2=m2, it=it, best=best, hist=hist, args=vars(a), kind="es")
    say(f"ES over {D} numbers ({len(names)} tensors), {2 * a.es_pop} runs a generation, steps {tk}")
    t_start = t_ck = time.time(); it = it0 - 1
    for it in range(it0, iters):
        t = time.time()
        f0 = fit(x)
        if f0 < best:
            best = f0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        hist.append(f0)
        eps = torch.randn(a.es_pop, D, generator=gen)
        fp = torch.tensor([fit(x + a.es_sigma * e) for e in eps])
        fm = torch.tensor([fit(x - a.es_sigma * e) for e in eps])
        allf = torch.cat([fp, fm])
        for k, v in enumerate(allf.tolist()):   # a perturbed child can be the best seen
            if v < best:
                e = eps[k % a.es_pop] * (1 if k < a.es_pop else -1)
                put(x + a.es_sigma * e); best = v
                best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        rk = torch.empty(2 * a.es_pop); rk[allf.argsort()] = torch.arange(2 * a.es_pop, dtype=torch.float32)
        rk = rk / (2 * a.es_pop - 1) - 0.5                       # centred ranks, low loss = -0.5
        g = ((rk[:a.es_pop] - rk[a.es_pop:])[:, None] * eps).sum(0) / (a.es_pop * a.es_sigma)
        b1, b2 = 0.9, 0.999
        m1 = b1 * m1 + (1 - b1) * g; m2 = b2 * m2 + (1 - b2) * g * g
        mh = m1 / (1 - b1 ** (it + 1)); vh = m2 / (1 - b2 ** (it + 1))
        x = x - a.es_lr * lr_factor(it, iters) * mh / (vh.sqrt() + 1e-8)
        if it % 5 == 0 or it == iters - 1:
            put(x); P = {k: round(float(v), 4) for k, v in model.P().items()}
            say(f"gen {it:5d} centre {f0:.4f} kids {float(allf.min()):.4f}..{float(allf.median()):.4f} best {best:.4f} "
                f"{time.time() - t:.1f}s/gen  {P}")
        if time.time() - t_ck > a.ckpt_min * 60:
            checkpoint_now(it); t_ck = time.time()
        if it % 50 == 0 and best_state is not None:
            finish(model, a, tgt, total, say, name, best, best_state, it + 1); put(x)
        if a.hours and time.time() - t_start > a.hours * 3600:
            say(f"time limit at gen {it}"); break
    checkpoint_now(it)
    finish(model, a, tgt, total, say, name, best, best_state, it + 1)
    return best


if __name__ == "__main__":
    main()
