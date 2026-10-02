"""Train the paint layer (--paint K) on the frozen dynamics of a trained checkpoint.

The paint never changes the dynamics, so the dynamics' trajectory from the seed is recorded
once (what the paint reads each step and where the move sends each cell) and the paint is
fitted on that record: an unroll costs only the paint head, so the paint can be trained to
long horizons. The result is an ordinary checkpoint of fluoddity_ca_gecko.py (dynamics +
paint), so every helper (gif, ablate) works on it.

  python3 fca_gecko_paint.py --dyn s1 --tag s1_paint --paint 4 --tmax 200 --iters 2000
"""
import argparse, copy, math, os, sys, time
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fluoddity_ca_gecko as f
from fca_gecko_ablate import build

PAINT = ("pw1", "pb1", "pw2", "pb2", "pseed")

ap = argparse.ArgumentParser()
ap.add_argument("--dyn", required=True, help="tag of the dynamics checkpoint")
ap.add_argument("--tag", required=True)
ap.add_argument("--paint", type=int, default=4)
ap.add_argument("--paint_w", type=int, default=64)
ap.add_argument("--init", default="", help="tag of a paint checkpoint to start the paint from")
ap.add_argument("--tmax", type=int, default=200)
ap.add_argument("--t0", type=int, default=48)
ap.add_argument("--every", type=int, default=8)
ap.add_argument("--iters", type=int, default=2000)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--wpix", type=float, default=0.5)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--hours", type=float, default=0)
ap.add_argument("--resume", default="auto")
ap.add_argument("--ckpt_min", type=float, default=10.0)
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
torch.set_num_threads(A.threads); torch.manual_seed(A.seed)
log = open(os.path.join(HERE, f"fca_gecko_{A.tag}.log"), "a")
def say(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()
say(f"== paint {vars(A)}")

m0, a0 = build(A.dyn)
ckd = torch.load(os.path.join(HERE, f"fca_gecko_{A.dyn}.pt"), weights_only=False)
a = copy.deepcopy(a0); a.paint, a.paint_w, a.tag, a.freeze_dyn, a.wpix = A.paint, A.paint_w, A.tag, True, A.wpix
pre = {p["name"]: p for p in f.load_presets()}[ckd["preset"]]
model = f.FCA(a, pre)
own = model.state_dict()
for k, v in m0.state_dict().items():
    if k in own and k not in PAINT: own[k] = v
ckp = os.path.join(HERE, f"fca_gecko_{A.tag}.pt")
ck = torch.load(ckp, weights_only=False) if A.resume == "auto" and os.path.exists(ckp) else None
if ck is not None:
    own.update({k: ck["cur"][k] for k in PAINT}); say(f"resumed at it {ck['it']} best {ck['best']:.4f}")
elif A.init:
    src = torch.load(os.path.join(HERE, f"fca_gecko_{A.init}.pt"), weights_only=False)["state"]
    own.update({k: src[k] for k in PAINT if k in src and src[k].shape == own[k].shape}); say(f"paint from {A.init}")
model.load_state_dict(own)
for nm, p in model.named_parameters(): p.requires_grad_(nm in PAINT)
n, Pc, nt = model.N, model.Pc, model.nt
tgt = f.load_target(a); total = float(tgt.sum()); trgb = f.load_target_rgb(a)
trgb_v = f.view(a, trgb)   # --zoom: scored on z x z averages

# ---- record the dynamics
t = time.time()
model.rec = []
with torch.no_grad():
    S, T = model.init_state(total); M0 = S[0].clone()
    for s in range(A.tmax):
        S, T = model.step(S, T); model.rec[-1]["m"] = S[0].clone()
rec, model.rec = model.rec, None
mb = sum(sum(v.numel() * v.element_size() for v in d.values() if torch.is_tensor(v)) for d in rec) / 2 ** 20
say(f"recorded {A.tmax} steps in {time.time() - t:.0f}s, {mb:.0f} MB; active cells at the end {rec[-1]['ia'].numel()}")

def pstep(P, k):
    d = rec[k]
    Q = model.paint_core(P, d)
    vals = (Q[:, None] * d["w"].reshape(nt * nt, -1)[None]).reshape(Pc, -1)
    return torch.zeros(Pc, n * n).index_add(1, d["tgt"], vals)

def render(P, m):
    return ((m[None] + P[:3]) * 0.5).reshape(3, n, n)

keep = [s for s in range(A.t0, A.tmax + 1) if s % A.every == 0 or s == A.tmax]
def unroll():
    pp = torch.tanh(F.pad(model.pseed, (model.s0, n - model.s1, model.s0, n - model.s1))).reshape(Pc, -1)
    P = M0[None] * pp
    out = {}
    for s in range(1, A.tmax + 1):
        P = checkpoint(pstep, P, s - 1, use_reentrant=False)
        if s in keep: out[s] = P
    return out

params = [p for nm, p in model.named_parameters() if nm in PAINT]
opt = torch.optim.Adam(params, lr=A.lr)
best, best_state, it0, hist = 1e9, None, 0, []
if ck is not None:
    opt.load_state_dict(ck["opt"]); best, best_state, it0, hist = ck["best"], ck["state"], ck["it"] + 1, ck["hist"]

def save(it):
    f.save_ckpt(ckp, preset=ckd["preset"], state=best_state, cur={k: v.detach().clone() for k, v in model.state_dict().items()},
                opt=opt.state_dict(), it=it, best=best, hist=hist, args=vars(a), kind="paint")

def finish(it):
    cur = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    r = f.long_eval(model, a, tgt, total)
    f.save_params(model, os.path.join(HERE, f"fca_gecko_{A.tag}.json"),
                  {"preset": ckd["preset"], "best": best, "iters": it, "args": vars(a), "long_eval": {str(k): v for k, v in r.items()}})
    say(f"best paint, from the seed: {r}")
    say("pictures:", *f.save_pictures(model, tgt, total, a, A.tag))
    model.load_state_dict(cur)

t_start = t_ck = time.time(); it = it0 - 1
for it in range(it0, A.iters):
    t = time.time()
    for g in opt.param_groups: g["lr"] = A.lr * f.lr_factor(it, A.iters)
    out = unroll()
    parts = []
    for s, P in out.items():
        R = f.view(a, render(P, rec[s - 1]["m"]))
        parts.append((f.ms_loss(R, trgb_v), f.pix_loss(R, trgb_v)))
    loss = sum(c + A.wpix * p for c, p in parts) / len(parts)
    lc = float(sum(c for c, _ in parts)) / len(parts)
    opt.zero_grad(); loss.backward()
    gn = 0.0
    for p in params:   # same per-tensor normalisation as the main fit
        if p.grad is None: continue
        nrm = float(p.grad.norm()); gn = max(gn, nrm)
        if math.isfinite(nrm) and nrm > 0: p.grad /= nrm + 1e-12
        elif not math.isfinite(nrm): p.grad.zero_()
    opt.step()
    hist.append(lc)
    if lc < best:
        best = lc; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if it % 10 == 0:
        per = {s: round(float(c), 3) for (s, _), (c, _) in zip(out.items(), parts) if s in (48, 96, 160, A.tmax)}
        say(f"it {it:5d} colour {lc:.4f} best {best:.4f} {per} gn {gn:.1e} {time.time() - t:.1f}s/it")
    if time.time() - t_ck > A.ckpt_min * 60: save(it); t_ck = time.time()
    if it % 250 == 0 and it > it0: finish(it)
    if A.hours and time.time() - t_start > A.hours * 3600: say(f"time limit at it {it}"); break
save(it); finish(it + 1)
