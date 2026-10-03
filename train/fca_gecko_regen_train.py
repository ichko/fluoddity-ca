"""Regeneration fine-tune without a pool: every iteration
  (a) grows from the seed to L in [t0, t1+hold] and scores it every 8 steps from t0 (the
      growth the model already has, kept), and
  (b) grows a gecko to a random age in [--t_dmg] without gradient, cuts a random disc on the
      body (fluoddity_ca_gecko.random_disc_cut, every per-cell number and the trail under it
      zeroed, the mass given back by --reinject), then runs --seg steps WITH gradient and
      scores every 4th frame against the whole target.
The pool versions (train_pool --damage) drift: their grown states are the model's own, so
growth and regrowth degrade together. Here every damaged state is the current model's own
gecko at a known age. Selection: the best (undamaged + wdmg * damaged) / (1 + wdmg), the
same long evaluation as the pool runs. Checkpoints every --ckpt_min minutes, resumable; the
checkpoint is an ordinary fluoddity_ca_gecko.py checkpoint (build(tag) loads it).

  python3 fca_gecko_regen_train.py --tag rgR --init fca_gecko_dynD.pt --iters 800 --hours 6
"""
import argparse, math, os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fluoddity_ca_gecko as f

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--init", default="fca_gecko_dynD.pt")
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--lr_brake", type=float, default=2e-2)
ap.add_argument("--lr_seed", type=float, default=1e-2)
ap.add_argument("--iters", type=int, default=800)
ap.add_argument("--hours", type=float, default=0)
ap.add_argument("--t_dmg", default="120,320")
ap.add_argument("--seg", default="48,96")
ap.add_argument("--damage_r", default="8,22")
ap.add_argument("--reinject", default="edge")
ap.add_argument("--wgrow", type=float, default=1.0, help="weight of the growth loss (a) against the regrowth loss (b)")
ap.add_argument("--state_noise", type=float, default=0.0)
ap.add_argument("--wdmg", type=float, default=1.0)
ap.add_argument("--eval_every", type=int, default=25)
ap.add_argument("--ckpt_min", type=float, default=10.0)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
torch.set_num_threads(A.threads); torch.manual_seed(A.seed)
log = open(os.path.join(HERE, f"fca_gecko_{A.tag}.log"), "a")
def say(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()
say(f"== regen_train {vars(A)}")

ckp = os.path.join(HERE, f"fca_gecko_{A.tag}.pt")
resume = os.path.exists(ckp)
src = torch.load(ckp if resume else os.path.join(HERE, A.init) if not os.path.isabs(A.init) else A.init, weights_only=False)
a = argparse.Namespace(**src["args"])
for k, v in dict(press_mode=0, paint=-1, paint_w=64, rgb=False, colour=False, zoom=1).items():
    if not hasattr(a, k): setattr(a, k, v)
a.tag = A.tag; a.reinject = A.reinject; a.damage_r = A.damage_r; a.state_noise = A.state_noise; a.wdmg = A.wdmg
a.lr = A.lr; a.regen_train = vars(A)
pre = {p["name"]: p for p in f.load_presets()}[src["preset"]]
model = f.FCA(a, pre)
model.load_state_dict(src["cur"] if resume else src["state"])
tgt = f.load_target(a); total = float(tgt.sum())
trgb = f.load_target_rgb(a) if f.has_col(a) else None
seedp = [p for n, p in model.named_parameters() if n.startswith("seed_") and p.requires_grad]
brk = [p for n, p in model.named_parameters() if n in ("wb", "bb") and p.requires_grad]
rest = [p for n, p in model.named_parameters() if not n.startswith("seed_") and n not in ("wb", "bb") and p.requires_grad]
groups = [{"params": rest, "lr": A.lr}, {"params": seedp, "lr": A.lr_seed * a.seed_speed}]
if brk: groups.append({"params": brk, "lr": A.lr_brake})
opt = torch.optim.Adam(groups); base_lr = [g["lr"] for g in opt.param_groups]
gen = torch.Generator().manual_seed(A.seed + 5)
tdm = [int(x) for x in A.t_dmg.split(",")]; sg = [int(x) for x in A.seg.split(",")]; dr = [float(x) for x in A.damage_r.split(",")]

def evaluate():
    r = f.long_eval(model, a, tgt, total); su = f.eval_score(r)
    rd = f.damage_eval(model, a, tgt, total, reinject=A.reinject); sd = f.damage_score(rd)
    return r, su, rd, sd, (su + A.wdmg * sd) / (1 + A.wdmg)

it0, hist, evals = 0, [], []
if resume:
    opt.load_state_dict(src["opt"]); it0 = src["it"] + 1; hist = src.get("hist", []); evals = src.get("evals", [])
    best, best_state, best_it = src["best"], src["state"], src.get("best_it", 0)
    if "gen" in src: gen.set_state(src["gen"])
    if "rng" in src: torch.set_rng_state(src["rng"])
    say(f"resumed at it {src['it']} best {best:.4f}")
else:
    r, su, rd, sd, best = evaluate(); best_it = 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    evals.append((0, r, rd))
    say(f"from {A.init}: undamaged {r} {su:.4f}; damaged ({A.reinject}) {rd} {sd:.4f}; score {best:.4f}")

def checkpoint_now(it):
    f.save_ckpt(ckp, preset=src["preset"], state=best_state, cur={k: v.detach().clone() for k, v in model.state_dict().items()},
                opt=opt.state_dict(), it=it, best=best, best_it=best_it, hist=hist, evals=evals, args=vars(a),
                rng=torch.get_rng_state(), gen=gen.get_state(), kind="regen")

def finish(it):
    cur = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    r = f.long_eval(model, a, tgt, total)
    f.save_params(model, os.path.join(HERE, f"fca_gecko_{A.tag}.json"), {"preset": src["preset"], "best": best, "iters": it,
                  "args": vars(a), "long_eval": {str(k): v for k, v in r.items()}})
    say("pictures:", *f.save_pictures(model, tgt, total, a, A.tag))
    model.load_state_dict(cur)

t_start = t_ck = time.time(); it = it0 - 1
for it in range(it0, A.iters):
    t = time.time()
    for g, b in zip(opt.param_groups, base_lr): g["lr"] = b * f.lr_factor(it, A.iters)
    opt.zero_grad(); ok = True
    # (a) growth from the seed
    L = int(torch.randint(a.t0, a.t1 + a.hold + 1, (1,), generator=gen))
    keep = {k for k in range(a.t0, L + 1) if k % 8 == 0 or k == L}
    out, _, _ = model.run(total, L, keep=keep, full=True)
    lg = sum(f.frame_loss(model, St, tgt, trgb, a)[0] for St in out.values()) / len(out)
    if torch.isfinite(lg): (A.wgrow * lg / (1 + A.wgrow)).backward()
    else: ok = False
    # (b) regrowth after a cut
    td = int(torch.randint(tdm[0], tdm[1] + 1, (1,), generator=gen))
    with torch.no_grad():
        _, S, T = model.run(total, td, ckpt=False)
        cut, (cx, cy, r) = f.random_disc_cut(model, S, dr[0], dr[1], gen)
        fr = float((S[0] * cut).sum()) / total
        S, T = f.damage_state(model, S, T, cut, total, A.reinject)
        if A.state_noise > 0:
            mm = S[0].clamp_min(1e-12); sn = A.state_noise
            S[1:3] = S[1:3] * (1 + sn * torch.randn(2, S.shape[1], generator=gen))
            if model.Hd:
                hh = (S[3:3 + model.Hd] / mm + sn * torch.randn(model.Hd, S.shape[1], generator=gen)).clamp(-0.99, 0.99)
                S[3:3 + model.Hd] = hh * S[0]
    Ls = int(torch.randint(sg[0], sg[1] + 1, (1,), generator=gen))
    keep = {k for k in range(4, Ls + 1, 4)} | {Ls}
    out, _, _ = model.run(total, Ls, keep=keep, full=True, state=(S, T))
    ld = sum(f.frame_loss(model, St, tgt, trgb, a)[0] for St in out.values()) / len(out)
    if torch.isfinite(ld): (ld / (1 + A.wgrow)).backward()
    else: ok = False
    gn = 0.0
    for g in opt.param_groups:
        for p in g["params"]:
            if p.grad is None: continue
            nrm = float(p.grad.norm()); gn = max(gn, nrm)
            if not math.isfinite(nrm): p.grad.zero_(); continue
            if nrm > 0: p.grad /= nrm + 1e-12
    if ok: opt.step()
    hist.append((float(lg), float(ld)))
    if it % 10 == 0:
        say(f"it {it:5d} grow seed->{L}: {float(lg):.4f}  regrow {td}+cut r{r:.0f} {fr:.0%} ->{Ls}: {float(ld):.4f}  gn {gn:.1e} "
            f"lr {opt.param_groups[0]['lr']:.2e} {time.time() - t:.1f}s/it")
    if it % A.eval_every == 0 and it > it0:
        r_, su, rd, sd, sc = evaluate(); evals.append((it, r_, rd)); tag = ""
        if sc < best:
            best, best_it = sc, it; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; tag = " NEW BEST"
        say(f"eval it {it}: {r_} undamaged {su:.4f} damaged {sd:.4f} {rd} score {sc:.4f} (best {best:.4f} at {best_it}){tag}")
    if time.time() - t_ck > A.ckpt_min * 60:
        checkpoint_now(it); t_ck = time.time()
    if it % 100 == 0 and it > it0: finish(it + 1)
    if A.hours and time.time() - t_start > A.hours * 3600:
        say(f"time limit at it {it}"); break
checkpoint_now(it); finish(it + 1)
