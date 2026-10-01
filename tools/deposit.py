import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

HALF = 0.0015 * 512          # particle half-size in pixels (1024 px grid spans 2 world units)
SIG = 0.163                  # brush.frag gaussian sigma, in quad-uv units
P = 0.938                    # trail persistence

def w(d_px):                 # brush.frag: gaussian(uv-.5, .163)^2, relative to centre
    d_uv = d_px / (2 * HALF)
    return np.exp(-d_uv**2 / (2 * SIG**2))**2

cases = [("on a pixel centre", (1.5, 1.5)), ("on an edge midpoint", (2.0, 1.5)),
         ("on a pixel corner", (2.0, 2.0)), ("somewhere generic", (1.8, 1.65))]
vel = np.array([0.8, 0.35])

fig, axs = plt.subplots(1, 5, figsize=(22, 5), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25]})
for ax, (name, (px, py)) in zip(axs, cases):
    ax.set_xlim(0.5, 3.5); ax.set_ylim(0.5, 3.5); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for i in range(4):
        for j in range(4):
            cx, cy = i + 0.5, j + 0.5
            d = np.hypot(cx - px, cy - py)
            inside = abs(cx - px) < HALF and abs(cy - py) < HALF and d <= HALF
            wt = w(d) if inside else 0.0
            ax.add_patch(Rectangle((i, j), 1, 1, fc=plt.cm.Oranges(0.15 + 0.75*float(wt)) if wt > 0 else "white",
                                   ec="0.6", lw=1))
            ax.plot(cx, cy, "+", color="0.4", ms=8)
            if inside:
                ax.text(cx, cy - 0.28, f"w = {wt:.4f}", ha="center", fontsize=10, weight="bold")
                ax.annotate("", (cx + vel[0] * 0.5 * wt, cy + vel[1] * 0.5 * wt), (cx, cy),
                            arrowprops=dict(arrowstyle="->", color="crimson", lw=2))
    ax.add_patch(Rectangle((px - HALF, py - HALF), 2 * HALF, 2 * HALF, fill=False, ls="--", ec="navy"))
    ax.add_patch(Circle((px, py), HALF, fill=False, ec="navy", lw=1.5))
    ax.plot(px, py, "o", color="navy", ms=7)
    ax.annotate("", (px + vel[0] * 0.5, py + vel[1] * 0.5), (px, py),
                arrowprops=dict(arrowstyle="->", color="navy", lw=2.5))
    ax.set_title(f"particle {name}", fontsize=12)

ax = axs[4]
d = np.linspace(0, HALF, 300)
ax.plot(d, w(d), color="crimson", lw=2.5)
ax.axvline(0.5, color="0.6", ls=":"); ax.text(0.51, 0.6, "edge midpoint\n(0.5 px)", fontsize=9)
ax.axvline(0.7071, color="0.6", ls=":"); ax.text(0.66, 0.8, "corner\n(0.71 px)", fontsize=9, ha="right")
ax.axvline(HALF, color="navy", ls="--"); ax.text(HALF - 0.01, 0.95, "disk edge 0.77 px\n(cut off)", fontsize=9, ha="right", color="navy")
ax.set_xlabel("distance from particle to pixel centre (pixels)"); ax.set_ylabel("w (relative to dead centre)")
ax.set_title("the weight w vs distance", fontsize=12)
ax.set_xlim(0, 0.85); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)

fig.suptitle("Fluoddity deposit:   pixel_vel  +=  particle_vel × (1 − persistence) × w × const      "
             "(navy: particle, its velocity, its 1.54-px square and 0.77-px disk;  red: what each pixel receives, drawn to scale;  + pixel centres)",
             fontsize=12)
fig.tight_layout()
fig.savefig("deposit.png", dpi=110)
