import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
dist, spread, sig = 2.5, 162, 1.0
a = np.radians(spread)/2
eyes = [("left eye kernel", (np.cos(a)*dist, np.sin(a)*dist)), ("right eye kernel", (np.cos(a)*dist, -np.sin(a)*dist))]
R = 6; xs = np.arange(-R, R+1); X, Y = np.meshgrid(xs, xs)
fig, axs = plt.subplots(1, 2, figsize=(10, 5.4))
for ax, (title, c) in zip(axs, eyes):
    V = np.exp(-0.5*((X-c[0])**2+(Y-c[1])**2)/sig**2); V /= V.sum()
    ax.imshow(V, cmap="Oranges", origin="lower", extent=(-R-.5, R+.5, -R-.5, R+.5))
    for k in range(-R, R+2): ax.axhline(k-.5, c="k", lw=.3, alpha=.3); ax.axvline(k-.5, c="k", lw=.3, alpha=.3)
    ax.add_patch(plt.Rectangle((-.5,-.5),1,1,fill=False,ec="k",lw=2))
    ax.annotate("", (4.5, 0), (0.6, 0), arrowprops=dict(arrowstyle="->", lw=2)); ax.text(4.0, 0.4, "velocity", fontsize=9)
    ax.plot(*c, "k+", ms=10)
    ax.set_title(f"{title}\nweights sum to 1; L or R = Σ weight × trail", fontsize=11); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Two separate kernels: Gaussian σ={sig} cell, {dist} cells out, ±{spread/2:.0f}° from the velocity; outlined = own cell", fontsize=11)
fig.tight_layout(); fig.savefig("kernel2.png", dpi=100)
