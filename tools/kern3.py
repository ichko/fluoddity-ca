import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
dist, ang, sig = 2.56, np.radians(81), 0.4
eyes = [("left eye", (np.cos(ang)*dist, np.sin(ang)*dist)), ("right eye", (np.cos(ang)*dist, -np.sin(ang)*dist))]
R = 4; xs = np.arange(-R, R+1); X, Y = np.meshgrid(xs, xs)
def bilinear(c):
    return np.clip(1-np.abs(X-c[0]),0,None)*np.clip(1-np.abs(Y-c[1]),0,None)
def gauss(c):
    V = np.exp(-0.5*((X-c[0])**2+(Y-c[1])**2)/sig**2); return V/V.sum()
fig, axs = plt.subplots(2, 2, figsize=(9, 9))
for r, (kind, f) in enumerate([("Fluoddity exactly: one bilinear sample", bilinear), (f"as a Gaussian kernel, σ = {sig} cells", gauss)]):
    for ccol, (name, c) in enumerate(eyes):
        ax = axs[r, ccol]; V = f(c)
        ax.imshow(V, cmap="Oranges", vmin=0, vmax=V.max(), origin="lower", extent=(-R-.5, R+.5, -R-.5, R+.5))
        for k in range(-R, R+2): ax.axhline(k-.5, c="k", lw=.3, alpha=.3); ax.axvline(k-.5, c="k", lw=.3, alpha=.3)
        for (x, y), v in np.ndenumerate(V.T):
            if V.T[x, y] > 0.02: ax.text(xs[x], xs[y], f"{V.T[x,y]:.2f}", ha="center", va="center", fontsize=8)
        ax.add_patch(plt.Rectangle((-.5,-.5),1,1,fill=False,ec="k",lw=2))
        ax.annotate("", (3.6, 0), (0.6, 0), arrowprops=dict(arrowstyle="->", lw=2)); ax.text(2.6, 0.3, "velocity", fontsize=9)
        ax.plot(*c, "b+", ms=12, mew=2)
        ax.set_title(f"{name} — {kind}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Eye kernels at Fluoddity defaults: 2.56 cells out, ±81° from the velocity  (+ = eye position, outlined = own cell)", fontsize=11)
fig.tight_layout(); fig.savefig("kernel3.png", dpi=100)
