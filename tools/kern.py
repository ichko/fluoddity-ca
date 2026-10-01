import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
dist, spread, size, feather = 2.5, 162, 1.0, 0.2          # brains.html defaults
sig = max(1.0, size*(0.6+0.9*feather))
a = np.radians(spread)/2
R_ = (np.cos(a)*dist,  np.sin(a)*dist); L_ = (np.cos(a)*dist, -np.sin(a)*dist)
shapes = [("shape 0: opposed\none reading = right − left", [(R_,1),(L_,-1)]),
          ("shape 1: both eyes (default)\ntwo readings, right and left", [(R_,1),(L_,1)]),
          ("shape 2: eyes and centre\nthree readings", [(R_,1),(L_,1),((0,0),-1)])]
R = 6; xs = np.arange(-R, R+1)
fig, axs = plt.subplots(1, 3, figsize=(15, 5.6))
for ax, (title, lobes) in zip(axs, shapes):
    X, Y = np.meshgrid(xs, xs)
    V = sum(w*np.exp(-0.5*((X-c[0])**2+(Y-c[1])**2)/sig**2) for c, w in lobes)
    m = np.abs(V).max()
    ax.imshow(V/m, cmap="coolwarm", vmin=-1, vmax=1, origin="lower", extent=(-R-.5, R+.5, -R-.5, R+.5))
    for k in range(-R, R+2): ax.axhline(k-.5, c="k", lw=.3, alpha=.3); ax.axvline(k-.5, c="k", lw=.3, alpha=.3)
    ax.add_patch(plt.Rectangle((-.5,-.5),1,1,fill=False,ec="k",lw=2))
    ax.annotate("", (4.5, 0), (0.6, 0), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(4.6, 0.3, "heading\n(velocity)", fontsize=9)
    for (c, w), nm in zip(lobes, ["right eye", "left eye", "centre"]):
        ax.plot(*c, "k+", ms=10); ax.text(c[0]+.2, c[1]+(.5 if c[1]>=0 else -.9), nm, fontsize=9)
    ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"brains.html kernel (defaults: eyes {dist} cells out, ±{spread/2:.0f}° from heading, Gaussian blob per eye; red +, blue −; outlined = own cell)", fontsize=11)
fig.tight_layout(); fig.savefig("kernel.png", dpi=100)
