import numpy as np, trimesh, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_die import FACES, SIZE
m = trimesh.load("hopeful_ai_die.stl")
N = 260
fig, axes = plt.subplots(1, 6, figsize=(18, 3.4), facecolor="white")
for ax, (label, n, u) in zip(axes, FACES):
    n, u = np.array(n, float), np.array(u, float); r = np.cross(u, n)
    g = np.linspace(-SIZE/2, SIZE/2, N)
    X, Y = np.meshgrid(g, g[::-1])
    origins = (X[..., None]*r + Y[..., None]*u + n*(SIZE/2 + 5)).reshape(-1, 3)
    dirs = np.tile(-n, (len(origins), 1))
    locs, idx, _ = m.ray.intersects_location(origins, dirs, multiple_hits=False)
    depth = np.full(len(origins), np.nan)
    depth[idx] = (locs - origins[idx]) @ (-n) - 5
    ax.imshow(depth.reshape(N, N), cmap="Blues", vmin=-0.2, vmax=1.6)
    ax.set_title(f"{label}  "); ax.axis("off")
plt.tight_layout(); plt.savefig("faces_preview.png", dpi=100)
