# linear/dependent.py  (T04 extension)
# Run from the repo root:  uv run python linear/dependent.py
# Scales each chapter vector to length 1 and, for all 45 pairs, prints |det| of the 2x2 matrix with the two
# as columns. That number is the area of the parallelogram they span, which for unit vectors is the sine of
# the angle between them: 0 means parallel (dependent), 1 means perpendicular. Draws the closest pair.
# There is nothing to edit in this file.
import sys
sys.path.insert(0, "linear")                       # so "from chapter_vectors import ..." works from the repo root
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chapter_vectors import V2
from transform_viz import draw

U = V2 / np.linalg.norm(V2, axis=1)[:, None]            # each chapter scaled to length 1
pairs = []                                               # (|det|, i, j) for every pair
for i in range(10):
    for j in range(i + 1, 10):
        d = abs(np.linalg.det(np.column_stack([U[i], U[j]])))
        pairs.append((d, i, j))
pairs.sort()                                             # smallest first
for d, i, j in pairs:
    tag = "  <-- chapters 2 and 7" if (i, j) == (2, 7) else ""
    print(f"chapters {i},{j}  |det| = {d:.5f}{tag}")
print(f"smallest: chapters {pairs[0][1]},{pairs[0][2]} at {pairs[0][0]:.5f}   largest: chapters {pairs[-1][1]},{pairs[-1][2]} at {pairs[-1][0]:.5f}")

d, i, j = pairs[0]
fig, ax = plt.subplots(figsize=(6, 6))
draw(ax, np.column_stack([U[i], U[j]]), f"chapters {i} and {j} as columns")
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
plt.savefig("linear/dependent.png", dpi=110)
print("saved linear/dependent.png")
