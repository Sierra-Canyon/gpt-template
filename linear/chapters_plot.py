# linear/chapters_plot.py  (T01)
# Run from the repo root:  uv run python linear/chapters_plot.py
# Draws chapter 2 and chapter 7 as arrows, tip to tail, and their sum. Saves linear/chapters.png.
import sys
sys.path.insert(0, "linear")                       # so "from chapter_vectors" works from the repo root
import matplotlib
matplotlib.use("Agg")                              # draw to a file, never open a window
import matplotlib.pyplot as plt
from chapter_vectors import V2, WORD1, WORD2

vA, vB = V2[2], V2[7]
for start, vec, color, label in ((vA * 0, vA, "tab:blue", "chapter 2"),
                                 (vA, vB, "tab:orange", "chapter 7, tip to tail"),
                                 (vA * 0, vA + vB, "tab:green", "sum")):
    # quiver draws an arrow from start in the direction vec; the three keyword arguments stop it rescaling the arrow
    plt.quiver(start[0], start[1], vec[0], vec[1], angles="xy", scale_units="xy", scale=1, color=color, label=label)
top = (vA + vB).max() * 1.1
plt.xlim(0, top); plt.ylim(0, top)
plt.xlabel(f"count of '{WORD1}'"); plt.ylabel(f"count of '{WORD2}'")
plt.legend(); plt.savefig("linear/chapters.png", dpi=120)
print("saved linear/chapters.png   chapter 2", vA, "  chapter 7", vB, "  sum", vA + vB)
