# linear/landing.py  (T02)
# Run from the repo root:  uv run python linear/landing.py
# Checks the three matrices from your hand page (where i-hat and j-hat land are the columns), then
# sends all ten chapter vectors through each one and draws before/after. The lines you edit are the
# three np.array(...) rows under MATRICES.
import sys
sys.path.insert(0, "linear")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chapter_vectors import V2, WORD1, WORD2

c = np.cos(np.pi / 4)
MATRICES = {
    "rotate 90": np.array([[0.0, 0.0], [0.0, 0.0]]),   # fill in from your page: rows of the matrix, as NumPy writes it
    "shear":     np.array([[0.0, 0.0], [0.0, 0.0]]),
    "scale":     np.array([[0.0, 0.0], [0.0, 0.0]]),
    "rotate 45": np.array([[c, -c], [c, c]]),          # given, not on your page
}

# In: two vectors.  Out: the angle between them in degrees.
def angle(a, b):
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))   # @ is the dot product
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))        # clip keeps rounding from pushing cos past 1

if __name__ == "__main__":
    e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])     # i-hat and j-hat
    U = V2 / np.linalg.norm(V2, axis=1)[:, None]            # each chapter scaled to length 1, still (10, 2). [:, None] makes the ten lengths a column so each row is divided by its own length
    ratio = V2[:, 1] / V2[:, 0]                             # WORD2 count over WORD1 count, per chapter
    lo, hi = ratio.argmin(), ratio.argmax()                 # the two chapters whose arrows point furthest apart

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (name, M) in zip(axes, MATRICES.items()):
        print(f"{name:10} i lands at {M @ e1}   j lands at {M @ e2}")
        moved = U @ M.T                                     # every chapter through M at once: row v becomes M @ v
        v, mv = U[2], moved[2]
        print(f"{'':10} chapter 2: L2 {np.linalg.norm(v):.4f} -> {np.linalg.norm(mv):.4f}"
              f"   L1 {np.abs(v).sum():.4f} -> {np.abs(mv).sum():.4f}"
              f"   angle(ch{lo}, ch{hi}) {angle(U[lo], U[hi]):.3f} -> {angle(moved[lo], moved[hi]):.3f} degrees")
        for row, color in ((U, "lightgray"), (moved, "tab:blue")):
            ax.quiver(np.zeros(10), np.zeros(10), row[:, 0], row[:, 1], angles="xy", scale_units="xy", scale=1, color=color)
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect("equal"); ax.grid(True)
        ax.set_title(f"{name}  (gray: {WORD1}/{WORD2} chapters before)")
    plt.savefig("linear/transforms.png", dpi=110)

    # M @ v is a combination of the columns: v[0] copies of column 0 plus v[1] copies of column 1.
    M = MATRICES["shear"]
    v = V2[2]
    print("M @ v                    ", M @ v)
    print("v[0]*col0 + v[1]*col1    ", v[0] * M[:, 0] + v[1] * M[:, 1])
