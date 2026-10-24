# linear/transform_viz.py  (T03, T04)
# Run from the repo root with four numbers for one matrix, read across its rows (a b c d is [[a, b], [c, d]]):
#     uv run python linear/transform_viz.py 0 -1 1 0
# or eight numbers for two matrices A then B, to see AB and BA side by side:
#     uv run python linear/transform_viz.py 0 -1 1 0  1 1 0 1
# Draws the grid before (gray) and after (blue), the unit square after (gold), where i-hat lands (green)
# and where j-hat lands (red), and prints the determinant. There is nothing to edit in this file.
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# In: an axes to draw on, a 2x2 matrix, a title.  Out: nothing; draws the before/after picture on the axes.
def draw(ax, M, title):
    M = np.asarray(M, dtype=float)
    for t in np.linspace(-3, 3, 13):                                 # 13 evenly spaced numbers from -3 to 3: one grid line each
        for line in (np.array([[t, -3], [t, 3]]), np.array([[-3, t], [3, t]])):   # a vertical and a horizontal line, as two points
            ax.plot(*line.T, color="lightgray", lw=0.6)              # the grid before. .T splits the points into an x row and a y row; the * hands those two rows to plot as its two arguments
            ax.plot(*(line @ M.T).T, color="steelblue", lw=0.6)      # the same line after M (line @ M.T applies M to each point)
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    ax.fill(*(square @ M.T).T, color="gold", alpha=0.4)              # where the unit square went
    ax.quiver(0, 0, M[0, 0], M[1, 0], angles="xy", scale_units="xy", scale=1, color="green")  # column 0: i-hat lands here
    ax.quiver(0, 0, M[0, 1], M[1, 1], angles="xy", scale_units="xy", scale=1, color="red")    # column 1: j-hat lands here
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal")
    ax.set_title(f"{title}   det = {np.linalg.det(M):.4f}")

if __name__ == "__main__":
    nums = [float(x) for x in sys.argv[1:]]
    if len(nums) == 4:
        M = np.array(nums).reshape(2, 2)
        fig, ax = plt.subplots(figsize=(6, 6))
        draw(ax, M, "M")
        plt.savefig("linear/viz.png", dpi=110)
        print("det =", np.linalg.det(M))
    elif len(nums) == 8:
        A = np.array(nums[:4]).reshape(2, 2)
        B = np.array(nums[4:]).reshape(2, 2)
        fig, axes = plt.subplots(2, 2, figsize=(11, 11))
        draw(axes[0, 0], A, "A")
        draw(axes[0, 1], B, "B")
        draw(axes[1, 0], A @ B, "AB  (B first, then A)")
        draw(axes[1, 1], B @ A, "BA  (A first, then B)")
        plt.savefig("linear/composition.png", dpi=110)
        print("AB =\n", A @ B, "\nBA =\n", B @ A)
        print("det(A) det(B) =", np.linalg.det(A) * np.linalg.det(B),
              "  det(AB) =", np.linalg.det(A @ B), "  det(BA) =", np.linalg.det(B @ A))
    else:
        print("give 4 numbers for one matrix or 8 for two")
