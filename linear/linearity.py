# linear/linearity.py  (T02 extension)
# Run from the repo root:  uv run python linear/linearity.py
# Four things people do to word counts before a model sees them. Which of them are linear?
# A function f is linear when f(u + v) = f(u) + f(v) and f(3u) = 3 f(u); then a matrix does it,
# and the matrix's columns are f(i-hat) and f(j-hat). There is nothing to edit in this file.
import sys
sys.path.insert(0, "linear")
import numpy as np
from chapter_vectors import V2, size

vA, vB = V2[2], V2[7]
mean = V2.mean(axis=0)                               # the average chapter vector

candidates = {
    "per 1000 words":       lambda v: v * 1000 / size,                    # counts as a rate
    "total and difference": lambda v: np.array([v[0] + v[1], v[0] - v[1]]),
    "centered":             lambda v: v - mean,                           # subtract the average chapter
    "log counts":           lambda v: np.log1p(v),                        # log(1 + count), each coordinate
}

e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
print("mean chapter vector:", mean)
print(f"{'function':22} {'add gap':>10} {'scale gap':>10} {'matrix gap':>11}")
for name, f in candidates.items():
    add_gap = np.abs(f(vA + vB) - (f(vA) + f(vB))).max()      # f(u + v) against f(u) + f(v)
    scale_gap = np.abs(f(3 * vA) - 3 * f(vA)).max()           # f(3u) against 3 f(u)
    M = np.column_stack([f(e1), f(e2)])                        # the matrix IF f were linear
    matrix_gap = np.abs(M @ vA - f(vA)).max()                  # does that matrix reproduce f?
    print(f"{name:22} {add_gap:10.1f} {scale_gap:10.1f} {matrix_gap:11.1f}")
    print(f"{'':22} matrix from where i and j land: {M.round(4).tolist()}")
