# linear/compose3.py  (T03 extension)
# Run from the repo root:  uv run python linear/compose3.py
# Multiplies your T02 rotation, shear and scale in all six orders and prints each product with its determinant.
# There is nothing to edit in this file; it reads MATRICES from linear/landing.py.
import sys
sys.path.insert(0, "linear")                       # so "from chapter_vectors import ..." works from the repo root
from itertools import permutations
import numpy as np
from landing import MATRICES

three = {"R": MATRICES["rotate 90"], "H": MATRICES["shear"], "S": MATRICES["scale"]}
seen = []                                                   # the distinct products found so far
for order in permutations("RHS"):                           # the six orders: RHS, RSH, HRS, HSR, SRH, SHR
    P = three[order[0]] @ three[order[1]] @ three[order[2]]  # applied right to left: the last letter acts first
    new = not any(np.allclose(P, Q) for Q in seen)          # is this product different from every one before it?
    if new:
        seen.append(P)
    print(f"{order[0]} @ {order[1]} @ {order[2]} = {P.tolist()}   det = {np.linalg.det(P):.4f}   {'new' if new else 'same as one above'}")
print(len(seen), "distinct matrices from 6 orders")
