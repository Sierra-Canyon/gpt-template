# micrograd/expr.py  (T08)
# Run from the repo root:  uv run python micrograd/expr.py          (build the expression from your counts and draw it)
#                     or:  uv run python micrograd/expr.py grads    (T08 Step 5: the gradients you set by hand, checked by nudging)
# The expression is T01's two-word dot product of chapter 2 and chapter 7, counts per 1,000 words, built out of
# YOUR Value objects from micrograd/engine.py, then scaled by s. Nothing to edit in this file.
import sys
sys.path.insert(0, "linear")
sys.path.insert(0, "micrograd")
from chapter_vectors import V2, size, WORD1, WORD2
from engine import Value
from viz import draw_dot, print_graph

per_1000 = V2 / size * 1000                        # (10, 2): each chapter's two counts per 1,000 words
a1, a2 = [round(float(v), 1) for v in per_1000[2]]   # chapter 2: WORD1 then WORD2, to one decimal so the hand page is doable
b1, b2 = [round(float(v), 1) for v in per_1000[7]]   # chapter 7
print(f"chapter 2 per 1,000 words: {WORD1} {a1}  {WORD2} {a2}")
print(f"chapter 7 per 1,000 words: {WORD1} {b1}  {WORD2} {b2}")

A1, A2 = Value(a1, label="a1"), Value(a2, label="a2")
B1, B2 = Value(b1, label="b1"), Value(b2, label="b2")
S = Value(0.01, label="s")                         # a scale, so the chain rule has a product to pass through at the top
p1 = A1 * B1; p1.label = "p1"                      # the WORD1 product
p2 = A2 * B2; p2.label = "p2"                      # the WORD2 product
d = p1 + p2;  d.label = "d"                        # the two-word dot product
L = d * S;    L.label = "L"
print(f"p1 = {p1.data:.2f}   p2 = {p2.data:.2f}   d = {d.data:.2f}   L = {L.data:.4f}")

if len(sys.argv) > 1 and sys.argv[1] == "grads":
    # Step 5: the gradients, set by hand from the output back, exactly as he did in the video.
    L.grad = 1.0
    d.grad = S.data * L.grad;  S.grad = d.data * L.grad         # L = d * s
    p1.grad = d.grad;          p2.grad = d.grad                 # d = p1 + p2: a plus passes the gradient through
    A1.grad = B1.data * p1.grad;  B1.grad = A1.data * p1.grad   # p1 = a1 * b1
    A2.grad = B2.data * p2.grad;  B2.grad = A2.data * p2.grad   # p2 = a2 * b2

    # In: how much to add to each of the five leaves.  Out: L recomputed with plain floats.
    def L_with(da1=0.0, da2=0.0, db1=0.0, db2=0.0, ds=0.0):
        return ((a1 + da1) * (b1 + db1) + (a2 + da2) * (b2 + db2)) * (0.01 + ds)

    h = 1e-6
    for name, v, kw in (("a1", A1, "da1"), ("a2", A2, "da2"), ("b1", B1, "db1"), ("b2", B2, "db2"), ("s", S, "ds")):
        nudge = (L_with(**{kw: h}) - L_with()) / h
        print(f"dL/d{name}  by hand {v.grad: 10.4f}   by nudging {nudge: 10.4f}")
    draw_dot(L).render("micrograd/expr-grads", format="svg", cleanup=True)
    print("drawn: micrograd/expr-grads.svg")
else:
    print_graph(L)
    try:
        draw_dot(L).render("micrograd/expr", format="svg", cleanup=True)
        print("drawn: micrograd/expr.svg")
    except Exception as e:                         # the dot program is missing: the text graph above is the fallback
        print("could not draw (is graphviz installed with brew?):", type(e).__name__)
